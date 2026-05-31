"""FastAPI application serving FER2013 CustomCNN inference and training artifacts."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, Dict, List

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from fer_project.config import CFG

from backend.inference import InferenceEngine
from backend.schemas import (
    Base64ImageRequest,
    HealthResponse,
    HistoryResponse,
    ModelInfoResponse,
    PredictionResponse,
    RootHealthResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("fer_api")

SERVICE_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = Path(
    os.environ.get(
        "FER_CHECKPOINT_PATH",
        str(SERVICE_ROOT / "fer_project" / "outputs" / "checkpoints" / "best_model.pth"),
    )
)
HISTORY_PATH = SERVICE_ROOT / "fer_project" / "outputs" / "results" / "history.json"
PLOTS_DIR = SERVICE_ROOT / "fer_project" / "outputs" / "plots"
ALLOWED_PLOTS = frozenset(
    {"confusion_matrix", "training_curves", "misclassified", "gradcam_samples"}
)

START_MONO = time.monotonic()
_engine: InferenceEngine | None = None
_cached_history: Dict[str, Any] | None = None
_cached_val_accuracy: float = 0.611
_model_load_error: str | None = None

_MIN_CHECKPOINT_BYTES = 1_000_000
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _checkpoint_bytes(path: Path = CHECKPOINT_PATH) -> int | None:
    """Return checkpoint size on disk, or None if the file is absent."""
    return path.stat().st_size if path.is_file() else None


def _space_repo_id() -> str | None:
    """Build ``author/name`` for the running Hugging Face Space, if available."""
    author = os.environ.get("SPACE_AUTHOR_NAME")
    repo = os.environ.get("SPACE_REPO_NAME")
    if author and repo:
        return f"{author}/{repo}"
    space_id = os.environ.get("SPACE_ID")
    return space_id if space_id and "/" in space_id else None


def _download_checkpoint_from_hub(path: Path) -> None:
    """Fetch ``best_model.pth`` from the Space repo when the local copy is an LFS stub."""
    repo_id = _space_repo_id()
    if not repo_id:
        LOGGER.warning("Cannot download checkpoint: SPACE_AUTHOR_NAME / SPACE_REPO_NAME not set.")
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        LOGGER.error("huggingface_hub is required to download checkpoints: %s", exc)
        return

    rel = "fer_project/outputs/checkpoints/best_model.pth"
    LOGGER.info("Downloading checkpoint from Space repo %s …", repo_id)
    hf_hub_download(
        repo_id=repo_id,
        repo_type="space",
        filename=rel,
        local_dir=str(SERVICE_ROOT),
    )
    if not path.is_file() or path.stat().st_size < _MIN_CHECKPOINT_BYTES:
        LOGGER.error("Checkpoint download did not produce a valid file at %s", path)


def _resolve_checkpoint(path: Path = CHECKPOINT_PATH) -> Path:
    """Ensure the checkpoint file exists and is a real weights file, not an LFS pointer."""
    if path.is_file():
        size = path.stat().st_size
        if size >= _MIN_CHECKPOINT_BYTES:
            head = path.read_bytes()[:64]
            if not head.startswith(_LFS_POINTER_PREFIX):
                return path
            LOGGER.warning("Checkpoint at %s is a Git LFS pointer (%s bytes).", path, size)
        else:
            LOGGER.warning("Checkpoint at %s is too small (%s bytes).", path, size)
    _download_checkpoint_from_hub(path)
    return path


def _uptime_seconds() -> float:
    """Return process uptime in seconds for health probes."""
    return round(time.monotonic() - START_MONO, 3)


def get_engine() -> InferenceEngine:
    """Return the singleton inference engine or raise 503."""
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail=_model_load_error or "Model is not loaded yet.",
        )
    return _engine


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the PyTorch model once at startup."""
    global _engine, _cached_history, _cached_val_accuracy, _model_load_error
    _model_load_error = None
    device = os.environ.get("MODEL_DEVICE", "cpu")
    checkpoint = _resolve_checkpoint()
    size = _checkpoint_bytes(checkpoint)
    LOGGER.info("Checkpoint path=%s size=%s", checkpoint, size)
    if not checkpoint.is_file() or size is None or size < _MIN_CHECKPOINT_BYTES:
        _model_load_error = (
            f"Checkpoint missing or invalid at {checkpoint} (size={size}). "
            "Ensure fer_project/outputs/checkpoints/best_model.pth is pushed with Git LFS."
        )
        LOGGER.error(_model_load_error)
        _engine = None
        yield
        return
    try:
        import torch

        torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "1"))))
        _engine = InferenceEngine(checkpoint, device=device)
    except Exception as exc:  # noqa: BLE001
        _model_load_error = str(exc)
        LOGGER.exception("Failed to load model: %s", exc)
        _engine = None
    if HISTORY_PATH.is_file():
        try:
            _cached_history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
            vacc = _cached_history.get("val_acc", [])
            if isinstance(vacc, list) and vacc:
                _cached_val_accuracy = float(max(vacc))
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            LOGGER.warning("Could not parse history for val accuracy: %s", exc)
    yield
    _engine = None


app = FastAPI(
    title="FER Emotion API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=RootHealthResponse)
def read_root() -> RootHealthResponse:
    """Public health metadata including class list."""
    return RootHealthResponse(
        status="ok",
        model="CustomCNN",
        classes=list(CFG.CLASS_NAMES),
        model_loaded=_engine is not None,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness with uptime for orchestrators (e.g. Render)."""
    return HealthResponse(
        status="ok",
        uptime_seconds=_uptime_seconds(),
        model_loaded=_engine is not None,
        checkpoint_bytes=_checkpoint_bytes(),
        model_error=_model_load_error,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: Annotated[UploadFile, File(..., description="Raster image file.")],
    engine: InferenceEngine = Depends(get_engine),
) -> PredictionResponse:
    """Run emotion classification on a multipart image upload."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded part must be an image/* MIME type.")
    try:
        raw = await file.read()
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to read upload: %s", exc)
        raise HTTPException(status_code=400, detail="Could not read uploaded file.") from exc
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file body.")
    try:
        out = engine.predict(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail="Inference failed.") from exc
    return PredictionResponse(
        emotion=str(out["emotion"]),
        confidence=float(out["confidence"]),
        probabilities={k: float(v) for k, v in out["probabilities"].items()},
        processing_time_ms=float(out["processing_time_ms"]),
        model_used=engine.model_used,
    )


@app.post("/predict/base64", response_model=PredictionResponse)
def predict_base64(
    body: Base64ImageRequest,
    engine: InferenceEngine = Depends(get_engine),
) -> PredictionResponse:
    """Classify an image provided as base64-encoded bytes."""
    try:
        out = engine.predict_from_base64(body.image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail="Inference failed.") from exc
    return PredictionResponse(
        emotion=str(out["emotion"]),
        confidence=float(out["confidence"]),
        probabilities={k: float(v) for k, v in out["probabilities"].items()},
        processing_time_ms=float(out["processing_time_ms"]),
        model_used=engine.model_used,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info(engine: InferenceEngine = Depends(get_engine)) -> ModelInfoResponse:
    """Summarize architecture and training accuracy."""
    arch_lines = str(engine.model).strip().splitlines()
    arch_summary = "\n".join(arch_lines[:80])
    if len(arch_lines) > 80:
        arch_summary += "\n... (truncated)"
    return ModelInfoResponse(
        model_name=engine.model_used,
        num_classes=len(engine.class_names),
        class_names=list(engine.class_names),
        total_params=engine.total_params,
        trainable_params=engine.trainable_params,
        val_accuracy=round(_cached_val_accuracy, 4),
        architecture_summary=arch_summary,
    )


@app.get("/results/history", response_model=HistoryResponse)
def results_history() -> HistoryResponse:
    """Expose persisted training curves from ``history.json``."""
    if not HISTORY_PATH.is_file():
        raise HTTPException(status_code=404, detail="Training history file not found.")
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=500, detail="Could not read history file.") from exc
    try:
        return HistoryResponse(
            train_loss=[float(x) for x in data["train_loss"]],
            val_loss=[float(x) for x in data["val_loss"]],
            train_acc=[float(x) for x in data["train_acc"]],
            val_acc=[float(x) for x in data["val_acc"]],
            lr=[float(x) for x in data["lr"]],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Invalid history JSON schema.") from exc


@app.get("/results/plots/{name}")
def results_plot(name: str) -> FileResponse:
    """Serve a PNG plot from ``fer_project/outputs/plots``."""
    if name not in ALLOWED_PLOTS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown plot '{name}'. Allowed: {sorted(ALLOWED_PLOTS)}",
        )
    path = (PLOTS_DIR / f"{name}.png").resolve()
    try:
        path.relative_to(PLOTS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid plot path.") from None
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Plot image not found on server.")
    return FileResponse(path, media_type="image/png", filename=f"{name}.png")
