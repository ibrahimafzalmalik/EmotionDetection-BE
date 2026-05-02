"""Torch inference engine loading the trained CustomCNN checkpoint."""

from __future__ import annotations

import base64
import io
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

# WHY: `uvicorn backend.main:app` runs with service root on sys.path; insert parent so `fer_project` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fer_project.config import CFG  # noqa: E402
from fer_project.models.custom_cnn import CustomCNN  # noqa: E402

LOGGER = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class InferenceEngine:
    """Loads CustomCNN once and runs deterministic val/test preprocessing."""

    def __init__(self, checkpoint_path: Path, device: str = "cpu") -> None:
        """Build model, load weights, and compile the inference transform pipeline.

        Args:
            checkpoint_path: Path to ``best_model.pth`` containing ``model_state_dict``.
            device: Torch device string (``cpu`` or ``cuda``).
        """
        self.device = torch.device(device)
        self.class_names: List[str] = list(CFG.CLASS_NAMES)
        self.model_used = "CustomCNN"

        self._transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        self.model = CustomCNN(num_classes=CFG.NUM_CLASSES, dropout_rate=CFG.DROPOUT_RATE)
        try:
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            # WHY: Older PyTorch builds omit ``weights_only``; keep local dev working without pinning.
            state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.total_params = int(total_params)
        self.trainable_params = int(trainable_params)

        LOGGER.info(
            "InferenceEngine ready on %s (%s params, checkpoint=%s)",
            self.device,
            self.total_params,
            checkpoint_path,
        )

    def predict(self, image_bytes: bytes) -> Dict[str, object]:
        """Decode bytes as an image, run a forward pass, and format API output.

        Args:
            image_bytes: Raw file or decoded base64 bytes.

        Returns:
            Dict with emotion, confidence, probabilities, processing_time_ms (no model_used).

        Raises:
            ValueError: If bytes cannot be decoded as a raster image.
        """
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Invalid or corrupted image data.") from exc

        tensor = self._transform(img).unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        confidences = probs.cpu().tolist()
        probabilities = {name: float(confidences[i]) for i, name in enumerate(self.class_names)}
        best_idx = int(torch.argmax(probs).item())
        emotion = self.class_names[best_idx]
        confidence = float(probs[best_idx].item())

        return {
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": probabilities,
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def predict_from_base64(self, b64_string: str) -> Dict[str, object]:
        """Decode a base64 payload and delegate to :meth:`predict`.

        Args:
            b64_string: Standard base64 (padding optional); data-URL prefix stripped if present.

        Returns:
            Same structure as :meth:`predict`.

        Raises:
            ValueError: If base64 or image decoding fails.
        """
        raw = b64_string.strip()
        if raw.startswith("data:") and "base64," in raw:
            raw = raw.split("base64,", 1)[1]
        raw = raw.replace("\n", "").replace("\r", "")
        try:
            decoded = base64.b64decode(raw, validate=False)
        except Exception as exc:  # noqa: BLE001 — surface as 400
            raise ValueError("Invalid base64 encoding.") from exc
        if not decoded:
            raise ValueError("Empty image payload after base64 decode.")
        return self.predict(decoded)
