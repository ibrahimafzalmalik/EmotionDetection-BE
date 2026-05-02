"""Central configuration module for the FER2013 project."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch


@dataclass(frozen=True)
class Config:
    """Configuration container for project-level hyperparameters and paths."""

    PROJECT_ROOT: Path = Path(__file__).resolve().parent
    DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
    OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
    CHECKPOINT_DIR: Path = OUTPUT_DIR / "checkpoints"
    PLOTS_DIR: Path = OUTPUT_DIR / "plots"
    RESULTS_DIR: Path = OUTPUT_DIR / "results"
    CHECKPOINT_PATH: Path = CHECKPOINT_DIR / "best_model.pth"

    IMG_SIZE: int = 48
    TRANSFER_IMG_SIZE: int = 224
    BATCH_SIZE: int = 64
    NUM_CLASSES: int = 7
    EPOCHS: int = 50
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    DROPOUT_RATE: float = 0.5
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True

    USE_TRANSFER_LEARNING: bool = False
    TRANSFER_MODEL_NAME: str = "resnet50"
    FREEZE_BACKBONE: bool = True
    EARLY_STOPPING_PATIENCE: int = 7

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SEED: int = 42

    CLASS_NAMES: List[str] = (
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    )


CFG = Config()


def ensure_directories(cfg: Config = CFG) -> None:
    """Create project output and cache directories if missing."""
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_effective_image_size(cfg: Config = CFG) -> int:
    """Return active image size based on selected training strategy."""
    return cfg.TRANSFER_IMG_SIZE if cfg.USE_TRANSFER_LEARNING else cfg.IMG_SIZE


def validate_config(cfg: Config = CFG) -> None:
    """Validate critical configuration consistency and paths."""
    if len(cfg.CLASS_NAMES) != cfg.NUM_CLASSES:
        raise ValueError(
            "NUM_CLASSES must match the number of CLASS_NAMES. "
            f"Got {cfg.NUM_CLASSES} vs {len(cfg.CLASS_NAMES)}."
        )
    if not cfg.DATA_DIR.exists():
        raise FileNotFoundError(
            f"DATA_DIR not found at '{cfg.DATA_DIR}'. "
            "Place FER2013 train/ and test/ folders under data/raw/."
        )
    if cfg.TRANSFER_MODEL_NAME not in {"resnet50", "vgg16", "mobilenet_v2"}:
        raise ValueError(
            "TRANSFER_MODEL_NAME must be one of: resnet50, vgg16, mobilenet_v2."
        )


def seed_everything(seed: int = CFG.SEED) -> None:
    """Seed random generators for deterministic experimentation.

    Args:
        seed: Global seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

