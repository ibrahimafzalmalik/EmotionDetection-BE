"""Pydantic request and response models for the FER inference API."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Single-image emotion prediction with per-class scores."""

    emotion: str = Field(..., description="Predicted class label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax probability of top class.")
    probabilities: Dict[str, float] = Field(..., description="Softmax probabilities for all classes.")
    processing_time_ms: float = Field(..., description="Wall time for inference in milliseconds.")
    model_used: str = Field(..., description="Identifier of the loaded model architecture.")


class ModelInfoResponse(BaseModel):
    """Static metadata and training summary for the served checkpoint."""

    model_name: str
    num_classes: int
    class_names: List[str]
    total_params: int
    trainable_params: int
    val_accuracy: float = Field(..., description="Best or final validation accuracy from training history.")
    architecture_summary: str = Field(
        ...,
        description="Human-readable outline of the module graph (truncated for JSON size).",
    )


class HealthResponse(BaseModel):
    """Liveness probe payload."""

    status: str
    uptime_seconds: float


class HistoryResponse(BaseModel):
    """Per-epoch training curves persisted from the trainer."""

    train_loss: List[float]
    val_loss: List[float]
    train_acc: List[float]
    val_acc: List[float]
    lr: List[float]


class Base64ImageRequest(BaseModel):
    """JSON body for base64-encoded image inference."""

    image: str = Field(..., min_length=1, description="Base64-encoded image bytes (no data URL prefix required).")


class RootHealthResponse(BaseModel):
    """Minimal service metadata for the landing health route."""

    status: str
    model: str
    classes: List[str]
