"""Evaluation utilities for FER2013 models."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from fer_project.config import CFG, Config, ensure_directories, seed_everything
from fer_project.training.train import build_model
from fer_project.utils.dataset import get_dataloaders
from fer_project.utils.metrics import plot_confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def _save_predictions_csv(
    true_labels: list[int],
    pred_labels: list[int],
    confidences: list[float],
    class_names: list[str],
    output_path: Path,
) -> None:
    """Save per-sample predictions as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", "predicted_label", "confidence", "correct"])
        for t, p, c in zip(true_labels, pred_labels, confidences):
            writer.writerow([class_names[t], class_names[p], f"{c:.6f}", int(t == p)])


def evaluate(cfg: Config = CFG) -> dict[str, Any]:
    """Evaluate best model checkpoint on test split.

    Args:
        cfg: Project configuration.

    Returns:
        Dictionary of evaluation artifacts.
    """
    ensure_directories(cfg)
    seed_everything(cfg.SEED)

    _, _, test_loader, _, metadata = get_dataloaders(
        data_dir=cfg.DATA_DIR,
        batch_size=cfg.BATCH_SIZE,
        seed=cfg.SEED,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY and cfg.DEVICE == "cuda",
    )
    class_names = metadata["class_names"]

    model = build_model(cfg)
    checkpoint = torch.load(
        cfg.CHECKPOINT_PATH,
        map_location=cfg.DEVICE,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(cfg.DEVICE)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluate", leave=False):
            inputs = inputs.to(cfg.DEVICE, non_blocking=True)
            targets = targets.to(cfg.DEVICE, non_blocking=True)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            conf, preds = torch.max(probs, dim=1)
            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            confidences.extend(conf.cpu().numpy().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    LOGGER.info("Test Accuracy: %.4f", accuracy)
    LOGGER.info("Classification report:\n%s", report)

    cm_path = plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        output_path=cfg.PLOTS_DIR / "confusion_matrix.png",
        normalize=True,
    )

    predictions_path = cfg.RESULTS_DIR / "predictions.csv"
    _save_predictions_csv(
        true_labels=y_true,
        pred_labels=y_pred,
        confidences=confidences,
        class_names=class_names,
        output_path=predictions_path,
    )
    LOGGER.info("Saved confusion matrix to %s", cm_path)
    LOGGER.info("Saved predictions to %s", predictions_path)

    return {
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix_path": str(cm_path),
        "predictions_path": str(predictions_path),
    }


if __name__ == "__main__":
    evaluate(CFG)

