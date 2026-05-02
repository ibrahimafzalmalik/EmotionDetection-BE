"""Metrics and plotting utilities for FER2013 experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from fer_project.config import CFG


def plot_training_curves(
    history: dict[str, list[float]], output_path: Path | None = None
) -> Path:
    """Plot train/validation loss and accuracy curves.

    Args:
        history: Training history dictionary from Trainer.fit().
        output_path: Optional custom save path.

    Returns:
        Saved figure path.
    """
    save_path = output_path or (CFG.PLOTS_DIR / "training_curves.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy")
    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_confusion_matrix(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str],
    output_path: Path | None = None,
    normalize: bool = True,
) -> Path:
    """Plot and save a confusion matrix heatmap.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Ordered class names.
        output_path: Optional save path.
        normalize: Whether to normalize by true labels.

    Returns:
        Saved figure path.
    """
    save_path = output_path or (CFG.PLOTS_DIR / "confusion_matrix.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if normalize:
        cm = cm.astype(np.float64)
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def compute_metrics(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> dict[str, Any]:
    """Compute aggregate and per-class metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with accuracy and precision/recall/f1 details.
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "per_class": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
        },
        "macro_avg": {"precision": macro[0], "recall": macro[1], "f1": macro[2]},
        "weighted_avg": {"precision": weighted[0], "recall": weighted[1], "f1": weighted[2]},
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def show_misclassified(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: list[str],
    n: int = 16,
    output_path: Path | None = None,
    device: str = CFG.DEVICE,
) -> Path:
    """Render a grid of misclassified samples.

    Args:
        model: Trained classification model.
        dataloader: Data loader with normalized images.
        class_names: Ordered class names.
        n: Number of misclassified samples to display.
        output_path: Optional output path.
        device: Inference device.

    Returns:
        Saved image path.
    """
    save_path = output_path or (CFG.PLOTS_DIR / "misclassified.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    mistakes: list[tuple[torch.Tensor, int, int]] = []
    inv_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    inv_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1)
            wrong_mask = preds != targets
            for idx in wrong_mask.nonzero(as_tuple=False).flatten().tolist():
                mistakes.append((inputs[idx].cpu(), int(targets[idx].cpu()), int(preds[idx].cpu())))
                if len(mistakes) >= n:
                    break
            if len(mistakes) >= n:
                break

    cols = 4
    rows = max(1, int(np.ceil(len(mistakes) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes_arr = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else np.array([axes])

    for ax_idx, ax in enumerate(axes_arr):
        if ax_idx < len(mistakes):
            image, true_label, pred_label = mistakes[ax_idx]
            img = (image * inv_std + inv_mean).clamp(0, 1).permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_title(
                f"T: {class_names[true_label]} | P: {class_names[pred_label]}",
                color="red",
                fontsize=9,
            )
            ax.axis("off")
        else:
            ax.axis("off")

    fig.suptitle("Misclassified Samples", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path

