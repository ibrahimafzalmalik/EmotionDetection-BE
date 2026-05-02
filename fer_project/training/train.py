"""Training pipeline for FER2013 models."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from fer_project.config import CFG, Config, ensure_directories, seed_everything
from fer_project.models.custom_cnn import CustomCNN
from fer_project.models.transfer_model import TransferModel
from fer_project.utils.dataset import get_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


class Trainer:
    """Trainer encapsulating optimization, validation, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        class_weights: torch.Tensor,
        cfg: Config = CFG,
    ) -> None:
        """Initialize trainer components and state.

        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            class_weights: Class weights tensor for loss balancing.
            cfg: Project configuration.
        """
        self.cfg = cfg
        self.device = torch.device(cfg.DEVICE)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            params=params,
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY,
        )
        # WHY: Cosine annealing improves late-stage convergence stability for FER.
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.EPOCHS)
        # WHY: class weighting counters FER2013's strong class imbalance.
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))

        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

    def train_one_epoch(self) -> tuple[float, float]:
        """Train model for one epoch.

        Returns:
            Tuple containing average loss and accuracy.
        """
        self.model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        progress = tqdm(self.train_loader, desc="Train", leave=False)
        for inputs, targets in progress:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            preds = logits.argmax(dim=1)
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (preds == targets).sum().item()
            running_total += batch_size

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{(running_correct / max(running_total, 1)):.4f}",
            )

        return running_loss / running_total, running_correct / running_total

    def validate_one_epoch(self) -> tuple[float, float]:
        """Evaluate model on validation set.

        Returns:
            Tuple containing average validation loss and accuracy.
        """
        self.model.eval()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        with torch.no_grad():
            progress = tqdm(self.val_loader, desc="Val", leave=False)
            for inputs, targets in progress:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                preds = logits.argmax(dim=1)

                batch_size = targets.size(0)
                running_loss += loss.item() * batch_size
                running_correct += (preds == targets).sum().item()
                running_total += batch_size

                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{(running_correct / max(running_total, 1)):.4f}",
                )

        return running_loss / running_total, running_correct / running_total

    def fit(self) -> dict[str, list[float]]:
        """Run full training loop with early stopping.

        Returns:
            Training history dictionary.
        """
        for epoch in range(1, self.cfg.EPOCHS + 1):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate_one_epoch()
            current_lr = float(self.optimizer.param_groups[0]["lr"])

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            LOGGER.info(
                "Epoch %d/%d | train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f lr=%.6f",
                epoch,
                self.cfg.EPOCHS,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                current_lr,
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch=epoch)
            else:
                self.epochs_without_improvement += 1

            self.scheduler.step()

            if self.epochs_without_improvement >= self.cfg.EARLY_STOPPING_PATIENCE:
                LOGGER.info(
                    "Early stopping triggered at epoch %d (best epoch: %d, best val_acc: %.4f)",
                    epoch,
                    self.best_epoch,
                    self.best_val_acc,
                )
                break

        self._save_history()
        return self.history

    def _save_checkpoint(self, epoch: int) -> None:
        """Save best model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "config": asdict(self.cfg),
        }
        torch.save(checkpoint, self.cfg.CHECKPOINT_PATH)
        LOGGER.info("Saved best checkpoint to %s", self.cfg.CHECKPOINT_PATH)

    def _save_history(self) -> None:
        """Persist training history to JSON file."""
        history_path = self.cfg.RESULTS_DIR / "history.json"
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        LOGGER.info("Saved history to %s", history_path)


def build_model(cfg: Config = CFG) -> nn.Module:
    """Factory for custom or transfer model according to config."""
    if cfg.USE_TRANSFER_LEARNING:
        return TransferModel(
            model_name=cfg.TRANSFER_MODEL_NAME,
            num_classes=cfg.NUM_CLASSES,
            freeze_backbone=cfg.FREEZE_BACKBONE,
        )
    return CustomCNN(num_classes=cfg.NUM_CLASSES, dropout_rate=cfg.DROPOUT_RATE)


def run_training(cfg: Config = CFG) -> dict[str, Any]:
    """Execute full training routine.

    Args:
        cfg: Project configuration.

    Returns:
        Run artifacts and metadata.
    """
    ensure_directories(cfg)
    seed_everything(cfg.SEED)

    train_loader, val_loader, test_loader, class_weights, metadata = get_dataloaders(
        data_dir=cfg.DATA_DIR,
        batch_size=cfg.BATCH_SIZE,
        seed=cfg.SEED,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY and cfg.DEVICE == "cuda",
    )
    model = build_model(cfg)
    trainer = Trainer(model, train_loader, val_loader, class_weights=class_weights, cfg=cfg)
    history = trainer.fit()
    return {
        "history": history,
        "metadata": metadata,
        "test_loader": test_loader,
        "model": model,
    }


if __name__ == "__main__":
    run_training(CFG)

