"""Dataset and dataloader utilities for FER2013."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import ImageFolder

from fer_project.config import CFG
from fer_project.utils.transforms import get_train_transforms, get_val_test_transforms

LOGGER = logging.getLogger(__name__)


class TransformSubset(Dataset):
    """A subset wrapper that applies a dedicated transform.

    Args:
        subset: Subset object from random_split.
        transform: Transform to apply per sample.
    """

    def __init__(self, subset: Subset, transform: Any) -> None:
        self.subset = subset
        self.transform = transform
        self.targets = [subset.dataset.targets[idx] for idx in subset.indices]

    def __len__(self) -> int:
        """Return subset size."""
        return len(self.subset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return transformed image-label pair."""
        image, target = self.subset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _log_class_distribution(name: str, targets: list[int], class_names: list[str]) -> None:
    """Log class frequencies for sanity checking."""
    counter = Counter(targets)
    LOGGER.info("%s class distribution:", name)
    for class_idx, class_name in enumerate(class_names):
        LOGGER.info("  %-10s -> %d", class_name, counter.get(class_idx, 0))


def _build_class_weights(targets: list[int], num_classes: int, device: str) -> torch.Tensor:
    """Compute class weights for imbalanced classification."""
    classes = np.arange(num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.array(targets))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def get_dataloaders(
    data_dir: Path = CFG.DATA_DIR,
    batch_size: int = CFG.BATCH_SIZE,
    seed: int = CFG.SEED,
    num_workers: int = CFG.NUM_WORKERS,
    pin_memory: bool = CFG.PIN_MEMORY,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, dict[str, Any]]:
    """Create FER2013 train/val/test dataloaders and class weights.

    Args:
        data_dir: Root path containing train and test folders.
        batch_size: Batch size for all dataloaders.
        seed: Seed for deterministic train/val split.
        num_workers: Data loader workers.
        pin_memory: Pin memory for faster host-to-device transfer.

    Returns:
        Tuple of train_loader, val_loader, test_loader, class_weights, metadata.
    """
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected FER2013 folders at '{train_dir}' and '{test_dir}'."
        )

    base_train = ImageFolder(root=str(train_dir), transform=None)
    test_dataset = ImageFolder(root=str(test_dir), transform=get_val_test_transforms())

    generator = torch.Generator().manual_seed(seed)
    train_len = int(0.8 * len(base_train))
    val_len = len(base_train) - train_len
    train_subset, val_subset = random_split(
        base_train, [train_len, val_len], generator=generator
    )

    train_dataset = TransformSubset(train_subset, transform=get_train_transforms())
    val_dataset = TransformSubset(val_subset, transform=get_val_test_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    class_names = base_train.classes
    _log_class_distribution("Train split", train_dataset.targets, class_names)
    _log_class_distribution("Validation split", val_dataset.targets, class_names)
    _log_class_distribution("Test set", test_dataset.targets, class_names)

    class_weights = _build_class_weights(
        targets=train_dataset.targets, num_classes=len(class_names), device=CFG.DEVICE
    )

    metadata = {
        "class_names": class_names,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
    }
    return train_loader, val_loader, test_loader, class_weights, metadata

