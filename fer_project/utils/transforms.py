"""Data transformation pipelines for FER2013."""

from __future__ import annotations

from torchvision import transforms

from fer_project.config import CFG, get_effective_image_size

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int | None = None) -> transforms.Compose:
    """Build training transformations with augmentation.

    Args:
        img_size: Target square image size.

    Returns:
        Torchvision transformation pipeline for training.
    """
    size = img_size or get_effective_image_size(CFG)
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_test_transforms(img_size: int | None = None) -> transforms.Compose:
    """Build deterministic validation/test preprocessing pipeline.

    Args:
        img_size: Target square image size.

    Returns:
        Torchvision transformation pipeline for validation and testing.
    """
    size = img_size or get_effective_image_size(CFG)
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


TRAIN_TRANSFORMS = get_train_transforms()
VAL_TEST_TRANSFORMS = get_val_test_transforms()

