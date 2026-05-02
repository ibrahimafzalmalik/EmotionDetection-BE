"""Custom CNN architecture for FER2013 classification."""

from __future__ import annotations

import torch
import torch.nn as nn

from fer_project.config import CFG, get_effective_image_size


class CustomCNN(nn.Module):
    """Three-stage CNN tailored for FER2013 expressions."""

    def __init__(self, num_classes: int = CFG.NUM_CLASSES, dropout_rate: float = CFG.DROPOUT_RATE) -> None:
        """Initialize convolutional and classifier blocks.

        Args:
            num_classes: Number of output classes.
            dropout_rate: Dropout probability for dense layers.
        """
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # WHY: stabilize activation scale before nonlinearities.
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )

        feat_size = self._infer_feature_size(get_effective_image_size(CFG))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def _infer_feature_size(self, img_size: int) -> int:
        """Infer flattened feature size after conv blocks."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feats = self.get_feature_maps(dummy)
            return int(feats.numel() / feats.shape[0])

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return output tensor from last convolutional block."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run model forward pass and return class logits.

        Args:
            x: Input tensor [B, 3, H, W].

        Returns:
            Logits tensor [B, num_classes].
        """
        # WHY: keep conv-feature extraction isolated to reuse in Grad-CAM.
        features = self.get_feature_maps(x)
        logits = self.classifier(features)
        return logits

