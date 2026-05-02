"""Transfer-learning model wrapper for FER2013."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet50_Weights,
    VGG16_Weights,
)

from fer_project.config import CFG

LOGGER = logging.getLogger(__name__)


class TransferModel(nn.Module):
    """Configurable transfer-learning classifier for FER2013."""

    def __init__(
        self,
        model_name: str = CFG.TRANSFER_MODEL_NAME,
        num_classes: int = CFG.NUM_CLASSES,
        freeze_backbone: bool = CFG.FREEZE_BACKBONE,
    ) -> None:
        """Initialize selected pretrained model and replace classifier head.

        Args:
            model_name: One of resnet50, vgg16, mobilenet_v2.
            num_classes: Number of target classes.
            freeze_backbone: Whether to freeze pretrained backbone layers.
        """
        super().__init__()
        self.model_name = model_name
        self.feature_layer_name = ""
        self.backbone: nn.Module
        self.classifier_head: nn.Module

        if model_name == "resnet50":
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
            self.backbone = model
            self.classifier_head = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes),
            )
            self.feature_layer_name = "backbone.layer4"
        elif model_name == "vgg16":
            model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            in_features = model.classifier[-1].in_features
            model.classifier = nn.Identity()
            self.backbone = model
            self.classifier_head = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes),
            )
            self.feature_layer_name = "backbone.features"
        elif model_name == "mobilenet_v2":
            model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            in_features = model.classifier[-1].in_features
            model.classifier = nn.Identity()
            self.backbone = model
            self.classifier_head = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes),
            )
            self.feature_layer_name = "backbone.features"
        else:
            raise ValueError("model_name must be one of: resnet50, vgg16, mobilenet_v2")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self._log_parameter_counts()

    def _extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled features according to backbone type."""
        if self.model_name == "resnet50":
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        if self.model_name == "vgg16":
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        x = self.backbone.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = torch.flatten(x, 1)
        return x

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return last convolutional feature maps for Grad-CAM."""
        if self.model_name == "resnet50":
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            return x
        return self.backbone.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits."""
        features = self._extract_backbone_features(x)
        logits = self.classifier_head(features)
        return logits

    def _log_parameter_counts(self) -> None:
        """Log total and trainable model parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        LOGGER.info("TransferModel(%s) total params: %d", self.model_name, total_params)
        LOGGER.info("TransferModel(%s) trainable params: %d", self.model_name, trainable_params)

