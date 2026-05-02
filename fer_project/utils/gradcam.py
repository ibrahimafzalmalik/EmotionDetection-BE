"""Grad-CAM implementation and batch visualization helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from fer_project.config import CFG


def _snapshot_relu_inplace(model: torch.nn.Module) -> list[tuple[torch.nn.ReLU, bool]]:
    return [(m, m.inplace) for m in model.modules() if isinstance(m, torch.nn.ReLU)]


def _restore_relu_inplace(snapshot: list[tuple[torch.nn.ReLU, bool]]) -> None:
    for module, prev in snapshot:
        module.inplace = prev


def _set_all_relu_inplace(model: torch.nn.Module, inplace: bool) -> None:
    """Grad-CAM uses register_full_backward_hook, which conflicts with ReLU(inplace=True)."""
    for m in model.modules():
        if isinstance(m, torch.nn.ReLU):
            m.inplace = inplace


class GradCAM:
    """Gradient-weighted Class Activation Mapping utility."""

    def __init__(self, model: torch.nn.Module, target_layer_name: str) -> None:
        """Initialize hooks and references.

        Args:
            model: Trained PyTorch model.
            target_layer_name: Dot-path to the target convolution layer.
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self._hooks: list[Any] = []
        self._register_hooks()

    def _get_layer(self) -> torch.nn.Module:
        """Resolve nested module path into layer object."""
        layer: torch.nn.Module = self.model
        for attr in self.target_layer_name.split("."):
            layer = getattr(layer, attr)
        return layer

    def _register_hooks(self) -> None:
        """Register forward and backward hooks."""
        target_layer = self._get_layer()

        def forward_hook(_module: torch.nn.Module, _input: tuple[Any], output: torch.Tensor) -> None:
            self.activations = output.detach().clone()

        def backward_hook(
            _module: torch.nn.Module,
            _grad_input: tuple[Any],
            grad_output: tuple[torch.Tensor],
        ) -> None:
            self.gradients = grad_output[0].detach().clone()

        self._hooks.append(target_layer.register_forward_hook(forward_hook))
        self._hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        """Release hook handles."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate Grad-CAM heatmap and overlay for one input batch item.

        Args:
            input_tensor: Input tensor of shape [1, C, H, W].
            class_idx: Target class index, defaults to top predicted class.

        Returns:
            Tuple of heatmap and RGB overlay arrays.
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        target = class_idx if class_idx is not None else int(torch.argmax(logits, dim=1).item())
        score = logits[:, target]
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        heatmap_uint8 = np.uint8(255 * cam)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        image = input_tensor.squeeze(0).detach().cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        overlay = np.clip(0.55 * image + 0.45 * heatmap_rgb, 0.0, 1.0)
        return cam, overlay


def visualize_batch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: list[str],
    target_layer_name: str,
    n: int = 8,
    output_path: Path | None = None,
    device: str = CFG.DEVICE,
) -> Path:
    """Generate Grad-CAM triplet visualization for random test samples.

    Args:
        model: Trained model.
        dataloader: Test dataloader.
        class_names: Ordered class names.
        target_layer_name: Target layer path for Grad-CAM.
        n: Number of samples.
        output_path: Optional save path.
        device: Inference device.

    Returns:
        Path to saved figure.
    """
    save_path = output_path or (CFG.PLOTS_DIR / "gradcam_samples.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_samples: list[tuple[torch.Tensor, int]] = []
    for inputs, targets in dataloader:
        for i in range(inputs.size(0)):
            all_samples.append((inputs[i], int(targets[i])))

    if not all_samples:
        raise ValueError("Dataloader is empty; cannot generate Grad-CAM samples.")

    chosen = random.sample(all_samples, k=min(n, len(all_samples)))

    relu_snapshot = _snapshot_relu_inplace(model)
    gradcam: GradCAM | None = None
    try:
        _set_all_relu_inplace(model, False)
        gradcam = GradCAM(model, target_layer_name=target_layer_name)

        rows = len(chosen)
        fig, axes = plt.subplots(rows, 3, figsize=(12, 3 * rows))
        if rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for row_idx, (image, true_label) in enumerate(chosen):
            input_tensor = image.unsqueeze(0).to(device)
            with torch.enable_grad():
                heatmap, overlay = gradcam.generate(input_tensor)
                probs = torch.softmax(model(input_tensor), dim=1)
                pred_label = int(torch.argmax(probs, dim=1).item())

            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            original = (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

            axes[row_idx, 0].imshow(original)
            axes[row_idx, 0].set_title(f"Original\nTrue: {class_names[true_label]}")
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(heatmap, cmap="jet")
            axes[row_idx, 1].set_title("Grad-CAM Heatmap")
            axes[row_idx, 1].axis("off")

            axes[row_idx, 2].imshow(overlay)
            axes[row_idx, 2].set_title(f"Overlay\nPred: {class_names[pred_label]}")
            axes[row_idx, 2].axis("off")

        plt.tight_layout()
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    finally:
        _restore_relu_inplace(relu_snapshot)
        if gradcam is not None:
            gradcam.remove_hooks()

    return save_path

