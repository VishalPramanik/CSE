"""
Grad-CAM visualization for qualitative unlearning analysis (Figures 3, 6).
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class GradCAM:
    """Gradient-weighted Class Activation Mapping."""

    def __init__(self, model: nn.Module, target_layer_name: str):
        self.model = model
        self._gradients = None
        self._activations = None
        self._hooks: list = []

        backbone = model.backbone if hasattr(model, "backbone") else model
        target = dict(backbone.named_modules())[target_layer_name]
        self._hooks.append(target.register_forward_hook(self._save_act))
        self._hooks.append(target.register_full_backward_hook(self._save_grad))

    def _save_act(self, mod, inp, out):
        self._activations = out.detach()

    def _save_grad(self, mod, gi, go):
        self._gradients = go[0].detach()

    @torch.enable_grad()
    def generate(self, image: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM heatmap for a single image (1, C, H, W)."""
        self.model.eval()
        image = image.requires_grad_(True)
        output = self.model(image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        if self._gradients is None or self._activations is None:
            return np.zeros((image.shape[2], image.shape[3]))

        if self._gradients.dim() == 4:
            weights = self._gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self._activations).sum(dim=1, keepdim=True)
        elif self._gradients.dim() == 3:
            weights = self._gradients.mean(dim=1, keepdim=True)
            B, N, C = self._activations.shape
            h = w = int(N ** 0.5) or 1
            act = self._activations[:, :h*w, :].reshape(B, h, w, C).permute(0, 3, 1, 2)
            w2 = weights.reshape(B, 1, 1, C).permute(0, 3, 1, 2)
            cam = (w2 * act).sum(dim=1, keepdim=True)
        else:
            return np.zeros((image.shape[2], image.shape[3]))

        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(image.shape[2], image.shape[3]),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def visualize_gradcam(
    images: List[torch.Tensor],
    heatmaps_before: List[np.ndarray],
    heatmaps_after: List[np.ndarray],
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> Figure:
    """Create comparison figure: Original | Grad-CAM Before | Grad-CAM After."""
    n = len(images)
    fig, axes = plt.subplots(n, 3, figsize=(figsize[0], figsize[1] * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(n):
        img = images[i].squeeze().cpu().numpy()
        if img.ndim == 3:
            img = np.clip(img.transpose(1, 2, 0) * std + mean, 0, 1)

        axes[i, 0].imshow(img); axes[i, 0].axis("off")
        axes[i, 1].imshow(img); axes[i, 1].imshow(heatmaps_before[i], cmap="jet", alpha=0.4)
        axes[i, 1].axis("off")
        axes[i, 2].imshow(img); axes[i, 2].imshow(heatmaps_after[i], cmap="jet", alpha=0.4)
        axes[i, 2].axis("off")

        if titles and i < len(titles):
            axes[i, 0].set_ylabel(titles[i], fontsize=12)

    for j, t in enumerate(["Original", "Grad-CAM (Before)", "Grad-CAM (After)"]):
        axes[0, j].set_title(t, fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
