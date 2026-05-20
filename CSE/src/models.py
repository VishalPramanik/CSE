"""
Backbone model wrappers with feature extraction hooks.

Supports ResNet-18, EfficientNet-B0, and Swin-T with consistent
interfaces for intermediate feature extraction (required by CSE Stages 1-2).
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger("CSE")


class FeatureExtractor(nn.Module):
    """
    Wraps a backbone to expose intermediate features via forward hooks.

    After a forward pass, ``self.features`` contains a dict mapping
    layer names to their output tensors (after global average pooling
    for spatial features).
    """

    def __init__(
        self,
        backbone: nn.Module,
        layer_names: List[str],
    ):
        super().__init__()
        self.backbone = backbone
        self.layer_names = layer_names
        self.features: Dict[str, torch.Tensor] = {}
        self._hooks: list = []

        # Register hooks on target layers
        named = dict(backbone.named_modules())
        for name in layer_names:
            if name not in named:
                raise ValueError(
                    f"Layer '{name}' not found. Available: {list(named.keys())}"
                )
            hook = named[name].register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            feat = output
            if feat.dim() == 4:          # (B, C, H, W) -- conv features
                feat = feat.mean(dim=(2, 3))
            elif feat.dim() == 3:        # (B, N, C) -- transformer patches
                feat = feat.mean(dim=1)
            self.features[name] = feat
        return hook_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.features = {}
        return self.backbone(x)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ─────────────────────────────────────────────────────────
# Layer registries
# ─────────────────────────────────────────────────────────

RESNET18_LAYERS = ["layer1", "layer2", "layer3", "layer4"]

EFFICIENTNET_LAYERS = [
    "features.1", "features.2", "features.3", "features.4",
    "features.5", "features.6", "features.7",
]

SWINT_LAYERS = ["features.1", "features.3", "features.5", "features.7"]


# ─────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────

def _try_load(model_fn, weights, **kwargs):
    """Load pretrained weights; fall back to random init on failure."""
    if weights is not None:
        try:
            return model_fn(weights=weights, **kwargs)
        except Exception as exc:
            logger.warning(
                "Could not download pretrained weights (%s). "
                "Falling back to random initialization.", exc
            )
    return model_fn(weights=None, **kwargs)


def _build_resnet18(pretrained: bool, num_classes: int):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    base = _try_load(models.resnet18, weights)
    base.fc = nn.Linear(base.fc.in_features, num_classes)
    nn.init.xavier_uniform_(base.fc.weight)
    nn.init.zeros_(base.fc.bias)
    return FeatureExtractor(base, RESNET18_LAYERS), RESNET18_LAYERS


def _build_efficientnet_b0(pretrained: bool, num_classes: int):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    base = _try_load(models.efficientnet_b0, weights)
    base.classifier[1] = nn.Linear(base.classifier[1].in_features, num_classes)
    nn.init.xavier_uniform_(base.classifier[1].weight)
    nn.init.zeros_(base.classifier[1].bias)
    return FeatureExtractor(base, EFFICIENTNET_LAYERS), EFFICIENTNET_LAYERS


def _build_swin_t(pretrained: bool, num_classes: int):
    weights = models.Swin_T_Weights.DEFAULT if pretrained else None
    base = _try_load(models.swin_t, weights)
    base.head = nn.Linear(base.head.in_features, num_classes)
    nn.init.xavier_uniform_(base.head.weight)
    nn.init.zeros_(base.head.bias)
    return FeatureExtractor(base, SWINT_LAYERS), SWINT_LAYERS


_BUILDERS = {
    "resnet18": _build_resnet18,
    "efficientnet_b0": _build_efficientnet_b0,
    "swin_t": _build_swin_t,
}


def build_model(
    backbone: str = "resnet18",
    pretrained: bool = True,
    num_classes: int = 10,
    device: torch.device = torch.device("cpu"),
) -> Tuple[FeatureExtractor, List[str]]:
    """
    Build a backbone wrapped with feature extraction hooks.

    Args:
        backbone: One of ``resnet18``, ``efficientnet_b0``, ``swin_t``.
        pretrained: Load ImageNet pretrained weights.
        num_classes: Number of output classes.
        device: Target device.

    Returns:
        ``(model, layer_names)`` tuple.
    """
    if backbone not in _BUILDERS:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose from: {list(_BUILDERS.keys())}")
    model, layers = _BUILDERS[backbone](pretrained, num_classes)
    return model.to(device), layers
