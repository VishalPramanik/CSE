from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
from torch import nn
from torchvision import models

from .method import LayerEdit


@dataclass(frozen=True)
class LayerSpec:
    name: str
    channel_dim: int


@dataclass(frozen=True)
class ModelSpec:
    name: str
    layers: Tuple[LayerSpec, ...]


MODEL_SPECS: Dict[str, ModelSpec] = {
    # Principal encoder block outputs, matching the block/stage-level treatment
    # described in Sec. 3.4 and the architecture panels in Fig. 4.
    "resnet18": ModelSpec(
        name="resnet18",
        layers=(
            LayerSpec("layer1", 1),
            LayerSpec("layer2", 1),
            LayerSpec("layer3", 1),
            LayerSpec("layer4", 1),
        ),
    ),
    "efficientnet_b0": ModelSpec(
        name="efficientnet_b0",
        layers=tuple(LayerSpec(f"features.{i}", 1) for i in range(1, 8)),
    ),
    "swin_t": ModelSpec(
        name="swin_t",
        layers=(
            LayerSpec("features.1", -1),
            LayerSpec("features.3", -1),
            LayerSpec("features.5", -1),
            LayerSpec("features.7", -1),
        ),
    ),
}


def build_model(name: str, weights: str = "imagenet1k") -> nn.Module:
    """Create one of the three backbones evaluated in the paper."""
    name = name.lower()
    use_weights = weights.lower() != "none"

    if name == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if use_weights else None
        return models.resnet18(weights=w)
    if name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if use_weights else None
        return models.efficientnet_b0(weights=w)
    if name == "swin_t":
        w = models.Swin_T_Weights.IMAGENET1K_V1 if use_weights else None
        return models.swin_t(weights=w)
    raise ValueError(f"unknown model '{name}'. Choose from {sorted(MODEL_SPECS)}")


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown model '{name}'. Choose from {sorted(MODEL_SPECS)}") from exc


class ChannelAffine(nn.Module):
    """Block-output implementation of the paper's runtime affine form (Eq. 14)."""

    def __init__(self, scale: torch.Tensor, bias: torch.Tensor, channel_dim: int) -> None:
        super().__init__()
        if scale.ndim != 1 or bias.ndim != 1 or scale.numel() != bias.numel():
            raise ValueError("scale and bias must be equal-length 1-D tensors")
        self.register_buffer("scale", scale.detach().clone())
        self.register_buffer("bias", bias.detach().clone())
        self.channel_dim = channel_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError("CSE ChannelAffine expects a tensor block output")
        dim = self.channel_dim if self.channel_dim >= 0 else x.ndim + self.channel_dim
        if dim < 0 or dim >= x.ndim:
            raise ValueError(f"invalid channel dimension {self.channel_dim} for shape {tuple(x.shape)}")
        if x.shape[dim] != self.scale.numel():
            raise ValueError(
                f"channel mismatch: block has {x.shape[dim]} channels, edit has {self.scale.numel()}"
            )
        shape = [1] * x.ndim
        shape[dim] = self.scale.numel()
        scale = self.scale.to(dtype=x.dtype, device=x.device).view(shape)
        bias = self.bias.to(dtype=x.dtype, device=x.device).view(shape)
        return x * scale + bias


class BlockWithAffine(nn.Module):
    """Wrap an existing block and apply CSE after the complete block output."""

    def __init__(self, block: nn.Module, affine: ChannelAffine) -> None:
        super().__init__()
        self.block = block
        self.cse_affine = affine

    def forward(self, *args, **kwargs):
        return self.cse_affine(self.block(*args, **kwargs))


def _set_submodule(root: nn.Module, path: str, new_module: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for token in parts[:-1]:
        parent = parent[int(token)] if token.isdigit() else getattr(parent, token)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)


def apply_edits(
    model: nn.Module,
    spec: ModelSpec,
    edits: Dict[str, LayerEdit],
) -> nn.Module:
    """Apply fitted CSE edits at complete block outputs.

    This is the exact runtime affine form given by Eqs. (12)/(14). The paper's
    optional algebraic fold-in is safe only when an architecture exposes a
    directly adjacent affine operator; see cse.fold for those primitives.
    """
    expected = {layer.name for layer in spec.layers}
    missing = expected.difference(edits)
    if missing:
        raise ValueError(f"missing edits for layers: {sorted(missing)}")

    for layer in spec.layers:
        block = model.get_submodule(layer.name)
        edit = edits[layer.name]
        wrapper = BlockWithAffine(block, ChannelAffine(edit.scale, edit.bias, layer.channel_dim))
        _set_submodule(model, layer.name, wrapper)
    return model


def iter_model_layers(spec: ModelSpec) -> Iterable[LayerSpec]:
    return iter(spec.layers)
