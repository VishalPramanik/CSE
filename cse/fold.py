from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def fold_input_affine_into_linear(layer: nn.Linear, scale: torch.Tensor, bias: torch.Tensor) -> nn.Linear:
    """Fold y = W (scale*x + bias) + b into an adjacent nn.Linear.

    This implements the paper's algebraic fold-in when the edited feature is
    consumed directly by a linear layer, with no intervening nonlinear or
    residual branch.
    """
    if scale.ndim != 1 or bias.ndim != 1 or scale.numel() != layer.in_features:
        raise ValueError("scale/bias must match linear input features")
    device, dtype = layer.weight.device, layer.weight.dtype
    scale = scale.to(device=device, dtype=dtype)
    bias = bias.to(device=device, dtype=dtype)
    old_w = layer.weight.detach().clone()
    old_b = layer.bias.detach().clone() if layer.bias is not None else torch.zeros(layer.out_features, device=device, dtype=dtype)
    new_b = old_b + old_w @ bias
    layer.weight.mul_(scale.unsqueeze(0))
    if layer.bias is None:
        layer.bias = nn.Parameter(new_b)
    else:
        layer.bias.copy_(new_b)
    return layer


@torch.no_grad()
def fold_input_affine_into_conv2d(layer: nn.Conv2d, scale: torch.Tensor, bias: torch.Tensor) -> nn.Conv2d:
    """Fold a channel-wise input affine into an adjacent Conv2d.

    For nonzero padding, a spatially constant input bias is not exactly constant
    at padded borders. To preserve exactness, bias fold-in is therefore accepted
    only for zero padding or zero bias. Scale folding is always exact.
    """
    if layer.groups != 1:
        raise ValueError("exact generic folding is implemented only for groups=1 Conv2d")
    if scale.ndim != 1 or bias.ndim != 1 or scale.numel() != layer.in_channels:
        raise ValueError("scale/bias must match convolution input channels")
    if any(p != 0 for p in layer.padding) and torch.any(bias != 0):
        raise ValueError("cannot exactly fold a nonzero input bias through padded Conv2d")

    device, dtype = layer.weight.device, layer.weight.dtype
    scale = scale.to(device=device, dtype=dtype)
    bias = bias.to(device=device, dtype=dtype)
    old_w = layer.weight.detach().clone()
    old_b = layer.bias.detach().clone() if layer.bias is not None else torch.zeros(layer.out_channels, device=device, dtype=dtype)

    # Bias contribution for zero-padding convolution of a spatial constant.
    kernel_sum = old_w.sum(dim=(2, 3))
    new_b = old_b + kernel_sum @ bias
    layer.weight.mul_(scale.view(1, -1, 1, 1))
    if layer.bias is None:
        layer.bias = nn.Parameter(new_b)
    else:
        layer.bias.copy_(new_b)
    return layer
