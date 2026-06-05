"""Stage 3: Subnet attenuation (Sec. 3.4).

Each selected channel is damped by an attenuation factor derived from its
discriminative score (Eq. 9). Because the diagonal attenuation matrix and
the diagonal standardizing matrix commute, ``M = S^{-1} A S = A`` and the
runtime edit reduces to a per-channel affine map (Eq. 26 / Alg. 1, lines
17-20):

    ``h_att = scale (.) h + bias``,    scale_c = 1 - beta_c,   bias_c = beta_c * mu_c

For non-selected channels ``beta_c = 0`` (full preservation). The map is
applied at block output via a forward hook, which is mathematically
identical to folding the per-channel scales/biases into the following
linear/conv layer (Alg. 1, line 22) but keeps the computational graph
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .features import infer_channel_dim


@dataclass
class AttenuationParams:
    """Per-channel runtime affine parameters for one layer.

    scale: ``(C,)`` multiplicative factors ``a_c = 1 - beta_c``
    bias:  ``(C,)`` additive terms ``beta_c * mu_c``
    """

    scale: torch.Tensor
    bias: torch.Tensor


def build_attenuation(
    salience: torch.Tensor,
    selected: torch.Tensor,
    mu: torch.Tensor,
    tau0: float = 0.1,
    lambda0: float = 0.5,
) -> AttenuationParams:
    """Compute per-channel scale/bias from salience scores (Eqs. 9, 10, 26).

    Only channels in ``selected`` are attenuated; all others are preserved
    exactly (``scale = 1``, ``bias = 0``).
    """
    salience = salience.double()
    mu = mu.double()

    # beta_c = clip_{[0,1]}((s_c - tau0) / (s_c + lambda0))   (Eq. 9)
    beta = torch.clamp((salience - tau0) / (salience + lambda0), min=0.0, max=1.0)
    beta = torch.where(selected, beta, torch.zeros_like(beta))

    scale = 1.0 - beta              # a_c = 1 - beta_c   (Eq. 10 diagonal)
    bias = beta * mu                # (I - M) mu, diagonal case  (Eq. 26)
    return AttenuationParams(scale=scale, bias=bias)


class AttenuationHook:
    """Applies a per-channel affine edit to a module's output at runtime."""

    def __init__(self, params: AttenuationParams, channel_dim: int | None = None):
        self.scale = params.scale.float()
        self.bias = params.bias.float()
        self.channel_dim = channel_dim
        self._handle = None

    def _reshape(self, vec: torch.Tensor, tensor: torch.Tensor, cdim: int) -> torch.Tensor:
        shape = [1] * tensor.dim()
        shape[cdim % tensor.dim()] = vec.numel()
        return vec.view(*shape).to(tensor.dtype).to(tensor.device)

    def __call__(self, _module, _inp, out):
        single = not isinstance(out, (tuple, list))
        tensor = out if single else out[0]
        cdim = self.channel_dim if self.channel_dim is not None else infer_channel_dim(tensor)
        scale = self._reshape(self.scale, tensor, cdim)
        bias = self._reshape(self.bias, tensor, cdim)
        edited = tensor * scale + bias
        if single:
            return edited
        return (edited, *out[1:])

    def register(self, module: nn.Module) -> "AttenuationHook":
        self._handle = module.register_forward_hook(self)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
