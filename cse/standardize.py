"""Stage 1: Feature standardization (Sec. 3.2).

We compute the joint mean and per-channel standard deviation across the
union ``D_t U D_b`` and standardize features. Joint standardization is
crucial for an unbiased variance-ratio computation in Stage 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LayerStats:
    """Per-layer standardization statistics.

    mu:    joint channel means, shape ``(C,)``      (Eq. 1)
    sigma: joint channel stds,  shape ``(C,)``      (Eq. 2)
    """

    mu: torch.Tensor
    sigma: torch.Tensor

    def standardize(self, feats: torch.Tensor) -> torch.Tensor:
        """Apply ``(h - mu) / sigma`` channel-wise (Eq. 3)."""
        return (feats - self.mu) / self.sigma


def compute_joint_stats(
    target_feats: torch.Tensor,
    background_feats: torch.Tensor,
    eps: float = 1e-6,
) -> LayerStats:
    """Joint mean/std over ``D_t U D_b`` for one layer.

    Parameters
    ----------
    target_feats:
        ``(n_t, C)`` pooled features from the target set ``D_t``.
    background_feats:
        ``(n_b, C)`` pooled features from the non-target set ``D_b``.
    eps:
        Stability constant inside the square root (Eq. 2).
    """
    joint = torch.cat([target_feats, background_feats], dim=0).double()
    mu = joint.mean(dim=0)
    # Population variance over the union, matching Eq. (2).
    var = ((joint - mu) ** 2).mean(dim=0)
    sigma = torch.sqrt(var + eps)
    return LayerStats(mu=mu, sigma=sigma)
