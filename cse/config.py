"""Configuration for Contrastive Subnet Erasure (CSE).

All defaults reproduce the hyperparameters reported in the paper
(Sec. 4 and Appendix C.4). Unless otherwise noted, the same values are
used for ResNet-18, EfficientNet-B0, and Swin-T with no per-backbone
tuning.

Reference
---------
Pramanik et al., "Selective Amnesia using Contrastive Subnet Erasure for
Class-Level Unlearning in Vision Models", CVPR 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class CSEConfig:
    """Hyperparameters for the CSE edit.

    Attributes
    ----------
    alpha:
        Regularization factor for the background covariance in the
        generalized eigenvalue problem, ``Sigma_b <- Sigma_b + alpha *
        trace(Sigma_b) / d * I`` (Eq. 6). Paper default: ``0.01``.
    k_max:
        Hard cap on the number of generalized eigenvectors used to score
        channels (Eq. 7, ``k_l = min(k_max, floor(beta * d_l))``).
        Paper default: ``50``.
    beta:
        Fraction of channels that bounds the eigenvector budget per layer
        (Eq. 7). Paper default: ``0.3``.
    tau_cov:
        Coverage threshold; the smallest channel subset whose cumulative
        salience reaches this fraction of the total is selected (Eq. 8).
        Paper default: ``0.85``.
    tau0:
        Minimum-score threshold in the attenuation transfer function
        (Eq. 9). Paper default: ``0.1``.
    lambda0:
        Transition-smoothness constant in the attenuation transfer
        function (Eq. 9). Paper default: ``0.5``.
    eps:
        Numerical-stability constant inside the per-channel std (Eq. 2).
        Paper default: ``1e-6``.
    nontarget_fraction:
        Fraction of images sampled per semantically related non-target
        class to build the background set ``D_b``. Paper default: ``0.10``.
    channel_dim:
        Mapping from layer name to the channel axis of that layer's output
        tensor. If a layer is absent, a heuristic is used (axis ``1`` for
        4-D conv maps, last axis for 3-D token tensors / 2-D vectors).
    """

    alpha: float = 0.01
    k_max: int = 50
    beta: float = 0.3
    tau_cov: float = 0.85
    tau0: float = 0.1
    lambda0: float = 0.5
    eps: float = 1e-6
    nontarget_fraction: float = 0.10
    channel_dim: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 < self.tau_cov < 1.0):
            raise ValueError(f"tau_cov must be in (0, 1), got {self.tau_cov}")
        if not (0.0 < self.beta <= 1.0):
            raise ValueError(f"beta must be in (0, 1], got {self.beta}")
        if self.k_max < 1:
            raise ValueError(f"k_max must be >= 1, got {self.k_max}")
        for name in ("alpha", "tau0", "lambda0", "eps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict) -> "CSEConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})
