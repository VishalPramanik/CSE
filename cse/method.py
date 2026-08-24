from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import scipy.linalg
import torch

from .config import CSEConfig


@dataclass
class LayerEdit:
    """Paper-aligned CSE state for one encoder block/layer."""

    mean: torch.Tensor
    std: torch.Tensor
    salience: torch.Tensor
    selected: torch.Tensor
    attenuation: torch.Tensor
    scale: torch.Tensor
    bias: torch.Tensor
    eigenvalues: torch.Tensor

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "LayerEdit":
        def move(x: torch.Tensor) -> torch.Tensor:
            return x.to(device=device, dtype=dtype if x.is_floating_point() else None)

        return LayerEdit(
            mean=move(self.mean),
            std=move(self.std),
            salience=move(self.salience),
            selected=move(self.selected),
            attenuation=move(self.attenuation),
            scale=move(self.scale),
            bias=move(self.bias),
            eigenvalues=move(self.eigenvalues),
        )

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "mean": self.mean,
            "std": self.std,
            "salience": self.salience,
            "selected": self.selected,
            "attenuation": self.attenuation,
            "scale": self.scale,
            "bias": self.bias,
            "eigenvalues": self.eigenvalues,
        }


class ContrastiveSubnetErasure:
    """Training-free CSE implementation following Sec. 3 and Algorithm 1.

    The implementation uses the paper's three stages:
      1) joint standardization,
      2) generalized-eigenvalue subnet discovery,
      3) calibrated per-channel attenuation.

    Main-text precedence is used where the paper is ambiguous: attenuation is
    applied only to channels selected by the compact-subnet coverage rule;
    unselected channels are preserved exactly.
    """

    def __init__(self, config: CSEConfig | None = None) -> None:
        self.config = config or CSEConfig()
        self.config.validate()

    @staticmethod
    def _check_features(name: str, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if x.ndim != 2:
            raise ValueError(f"{name} must have shape [N, D], got {tuple(x.shape)}")
        if x.shape[0] < 2:
            raise ValueError(f"{name} must contain at least two samples")
        if x.shape[1] < 1:
            raise ValueError(f"{name} must contain at least one channel")
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} contains NaN/Inf")
        return x.detach().to(device="cpu", dtype=torch.float64)

    def fit_layer(self, target: torch.Tensor, background: torch.Tensor) -> LayerEdit:
        """Fit one layer's CSE edit from pooled target/background features.

        Equations correspond to the paper as follows:
          Eqs. (1)-(3): joint mean/std and standardization
          Eqs. (4)-(6): target/background second moments + generalized EVP
          Eqs. (7)-(8): eigenvalue-weighted channel salience + compact subset
          Eqs. (9)-(12): attenuation, scale and mean-compensating bias
        """

        target = self._check_features("target", target)
        background = self._check_features("background", background)
        if target.shape[1] != background.shape[1]:
            raise ValueError("target and background feature dimensions must match")

        cfg = self.config
        joint = torch.cat([target, background], dim=0)

        # Stage 1: Eqs. (1)-(3). Population variance is used exactly as written.
        mean = joint.mean(dim=0)
        std = torch.sqrt(((joint - mean) ** 2).mean(dim=0) + cfg.epsilon)
        target_z = (target - mean) / std
        background_z = (background - mean) / std

        # Stage 2: Eq. (4). The manuscript writes 1/n * sum(h h^T), i.e. an
        # uncentered second moment in the jointly-standardized coordinates.
        sigma_t = (target_z.T @ target_z) / target_z.shape[0]
        sigma_b = (background_z.T @ background_z) / background_z.shape[0]

        d = sigma_t.shape[0]
        delta = cfg.alpha * torch.trace(sigma_b) / d
        sigma_b_reg = sigma_b + delta * torch.eye(d, dtype=sigma_b.dtype)

        # Eq. (6): solve Sigma_t v = rho Sigma_b_reg v. scipy.linalg.eigh is
        # used because both matrices are symmetric and Sigma_b_reg is PD in
        # normal paper settings.
        evals, evecs = scipy.linalg.eigh(
            sigma_t.numpy(),
            sigma_b_reg.numpy(),
            check_finite=True,
        )
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order], 0.0)
        evecs = evecs[:, order]

        # Appendix B.2 explicitly states Euclidean normalization for salience.
        norms = np.linalg.norm(evecs, axis=0, keepdims=True)
        norms = np.maximum(norms, np.finfo(evecs.dtype).eps)
        evecs = evecs / norms

        k = min(cfg.k_max, int(np.floor(cfg.eigen_fraction * d)))
        if k < 1:
            raise ValueError(
                f"CSE selected k=0 eigenvectors for d={d}; increase feature dimension "
                "or eigen_fraction."
            )

        rho = torch.from_numpy(evals[:k]).to(torch.float64)
        vec = torch.from_numpy(evecs[:, :k]).to(torch.float64)

        # Eq. (7): s_c = sum_j rho_j * v_j[c]^2.
        salience = ((vec**2) * rho.unsqueeze(0)).sum(dim=1)
        total = salience.sum()
        if not torch.isfinite(total) or total <= 0:
            raise RuntimeError("Degenerate salience: generalized eigenanalysis produced no positive mass")

        # Eq. (8): greedily choose the smallest prefix reaching tau_cov.
        sorted_idx = torch.argsort(salience, descending=True)
        cumulative = torch.cumsum(salience[sorted_idx], dim=0)
        threshold = cfg.coverage * total
        count = int(torch.searchsorted(cumulative, threshold).item()) + 1
        selected = sorted_idx[:count]

        # Eq. (9), applied only to selected channels as stated in Sec. 3.4.
        attenuation = torch.zeros_like(salience)
        selected_scores = salience[selected]
        beta_selected = (selected_scores - cfg.tau0) / (selected_scores + cfg.lambda0)
        beta_selected = torch.clamp(beta_selected, 0.0, 1.0)
        attenuation[selected] = beta_selected

        # Eqs. (10)-(12). S and A are diagonal, so M=S^{-1}AS=A. Therefore
        # scale=1-beta and bias=(I-M)mu=beta*mu exactly.
        scale = 1.0 - attenuation
        bias = attenuation * mean

        return LayerEdit(
            mean=mean.to(torch.float32),
            std=std.to(torch.float32),
            salience=salience.to(torch.float32),
            selected=selected.to(torch.long),
            attenuation=attenuation.to(torch.float32),
            scale=scale.to(torch.float32),
            bias=bias.to(torch.float32),
            eigenvalues=rho.to(torch.float32),
        )

    def fit(
        self,
        target_features: Mapping[str, torch.Tensor],
        background_features: Mapping[str, torch.Tensor],
    ) -> Dict[str, LayerEdit]:
        if set(target_features) != set(background_features):
            raise ValueError("target/background feature dictionaries must have identical layer keys")
        if not target_features:
            raise ValueError("at least one layer is required")
        return {
            layer: self.fit_layer(target_features[layer], background_features[layer])
            for layer in target_features
        }


def apply_pooled_edit(features: torch.Tensor, edit: LayerEdit) -> torch.Tensor:
    """Apply Eq. (12) to already-pooled [N, D] features."""
    if features.ndim != 2 or features.shape[1] != edit.scale.numel():
        raise ValueError("features must have shape [N, D] matching the edit")
    scale = edit.scale.to(device=features.device, dtype=features.dtype)
    bias = edit.bias.to(device=features.device, dtype=features.dtype)
    return features * scale + bias
