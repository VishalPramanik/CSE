"""
Contrastive Subnet Erasure (CSE) -- Core Algorithm.

Implements the three-stage pipeline from Section 3:
    Stage 1: Feature Extraction & Standardization (Eq. 1-3)
    Stage 2: Contrastive Subnet Discovery (Eq. 4-8)
    Stage 3: Subnet Attenuation & Weight Folding (Eq. 9-14)

Reference: Algorithm 1 in Appendix A.
"""

import logging
from typing import Dict, List
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.linalg import eigh

from .models import FeatureExtractor

logger = logging.getLogger("CSE")


@dataclass
class SubnetInfo:
    """Stores per-layer subnet discovery results."""
    layer_name: str
    channel_scores: np.ndarray       # s_c for all channels
    selected_channels: List[int]     # C: selected channel indices
    attenuation_factors: np.ndarray  # beta_c for all channels
    eigenvalues: np.ndarray          # Top-k eigenvalues rho_j
    n_channels: int                  # Total channels in layer
    n_selected: int                  # |C|


class ContrastiveSubnetErasure:
    """
    Training-free encoder-centric unlearning via contrastive subnet discovery.

    Usage::

        cse = ContrastiveSubnetErasure(model, layer_names)
        cse.fit(target_loader, nontarget_loader)   # Stages 1-2
        cse.apply()                                 # Stage 3
    """

    def __init__(
        self,
        model: FeatureExtractor,
        layer_names: List[str],
        alpha: float = 0.01,
        k_max: int = 50,
        beta: float = 0.3,
        tau_cov: float = 0.85,
        tau_0: float = 0.1,
        lambda_0: float = 0.5,
        epsilon: float = 1e-6,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.layer_names = layer_names
        self.alpha = alpha
        self.k_max = k_max
        self.beta = beta
        self.tau_cov = tau_cov
        self.tau_0 = tau_0
        self.lambda_0 = lambda_0
        self.epsilon = epsilon
        self.device = device

        self.subnet_info: Dict[str, SubnetInfo] = {}
        self._layer_stats: Dict[str, Dict] = {}
        self._attenuation_hooks: list = []

    # ─────────────────────────────────────────────────────
    # Stage 1: Feature Extraction & Standardization
    # ─────────────────────────────────────────────────────

    @torch.no_grad()
    def _extract_features(self, loader: DataLoader) -> Dict[str, torch.Tensor]:
        """Extract pooled features from all target layers."""
        self.model.eval()
        all_features: Dict[str, list] = {n: [] for n in self.layer_names}

        for images, _ in loader:
            images = images.to(self.device)
            self.model(images)
            for name in self.layer_names:
                all_features[name].append(self.model.features[name].cpu())

        return {n: torch.cat(fs, dim=0) for n, fs in all_features.items()}

    def _compute_statistics(
        self,
        target_feats: Dict[str, torch.Tensor],
        nontarget_feats: Dict[str, torch.Tensor],
    ) -> Dict[str, Dict]:
        """Joint mean and std across both datasets (Eq. 1-2)."""
        stats = {}
        for name in self.layer_names:
            combined = torch.cat([target_feats[name], nontarget_feats[name]], dim=0)
            mu = combined.mean(dim=0)
            sigma = (combined.var(dim=0, correction=0) + self.epsilon).sqrt()  # Eq. 2: population var
            stats[name] = {"mu": mu, "sigma": sigma}
        return stats

    @staticmethod
    def _standardize(features: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Standardize: h_hat = (h - mu) / sigma  (Eq. 3)."""
        return (features - mu.unsqueeze(0)) / sigma.unsqueeze(0)

    # ─────────────────────────────────────────────────────
    # Stage 2: Contrastive Subnet Discovery
    # ─────────────────────────────────────────────────────

    def _discover_subnet(
        self,
        target_std: np.ndarray,
        nontarget_std: np.ndarray,
        layer_name: str,
    ) -> SubnetInfo:
        """Identify discriminative channels via generalized eigenanalysis (Eq. 4-8)."""
        nt, d = target_std.shape
        nb = nontarget_std.shape[0]

        # Covariance matrices (Eq. 4)
        Sigma_t = (target_std.T @ target_std) / max(nt, 1)
        Sigma_b = (nontarget_std.T @ nontarget_std) / max(nb, 1)

        # Regularize background (Eq. 6)
        delta = self.alpha * np.trace(Sigma_b) / max(d, 1)
        Sigma_b_reg = Sigma_b + delta * np.eye(d)

        # Solve generalized eigenvalue problem
        eigenvalues, eigenvectors = eigh(Sigma_t, Sigma_b_reg)
        eigenvalues = eigenvalues[::-1].copy()
        eigenvectors = eigenvectors[:, ::-1].copy()

        # Top-k (Eq. 7 bounds)
        k_ell = max(1, min(self.k_max, int(self.beta * d)))
        top_evals = eigenvalues[:k_ell]
        top_evecs = eigenvectors[:, :k_ell]

        # Normalize eigenvectors
        norms = np.linalg.norm(top_evecs, axis=0, keepdims=True)
        top_evecs = top_evecs / np.maximum(norms, 1e-12)

        # Channel salience scores (Eq. 7)
        channel_scores = np.zeros(d)
        for j in range(k_ell):
            channel_scores += top_evals[j] * (top_evecs[:, j] ** 2)

        # Greedy channel selection (Eq. 8)
        total = channel_scores.sum()
        if total < 1e-12:
            selected = list(range(min(d, 5)))
        else:
            sorted_idx = np.argsort(-channel_scores)
            selected = []
            cumsum = 0.0
            for idx in sorted_idx:
                selected.append(int(idx))
                cumsum += channel_scores[idx]
                if cumsum >= self.tau_cov * total:
                    break

        # Attenuation factors (Eq. 9)
        beta_c = np.clip(
            (channel_scores - self.tau_0) / (channel_scores + self.lambda_0),
            0.0, 1.0,
        )
        mask = np.zeros(d)
        mask[selected] = 1.0
        beta_c = beta_c * mask

        return SubnetInfo(
            layer_name=layer_name,
            channel_scores=channel_scores,
            selected_channels=sorted(selected),
            attenuation_factors=beta_c,
            eigenvalues=top_evals,
            n_channels=d,
            n_selected=len(selected),
        )

    # ─────────────────────────────────────────────────────
    # Stage 3: Subnet Attenuation
    # ─────────────────────────────────────────────────────

    def _apply_attenuation(self) -> None:
        """Fold per-channel attenuation into the model via hooks (Eq. 10-14)."""
        named = dict(self.model.backbone.named_modules())

        for layer_name in self.layer_names:
            if layer_name not in self.subnet_info:
                continue

            info = self.subnet_info[layer_name]
            stats = self._layer_stats[layer_name]
            mu = stats["mu"].numpy()
            beta_c = info.attenuation_factors

            # scale = 1 - beta_c;  bias = beta_c * mu
            scale = torch.tensor(1.0 - beta_c, dtype=torch.float32).to(self.device)
            bias = torch.tensor(beta_c * mu, dtype=torch.float32).to(self.device)

            module = named[layer_name]

            def make_hook(s, b):
                def hook(mod, inp, out):
                    if out.dim() == 4:
                        return out * s.view(1, -1, 1, 1) + b.view(1, -1, 1, 1)
                    elif out.dim() == 3:
                        return out * s.view(1, 1, -1) + b.view(1, 1, -1)
                    else:
                        return out * s.view(1, -1) + b.view(1, -1)
                return hook

            h = module.register_forward_hook(make_hook(scale, bias))
            self._attenuation_hooks.append(h)

        logger.info("Attenuation folded into model via forward hooks.")

    # ─────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────

    def fit(
        self,
        target_loader: DataLoader,
        nontarget_loader: DataLoader,
    ) -> Dict[str, SubnetInfo]:
        """Run Stages 1-2: extract, standardize, discover subnets."""
        logger.info("Stage 1: Extracting and standardizing features...")
        target_feats = self._extract_features(target_loader)
        nontarget_feats = self._extract_features(nontarget_loader)
        self._layer_stats = self._compute_statistics(target_feats, nontarget_feats)

        logger.info("Stage 2: Contrastive subnet discovery...")
        for name in self.layer_names:
            st = self._layer_stats[name]
            ft_std = self._standardize(target_feats[name], st["mu"], st["sigma"]).numpy()
            fb_std = self._standardize(nontarget_feats[name], st["mu"], st["sigma"]).numpy()
            info = self._discover_subnet(ft_std, fb_std, name)
            self.subnet_info[name] = info
            logger.info(
                "  %s: %d/%d channels selected (%.1f%%), top eigenvalue=%.4f",
                name, info.n_selected, info.n_channels,
                100 * info.n_selected / info.n_channels, info.eigenvalues[0],
            )
        return self.subnet_info

    def apply(self) -> None:
        """Stage 3: Apply subnet attenuation. Must call fit() first."""
        if not self.subnet_info:
            raise RuntimeError("Call fit() before apply().")
        logger.info("Stage 3: Applying subnet attenuation...")
        self._apply_attenuation()
        total_sel = sum(i.n_selected for i in self.subnet_info.values())
        total_ch = sum(i.n_channels for i in self.subnet_info.values())
        logger.info("CSE complete: %d/%d channels attenuated (%.1f%%).",
                     total_sel, total_ch, 100 * total_sel / total_ch)

    def fit_and_apply(
        self,
        target_loader: DataLoader,
        nontarget_loader: DataLoader,
    ) -> Dict[str, SubnetInfo]:
        """Convenience: fit() then apply()."""
        info = self.fit(target_loader, nontarget_loader)
        self.apply()
        return info

    def summary(self) -> str:
        lines = ["CSE Subnet Summary", "=" * 50]
        for name, info in self.subnet_info.items():
            lines.append(
                f"Layer: {name}  |  Channels: {info.n_selected}/{info.n_channels}  |  "
                f"Coverage: {self.tau_cov}  |  Top-rho: {info.eigenvalues[0]:.4f}"
            )
        return "\n".join(lines)
