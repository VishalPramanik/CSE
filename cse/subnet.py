"""Stage 2: Contrastive subnet discovery (Sec. 3.3).

Operating on standardized features, we find directions where the target
concept exhibits disproportionately high variance relative to non-target
concepts, score channels by their eigenvalue-weighted participation, and
greedily select a minimal subnet that covers a fraction ``tau_cov`` of the
discriminative mass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from scipy.linalg import eigh


@dataclass
class SubnetResult:
    """Outcome of contrastive subnet discovery for one layer.

    salience:        per-channel discriminative score ``s_c``  (Eq. 7), ``(C,)``
    selected:        boolean mask of the selected subnet ``C``  (Eq. 8), ``(C,)``
    eigenvalues:     the top-k generalized eigenvalues ``rho_j`` (descending)
    num_selected:    number of channels in the subnet
    """

    salience: torch.Tensor
    selected: torch.Tensor
    eigenvalues: torch.Tensor
    num_selected: int


def _empirical_covariance(feats: torch.Tensor) -> torch.Tensor:
    """Empirical (uncentered second-moment) covariance ``(1/n) X^T X``.

    Standardized features are already centered jointly, so we use the
    second-moment form from Eq. (4).
    """
    n = feats.shape[0]
    x = feats.double()
    return (x.t() @ x) / n


def contrastive_subnet(
    target_std: torch.Tensor,
    background_std: torch.Tensor,
    alpha: float = 0.01,
    k_max: int = 50,
    beta: float = 0.3,
    tau_cov: float = 0.85,
) -> SubnetResult:
    """Discover the contrastive subnet for a single layer.

    Solves the regularized generalized eigenvalue problem (Eq. 6)

        ``Sigma_t v = rho (Sigma_b + delta I) v``,    delta = alpha * tr(Sigma_b)/d

    scores channels with the eigenvalue-weighted participation (Eq. 7),
    and greedily selects the smallest channel set reaching coverage
    ``tau_cov`` (Eq. 8).
    """
    d = target_std.shape[1]
    sigma_t = _empirical_covariance(target_std)        # Eq. (4)
    sigma_b = _empirical_covariance(background_std)     # Eq. (4)

    delta = alpha * (torch.trace(sigma_b) / d)          # Eq. (6) regularizer
    sigma_b_reg = sigma_b + delta * torch.eye(d, dtype=torch.double)

    # Symmetric-definite generalized eigenproblem (ascending eigenvalues).
    eigvals, eigvecs = eigh(
        sigma_t.cpu().numpy(),
        sigma_b_reg.cpu().numpy(),
    )
    # Reorder to descending: rho_1 >= rho_2 >= ... >= rho_d.
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Eigenvector budget k_l = min(k_max, floor(beta * d_l))  (Eq. 7).
    k = int(min(k_max, int(np.floor(beta * d))))
    k = max(k, 1)
    rho = torch.from_numpy(np.ascontiguousarray(eigvals[:k])).double()
    vecs = torch.from_numpy(np.ascontiguousarray(eigvecs[:, :k])).double()

    # Salience requires Euclidean-unit eigenvectors (Sec. B.2, "Salience
    # conservation"); scipy returns Sigma_b-orthonormal vectors, so renormalize.
    vecs = vecs / (vecs.norm(dim=0, keepdim=True) + 1e-12)

    # Clamp tiny negative eigenvalues from numerical noise to keep scores >= 0.
    rho_clamped = torch.clamp(rho, min=0.0)

    # s_c = sum_j rho_j (v_j[c])^2   (Eq. 7)
    salience = (rho_clamped.view(1, k) * (vecs ** 2)).sum(dim=1)

    selected = _greedy_coverage(salience, tau_cov)
    return SubnetResult(
        salience=salience,
        selected=selected,
        eigenvalues=rho,
        num_selected=int(selected.sum().item()),
    )


def _greedy_coverage(salience: torch.Tensor, tau_cov: float) -> torch.Tensor:
    """Greedily select channels in descending salience until coverage met.

    Returns a boolean mask such that the cumulative salience of the
    selected channels is at least ``tau_cov`` of the total (Eq. 8).
    """
    total = salience.sum()
    selected = torch.zeros_like(salience, dtype=torch.bool)
    if total <= 0:
        return selected  # degenerate layer: nothing discriminative to remove

    order = torch.argsort(salience, descending=True)
    running = 0.0
    threshold = tau_cov * total.item()
    for idx in order.tolist():
        selected[idx] = True
        running += salience[idx].item()
        if running >= threshold:
            break
    return selected
