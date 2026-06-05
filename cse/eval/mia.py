"""Membership inference attack (Appendix C.3).

A simple loss-threshold attack quantifies residual memorization on the
forget set. Members are forget-train samples; non-members are an
equally sized pool of forget-test samples. A threshold is calibrated on
one half and evaluated on the other; ``MIA ~= 0.5`` on a balanced pool
indicates random guessing (successful forgetting).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def _per_sample_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Cross-entropy loss ``-log p(y|x)`` for every sample in ``loader``."""
    losses = []
    for images, targets in loader:
        logits = model(images.to(device))
        loss = F.cross_entropy(logits, targets.to(device), reduction="none")
        losses.append(loss.cpu())
    if not losses:
        return torch.empty(0)
    return torch.cat(losses, dim=0)


def membership_inference(
    model: nn.Module,
    member_loader: DataLoader,
    nonmember_loader: DataLoader,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> float:
    """Loss-threshold MIA success rate on a balanced member/non-member pool."""
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()

    member_loss = _per_sample_loss(model, member_loader, device)
    nonmember_loss = _per_sample_loss(model, nonmember_loader, device)

    if was_training:
        model.train()

    n = min(member_loss.numel(), nonmember_loss.numel())
    if n == 0:
        return float("nan")

    g = torch.Generator().manual_seed(seed)
    member_loss = member_loss[torch.randperm(member_loss.numel(), generator=g)[:n]]
    nonmember_loss = nonmember_loss[torch.randperm(nonmember_loss.numel(), generator=g)[:n]]

    losses = torch.cat([member_loss, nonmember_loss])
    labels = torch.cat([torch.ones(n), torch.zeros(n)])  # 1 = member

    # Split pool in half: calibrate threshold on the first half.
    perm = torch.randperm(2 * n, generator=g)
    losses, labels = losses[perm], labels[perm]
    half = n
    cal_loss, cal_lab = losses[:half], labels[:half]
    test_loss, test_lab = losses[half:], labels[half:]

    # Members tend to have lower loss; threshold maximizes calibration accuracy.
    candidates = torch.unique(cal_loss)
    best_thr, best_acc = candidates[0] if candidates.numel() else torch.tensor(0.0), -1.0
    for thr in candidates:
        pred = (cal_loss < thr).float()
        acc = (pred == cal_lab).float().mean().item()
        if acc > best_acc:
            best_acc, best_thr = acc, thr

    test_pred = (test_loss < best_thr).float()
    return (test_pred == test_lab).float().mean().item()
