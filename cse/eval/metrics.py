"""Evaluation metrics (Appendix C.3).

We report forget/retain accuracies on train and test, the forget-success
score ``F = 1 - Acc_ft``, and their harmonic mean (H-Mean), which is large
only when forgetting and retention are simultaneously high.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> float:
    """Top-1 classification accuracy of ``model`` on ``loader``."""
    if loader is None:
        return float("nan")
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()

    correct, total = 0, 0
    for images, targets in loader:
        logits = model(images.to(device))
        preds = logits.argmax(dim=1).cpu()
        correct += (preds == targets).sum().item()
        total += targets.numel()

    if was_training:
        model.train()
    return correct / max(total, 1)


def forget_success(acc_ft: float) -> float:
    """Forget-success score ``F = 1 - Acc_ft`` (Appendix C.3)."""
    return 1.0 - acc_ft


def h_mean(acc_ft: float, acc_rt: float) -> float:
    """Harmonic mean of forget-success and retain-test accuracy.

    ``H = 2 F * Acc_rt / (F + Acc_rt)`` with ``F = 1 - Acc_ft``.
    """
    f = forget_success(acc_ft)
    denom = f + acc_rt
    if denom <= 0:
        return 0.0
    return 2.0 * f * acc_rt / denom


@dataclass
class UnlearningReport:
    """Container for the standard unlearning metrics."""

    acc_f: float    # forget-train accuracy   (lower is better)
    acc_ft: float   # forget-test accuracy    (lower is better)
    acc_r: float    # retain-train accuracy   (higher is better)
    acc_rt: float   # retain-test accuracy    (higher is better)
    mia: float      # membership inference    (closer to 0.5 = better forgetting)

    @property
    def h_mean(self) -> float:
        return h_mean(self.acc_ft, self.acc_rt)

    def as_row(self) -> str:
        return (
            f"Acc_f={self.acc_f:.3f}  Acc_ft={self.acc_ft:.3f}  "
            f"Acc_r={self.acc_r:.3f}  Acc_rt={self.acc_rt:.3f}  "
            f"H-Mean={self.h_mean:.3f}  MIA={self.mia:.3f}"
        )
