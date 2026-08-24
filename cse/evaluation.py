from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class AccuracyMetrics:
    forget_train: float
    forget_test: float
    retain_train: float
    retain_test: float

    @property
    def harmonic_mean(self) -> float:
        return harmonic_mean(self.forget_test, self.retain_test)


def harmonic_mean(forget_test_accuracy: float, retain_test_accuracy: float) -> float:
    """Paper metric: H = 2 F Acc_rt / (F + Acc_rt), F = 1 - Acc_ft."""
    f = 1.0 - float(forget_test_accuracy)
    r = float(retain_test_accuracy)
    return 0.0 if f + r == 0 else 2.0 * f * r / (f + r)


@torch.inference_mode()
def accuracy(model: nn.Module, loader: DataLoader, device: str | torch.device) -> float:
    model = model.to(device).eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = model(images).argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += labels.numel()
    if total == 0:
        raise ValueError("empty dataloader")
    return correct / total


@torch.inference_mode()
def per_sample_nll(model: nn.Module, loader: DataLoader, device: str | torch.device) -> np.ndarray:
    model = model.to(device).eval()
    values: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        values.append(loss.detach().cpu().numpy())
    if not values:
        raise ValueError("empty dataloader")
    return np.concatenate(values)


def loss_threshold_mia(
    member_losses: np.ndarray,
    nonmember_losses: np.ndarray,
    seed: int = 0,
) -> float:
    """Balanced loss-threshold MIA described in Appendix C.3.

    A balanced pool is built with n=min(#members, #nonmembers), then split 50/50.
    This operationalizes the paper's balanced-pool requirement even when the
    source train and test target sets have different sizes.
    """
    member = np.asarray(member_losses, dtype=float).ravel()
    nonmember = np.asarray(nonmember_losses, dtype=float).ravel()
    n = min(len(member), len(nonmember))
    if n < 2:
        raise ValueError("need at least two member and two non-member losses")
    rng = np.random.default_rng(seed)
    member = rng.choice(member, size=n, replace=False)
    nonmember = rng.choice(nonmember, size=n, replace=False)
    losses = np.concatenate([member, nonmember])
    labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    order = rng.permutation(len(losses))
    losses, labels = losses[order], labels[order]
    split = len(losses) // 2
    train_l, test_l = losses[:split], losses[split:]
    train_y, test_y = labels[:split], labels[split:]

    candidates = np.unique(train_l)
    if candidates.size == 1:
        candidates = np.array([candidates[0]])
    best_threshold = candidates[0]
    best_acc = -1.0
    for threshold in candidates:
        pred = (train_l < threshold).astype(int)
        acc = float((pred == train_y).mean())
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    return float(((test_l < best_threshold).astype(int) == test_y).mean())
