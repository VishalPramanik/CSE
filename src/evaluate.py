"""
Evaluation metrics for machine unlearning.

Implements all metrics from Section 4:
    - Classification accuracy (Accf, Accft, Accr, Accrt)
    - Harmonic mean (H-Mean) between forgetting and retention
    - Membership Inference Attack (MIA) success rate
"""

import logging
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger("CSE")


class Evaluator:
    """
    Evaluates an unlearned model on all standard metrics.

    Usage::

        evaluator = Evaluator(model, device)
        metrics = evaluator.evaluate(ft_loader, ftest_loader, rt_loader, rtest_loader)
    """

    def __init__(self, model: nn.Module, device: torch.device = torch.device("cpu")):
        self.model = model
        self.device = device

    @torch.no_grad()
    def compute_accuracy(self, loader: DataLoader) -> float:
        """Top-1 classification accuracy."""
        self.model.eval()
        correct = total = 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            preds = self.model(images).argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        return correct / max(total, 1)

    @staticmethod
    def compute_hmean(forget_test_acc: float, retain_test_acc: float) -> float:
        """H-Mean = 2 * F * Accrt / (F + Accrt), where F = 1 - Accft."""
        f = 1.0 - forget_test_acc
        denom = f + retain_test_acc
        if denom < 1e-12:
            return 0.0
        return 2.0 * f * retain_test_acc / denom

    @torch.no_grad()
    def compute_mia(
        self,
        member_loader: DataLoader,
        nonmember_loader: DataLoader,
    ) -> float:
        """
        Membership Inference Attack via loss-threshold (Appendix C.3).

        Lower MIA = better unlearning (0.5 = random guessing = ideal).
        """
        self.model.eval()

        def _losses(loader):
            out = []
            for imgs, labs in loader:
                imgs, labs = imgs.to(self.device), labs.to(self.device)
                loss = F.cross_entropy(self.model(imgs), labs, reduction="none")
                out.append(loss.cpu().numpy())
            return np.concatenate(out) if out else np.array([])

        mem = _losses(member_loader)
        non = _losses(nonmember_loader)
        if len(mem) == 0 or len(non) == 0:
            return 0.5

        n = min(len(mem), len(non))
        mem, non = mem[:n], non[:n]
        all_losses = np.concatenate([mem, non])
        all_labels = np.concatenate([np.ones(n), np.zeros(n)])

        # Split into calibration / test
        idx = np.arange(2 * n)
        np.random.RandomState(42).shuffle(idx)
        cal, tst = idx[:n], idx[n:]

        # Find best threshold on calibration set
        thresholds = np.unique(all_losses[cal])
        best_acc, best_t = 0.5, thresholds[0] if len(thresholds) else 0.0
        for t in thresholds:
            acc = ((all_losses[cal] < t).astype(float) == all_labels[cal]).mean()
            if acc > best_acc:
                best_acc, best_t = acc, t

        # Evaluate on test set
        preds = (all_losses[tst] < best_t).astype(float)
        return float((preds == all_labels[tst]).mean())

    def evaluate(
        self,
        forget_train_loader: DataLoader,
        forget_test_loader: DataLoader,
        retain_train_loader: DataLoader,
        retain_test_loader: DataLoader,
    ) -> Dict[str, float]:
        """Run full evaluation suite. Returns dict with all metrics."""
        logger.info("Computing evaluation metrics...")
        accf = self.compute_accuracy(forget_train_loader)
        accft = self.compute_accuracy(forget_test_loader)
        accr = self.compute_accuracy(retain_train_loader)
        accrt = self.compute_accuracy(retain_test_loader)
        hmean = self.compute_hmean(accft, accrt)
        mia = self.compute_mia(forget_train_loader, forget_test_loader)

        metrics = {
            "Accf": accf, "Accft": accft,
            "Accr": accr, "Accrt": accrt,
            "H-Mean": hmean, "MIA": mia,
        }
        logger.info(
            "  Accft=%.4f (down)  Accrt=%.4f (up)  H-Mean=%.4f (up)  MIA=%.4f (down)",
            accft, accrt, hmean, mia,
        )
        return metrics


def evaluate_original(
    model: nn.Module,
    forget_test_loader: DataLoader,
    retain_test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Quick evaluation of the original (unedited) model."""
    ev = Evaluator(model, device)
    return {"Accft": ev.compute_accuracy(forget_test_loader),
            "Accrt": ev.compute_accuracy(retain_test_loader)}
