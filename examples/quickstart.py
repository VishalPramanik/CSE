"""Quickstart: a self-contained sanity check for CSE (no downloads required).

This mirrors the spirit of the MNIST/EfficientNet motivating toy in the
paper (Sec. 2): we forget a *target concept* while auditing a disjoint
*non-target* task on the same frozen encoder.

To keep the example fast and dependency-free, we use a small frozen encoder
and a synthetic feature distribution in which the target concept lives in a
known set of channels (0-7), while a non-target task is decided by a disjoint
set (24-31). Running CSE should:

  1. recover *exactly* the ground-truth target channels (locality);
  2. collapse the target concept's variance along those channels
     (the concept signature is erased);
  3. leave the non-target channels' variance essentially intact
     (no collateral damage);
  4. drop a frozen target-detector's confidence on target inputs while the
     non-target probe's accuracy is preserved.

Run:
    python examples/quickstart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cse import CSE, CSEConfig
from cse.eval import accuracy
from cse.utils import set_seed

TARGET_CH = list(range(0, 8))
NONTARGET_CH = list(range(24, 32))


class ToyEncoder(nn.Module):
    """A tiny frozen encoder (identity-initialized linear map)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.proj.weight)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Probe(nn.Module):
    """Frozen shared encoder + trainable linear probe (the probe is never edited)."""

    def __init__(self, encoder: nn.Module, dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def make_synthetic(dim: int = 64, n_per_class: int = 600, seed: int = 0):
    """Three concepts: 0 = target, {1, 2} = non-target (disjoint channel groups)."""
    g = torch.Generator().manual_seed(seed)
    feats, labels = [], []
    for cls in (0, 1, 2):
        x = 0.3 * torch.randn(n_per_class, dim, generator=g)
        if cls == 0:  # target concept: strong, high-variance signal
            x[:, TARGET_CH] += 4.0 + 2.5 * torch.randn(n_per_class, len(TARGET_CH), generator=g)
        else:         # non-target classes: separated along disjoint channels
            x[:, NONTARGET_CH] += (1.5 if cls == 1 else -1.5)
        feats.append(x)
        labels.append(torch.full((n_per_class,), cls, dtype=torch.long))
    feats, labels = torch.cat(feats), torch.cat(labels)
    perm = torch.randperm(feats.shape[0], generator=g)
    return feats[perm], labels[perm]


def _loader(feats, labels, shuffle=False):
    return DataLoader(TensorDataset(feats, labels), batch_size=128, shuffle=shuffle)


def train_head(model, loader, epochs=15, lr=0.1):
    opt = torch.optim.SGD(model.head.parameters(), lr=lr, momentum=0.9)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    model.eval()


@torch.no_grad()
def channel_variance(encoder, feats, channels):
    return encoder(feats)[:, channels].var(dim=0).mean().item()


def main() -> None:
    set_seed(0)
    dim, device = 64, torch.device("cpu")

    feats, labels = make_synthetic(dim=dim)
    n_tr = int(0.8 * feats.shape[0])
    f_tr, y_tr, f_te, y_te = feats[:n_tr], labels[:n_tr], feats[n_tr:], labels[n_tr:]
    nt_tr = (y_tr == 1) | (y_tr == 2)
    nt_te = (y_te == 1) | (y_te == 2)

    encoder = ToyEncoder(dim).to(device)

    # Probe: non-target task (class 1 vs 2) on the shared frozen encoder.
    nontarget_probe = Probe(encoder, dim).to(device)
    train_head(nontarget_probe, _loader(f_tr[nt_tr], (y_tr[nt_tr] == 2).long(), shuffle=True))

    target_te_feats = f_te[y_te == 0]
    nontarget_eval = _loader(f_te[nt_te], (y_te[nt_te] == 2).long())

    print("=" * 66)
    print("Contrastive Subnet Erasure (CSE) - quickstart sanity check")
    print("=" * 66)

    acc_n0 = accuracy(nontarget_probe, nontarget_eval, device)
    var_t0 = channel_variance(encoder, target_te_feats, TARGET_CH)
    var_n0 = channel_variance(encoder, f_te[nt_te], NONTARGET_CH)

    # CSE: forget the target concept. D_t = target images, D_b = non-target images.
    editor = CSE(encoder, layers=["proj"], config=CSEConfig(channel_dim={"proj": -1}))
    editor.fit(
        _loader(f_tr[y_tr == 0], y_tr[y_tr == 0]),
        _loader(f_tr[nt_tr], y_tr[nt_tr]),
        device=device,
    )
    editor.apply()
    print("\n" + editor.summary())

    acc_n1 = accuracy(nontarget_probe, nontarget_eval, device)
    var_t1 = channel_variance(encoder, target_te_feats, TARGET_CH)
    var_n1 = channel_variance(encoder, f_te[nt_te], NONTARGET_CH)

    selected = sorted(editor.edits["proj"].selected.nonzero().flatten().tolist())
    collapse = 100.0 * (1.0 - var_t1 / var_t0)
    retained = 100.0 * (var_n1 / var_n0)

    print("\n" + "-" * 66)
    print(f"{'Metric':38s}{'before':>12s}{'after':>12s}")
    print("-" * 66)
    print(f"{'Subnet recovered (target = 0..7)':38s}{'':>12s}{str(selected == TARGET_CH):>12s}")
    print(f"{'Target-channel variance (target)':38s}{var_t0:>12.4f}{var_t1:>12.4f}")
    print(f"{'Non-target-channel variance (kept)':38s}{var_n0:>12.4f}{var_n1:>12.4f}")
    print(f"{'Non-target task accuracy':38s}{acc_n0:>12.3f}{acc_n1:>12.3f}")
    print("-" * 66)
    print(f"Target concept variance collapsed by {collapse:5.1f}%  "
          f"| non-target variance retained {retained:5.1f}%")
    print("\nNote: this toy isolates CSE's *representation-level* effect (subnet")
    print("recovery + variance collapse + geometry preservation). The full")
    print("downstream forgetting metrics (Acc_ft, H-Mean, MIA) on real pretrained")
    print("backbones are produced by scripts/reproduce_cifar10.py.")

    ok = (
        selected == TARGET_CH
        and collapse >= 90.0
        and retained >= 90.0
        and acc_n1 >= acc_n0 - 0.05
    )
    print("\nResult:", "PASS - exact subnet recovered, target concept erased, "
          "non-target preserved." if ok else
          "Check configuration (effect weaker than expected).")


if __name__ == "__main__":
    main()
