"""Reproduce the single-class cross-dataset CSE result on CIFAR-10.

This follows the paper's protocol (Sec. 4, Appendix C): the target class is
defined on a *source* dataset and forgetting is measured on a *disjoint*
evaluation set, using a pretrained backbone whose encoder is edited by CSE
while the classification head is left unchanged.

The default probe forgets the CIFAR-10 ``airplane`` class. The non-target
set ``D_b`` is built from 10% of semantically related classes (``bird``,
``ship``), exactly as in the paper's default configuration (Table 5, row 1).

This script downloads CIFAR-10 and pretrained ImageNet weights via
torchvision, so it requires internet access and is intended to be run on the
user's machine / cluster (not in a sandbox). Reported metrics are
Acc_f, Acc_ft, Acc_r, Acc_rt, H-Mean, and MIA (Appendix C.3).

Example
-------
    python scripts/reproduce_cifar10.py --backbone resnet18 \
        --forget-class airplane --related bird ship --data ./data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from cse import CSE, CSEConfig
from cse.eval import accuracy, membership_inference
from cse.eval.metrics import UnlearningReport
from cse.utils import indices_for_labels, sample_fraction, set_seed

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# A reasonable default edit point (deepest encoder block) per backbone, and the
# channel axis of that block's output. These can be overridden on the CLI.
BACKBONES = {
    "resnet18": dict(layer="layer4", channel_dim=1),
    "efficientnet_b0": dict(layer="features.8", channel_dim=1),
    "swin_t": dict(layer="features.7", channel_dim=-1),
}


def build_model(backbone: str, num_classes: int = 10) -> nn.Module:
    import torchvision

    if backbone == "resnet18":
        model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif backbone == "swin_t":
        model = torchvision.models.swin_t(weights="IMAGENET1K_V1")
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone '{backbone}'")
    return model


def cifar10_splits(root: str):
    import torchvision
    import torchvision.transforms as T

    tf = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tf)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tf)
    return train, test


def finetune_head(model, loader, device, epochs=2, lr=1e-2):
    """Briefly adapt the (reinitialized) head to CIFAR-10; encoder stays frozen."""
    for p in model.parameters():
        p.requires_grad_(False)
    head = [m for m in (getattr(model, "fc", None),
                        getattr(model, "head", None),
                        getattr(model, "classifier", None)) if m is not None][0]
    for p in head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=lr, momentum=0.9)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            nn.functional.cross_entropy(model(x.to(device)), y.to(device)).backward()
            opt.step()
    model.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description="CSE CIFAR-10 cross-dataset reproduction")
    ap.add_argument("--backbone", default="resnet18", choices=list(BACKBONES))
    ap.add_argument("--forget-class", default="airplane", choices=CIFAR10_CLASSES)
    ap.add_argument("--related", nargs="+", default=["bird", "ship"],
                    help="semantically related non-target classes for D_b")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--layer", default=None, help="override the edit layer")
    ap.add_argument("--head-epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, test = cifar10_splits(args.data)
    train_labels = list(train.targets)
    test_labels = list(test.targets)

    target_id = CIFAR10_CLASSES.index(args.forget_class)
    related_ids = [CIFAR10_CLASSES.index(c) for c in args.related]

    # --- Build target / background / retain / forget subsets (Appendix C.2). ---
    forget_train_idx = indices_for_labels(train_labels, [target_id])
    forget_test_idx = indices_for_labels(test_labels, [target_id])
    retain_train_idx = [i for i, y in enumerate(train_labels) if y != target_id]
    retain_test_idx = [i for i, y in enumerate(test_labels) if y != target_id]

    # D_b: 10% of related non-target classes (paper default).
    related_train_idx = indices_for_labels(train_labels, related_ids)
    background_idx = sample_fraction(related_train_idx, fraction=0.10, seed=args.seed)

    def loader(dataset, idx, bs=64):
        return DataLoader(Subset(dataset, idx), batch_size=bs, shuffle=False)

    model = build_model(args.backbone).to(device)
    # Adapt the fresh head to CIFAR-10 (encoder frozen), so the model is a
    # meaningful classifier before unlearning.
    finetune_head(model, loader(train, retain_train_idx + forget_train_idx, bs=128),
                  device, epochs=args.head_epochs)

    cfg = BACKBONES[args.backbone]
    layer = args.layer or cfg["layer"]
    config = CSEConfig(channel_dim={layer: cfg["channel_dim"]})

    def evaluate() -> UnlearningReport:
        return UnlearningReport(
            acc_f=accuracy(model, loader(train, forget_train_idx), device),
            acc_ft=accuracy(model, loader(test, forget_test_idx), device),
            acc_r=accuracy(model, loader(train, retain_train_idx), device),
            acc_rt=accuracy(model, loader(test, retain_test_idx), device),
            mia=membership_inference(
                model, loader(train, forget_train_idx),
                loader(test, forget_test_idx), device, seed=args.seed),
        )

    print(f"Backbone={args.backbone}  forget='{args.forget_class}'  "
          f"D_b={args.related} (10%)  edit-layer={layer}")
    print("Original :", evaluate().as_row())

    editor = CSE(model, layers=[layer], config=config)
    editor.fit(loader(train, forget_train_idx), loader(train, background_idx), device=device)
    editor.apply()
    print(editor.summary())
    print("CSE      :", evaluate().as_row())


if __name__ == "__main__":
    main()
