from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cse.config import CSEConfig
from cse.datasets import build_dataset, sample_background_by_classes, subset_by_classes
from cse.features import collect_features
from cse.method import ContrastiveSubnetErasure, apply_pooled_edit
from cse.models import apply_edits, build_model, get_model_spec
from cse.utils import load_config, set_seed


def smoke_test() -> None:
    """Dependency-light numerical check of the complete three-stage CSE core."""
    set_seed(7)
    n_t, n_b, d = 96, 96, 16
    background = torch.randn(n_b, d)
    target = torch.randn(n_t, d)
    # Inject target-specific variance into a compact subset of channels.
    target[:, :3] *= 4.0

    cse = ContrastiveSubnetErasure(CSEConfig())
    edit = cse.fit_layer(target, background)
    before = target.var(dim=0, unbiased=False)
    after = apply_pooled_edit(target, edit).var(dim=0, unbiased=False)

    if edit.selected.numel() == 0:
        raise RuntimeError("smoke test failed: no channels selected")
    if not torch.isfinite(after).all():
        raise RuntimeError("smoke test failed: non-finite edited features")

    print("CSE smoke test: PASS")
    print(f"selected channels: {edit.selected.tolist()}")
    print(f"mean target variance before/after: {before.mean():.4f} -> {after.mean():.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contrastive Subnet Erasure (CSE)")
    parser.add_argument("--smoke-test", action="store_true", help="run a local synthetic integrity test")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", choices=["resnet18", "efficientnet_b0", "swin_t"], default="resnet18")
    parser.add_argument("--weights", choices=["imagenet1k", "none"], default="imagenet1k")
    parser.add_argument("--target-dataset", choices=["cifar10", "cifar100", "imagenet", "lfw"])
    parser.add_argument("--target-root", type=str, default="./data")
    parser.add_argument("--target-classes", nargs="+")
    parser.add_argument("--background-dataset", choices=["cifar10", "cifar100", "imagenet", "lfw"])
    parser.add_argument("--background-root", type=str, default="./data")
    parser.add_argument("--background-classes", nargs="+")
    parser.add_argument("--download", action="store_true", help="allow torchvision downloads where supported")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-limit", type=int, default=None, help="optional development-only sample cap")
    parser.add_argument("--background-limit", type=int, default=None, help="optional development-only sample cap")
    parser.add_argument("--output", type=str, default="cse_edited_model.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test()
        return

    required = {
        "--target-dataset": args.target_dataset,
        "--target-classes": args.target_classes,
        "--background-dataset": args.background_dataset,
        "--background-classes": args.background_classes,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"Missing required arguments for image-data mode: {', '.join(missing)}")

    set_seed(args.seed)
    cfg = load_config(args.config)
    model = build_model(args.model, args.weights)
    spec = get_model_spec(args.model)

    target_ds = build_dataset(args.target_dataset, args.target_root, split="train", download=args.download)
    background_ds = build_dataset(args.background_dataset, args.background_root, split="train", download=args.download)
    target_subset = subset_by_classes(target_ds, args.target_classes)
    background_subset = sample_background_by_classes(
        background_ds,
        args.background_classes,
        fraction=cfg.non_target_fraction,
        seed=args.seed,
    )

    target_loader = DataLoader(
        target_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    background_loader = DataLoader(
        background_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    target_features = collect_features(
        model, target_loader, spec, args.device, max_samples=args.target_limit
    )
    background_features = collect_features(
        model, background_loader, spec, args.device, max_samples=args.background_limit
    )

    edits = ContrastiveSubnetErasure(cfg).fit(target_features, background_features)
    model = apply_edits(model, spec, edits)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": args.model,
            "weights": args.weights,
            "config": cfg.to_dict(),
            "target_dataset": args.target_dataset,
            "target_classes": args.target_classes,
            "background_dataset": args.background_dataset,
            "background_classes": args.background_classes,
            "model_state_dict": model.state_dict(),
            "cse_edits": {name: edit.state_dict() for name, edit in edits.items()},
        },
        output,
    )
    print(f"Saved CSE-edited model package to {output}")
    for name, edit in edits.items():
        print(f"  {name}: selected {edit.selected.numel()}/{edit.scale.numel()} channels")


if __name__ == "__main__":
    main()
