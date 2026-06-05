"""General-purpose CSE command-line runner.

Apply CSE to an arbitrary classifier given (a) a target image folder
(the concept to forget) and (b) a non-target image folder (the related
classes to preserve), then report unlearning metrics on held-out splits.

This is a thin, framework-level wrapper: it expects an
``torchvision.datasets.ImageFolder``-style directory layout and a model
constructor importable as ``module:function``. It is intended as a template
to adapt CSE to your own pipeline.

Example
-------
    python scripts/run_unlearning.py \
        --model torchvision.models:resnet18 \
        --layer layer4 --channel-dim 1 \
        --target-dir  data/forget \
        --background-dir data/retain_related \
        --forget-test data/forget_test \
        --retain-test data/retain_test
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader

from cse import CSE, CSEConfig
from cse.eval import accuracy
from cse.utils import set_seed


def load_model(spec: str, num_classes: int | None) -> nn.Module:
    module_name, fn_name = spec.split(":")
    fn = getattr(importlib.import_module(module_name), fn_name)
    model = fn()
    return model


def image_folder_loader(path: str, batch_size: int):
    import torchvision
    import torchvision.transforms as T

    tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = torchvision.datasets.ImageFolder(path, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run CSE on a model + image folders")
    ap.add_argument("--model", required=True, help="constructor as 'module:function'")
    ap.add_argument("--checkpoint", default=None, help="optional state_dict to load")
    ap.add_argument("--layer", required=True, help="dotted module path to edit")
    ap.add_argument("--channel-dim", type=int, default=None)
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--background-dir", required=True)
    ap.add_argument("--forget-test", default=None)
    ap.add_argument("--retain-test", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    # CSE hyperparameters (paper defaults).
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--k-max", type=int, default=50)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--tau-cov", type=float, default=0.85)
    ap.add_argument("--tau0", type=float, default=0.1)
    ap.add_argument("--lambda0", type=float, default=0.5)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model, None).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    channel_dim = {args.layer: args.channel_dim} if args.channel_dim is not None else {}
    config = CSEConfig(
        alpha=args.alpha, k_max=args.k_max, beta=args.beta,
        tau_cov=args.tau_cov, tau0=args.tau0, lambda0=args.lambda0,
        channel_dim=channel_dim,
    )

    target_loader = image_folder_loader(args.target_dir, args.batch_size)
    background_loader = image_folder_loader(args.background_dir, args.batch_size)

    editor = CSE(model, layers=[args.layer], config=config)
    editor.fit(target_loader, background_loader, device=device)

    def report(tag):
        if args.forget_test:
            ft = accuracy(model, image_folder_loader(args.forget_test, args.batch_size), device)
            print(f"  [{tag}] forget-test acc  = {ft:.3f}")
        if args.retain_test:
            rt = accuracy(model, image_folder_loader(args.retain_test, args.batch_size), device)
            print(f"  [{tag}] retain-test acc  = {rt:.3f}")

    print("Before CSE:")
    report("orig")
    editor.apply()
    print(editor.summary())
    print("After CSE:")
    report("cse")


if __name__ == "__main__":
    main()
