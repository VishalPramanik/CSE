#!/usr/bin/env python3
"""
Contrastive Subnet Erasure (CSE) -- Main Entry Point.

Usage:
    python main.py --mode demo
    python main.py --mode single --backbone resnet18 --dataset cifar10 --forget airplane
    python main.py --mode multi --dataset cifar100 --forget castle keyboard telephone
    python main.py --mode benchmark
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.default import CSEConfig, ExperimentConfig
from src.cross_eval import run_single_experiment, run_multi_seed, save_results
from src.utils import setup_logging, get_device


def parse_args():
    p = argparse.ArgumentParser(
        description="Contrastive Subnet Erasure (CSE) for Class-Level Unlearning")
    p.add_argument("--mode", default="demo", choices=["demo", "single", "multi", "benchmark"])
    p.add_argument("--backbone", default="resnet18",
                   choices=["resnet18", "efficientnet_b0", "swin_t"])
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--forget", nargs="+", default=["airplane"])
    p.add_argument("--nontarget", nargs="*", default=None)
    p.add_argument("--nontarget-fraction", type=float, default=0.10)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--k-max", type=int, default=50)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--tau-cov", type=float, default=0.85)
    p.add_argument("--tau-0", type=float, default=0.1)
    p.add_argument("--lambda-0", type=float, default=0.5)
    p.add_argument("--num-seeds", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output-dir", default="./results")
    p.add_argument("--device", default="cuda")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def build_config(args):
    nc = 10 if args.dataset == "cifar10" else 100
    return ExperimentConfig(
        cse=CSEConfig(alpha=args.alpha, k_max=args.k_max, beta=args.beta,
                      tau_cov=args.tau_cov, tau_0=args.tau_0, lambda_0=args.lambda_0),
        backbone=args.backbone, pretrained=True, num_classes=nc,
        eval_dataset=args.dataset, source_dataset=args.dataset,
        data_root=args.data_root, forget_classes=args.forget,
        nontarget_classes=args.nontarget, nontarget_fraction=args.nontarget_fraction,
        batch_size=args.batch_size, num_workers=args.num_workers,
        num_seeds=args.num_seeds, output_dir=args.output_dir,
        device=args.device, verbose=not args.quiet,
    )


def run_demo(args):
    log = logging.getLogger("CSE")
    log.info("=" * 60)
    log.info("CSE Demo: Forgetting 'airplane' on CIFAR-10 (ResNet-18)")
    log.info("=" * 60)

    cfg = ExperimentConfig(
        backbone="resnet18", eval_dataset="cifar10",
        forget_classes=["airplane"], nontarget_classes=["bird", "ship"],
        data_root=args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers, output_dir=args.output_dir, device=args.device,
    )
    m = run_single_experiment(cfg, seed=42)
    save_results(m, cfg.output_dir, "demo_results.json")
    log.info("\nDemo complete!")
    log.info("  Accft (down): %.4f  Accrt (up): %.4f  H-Mean (up): %.4f  MIA (down): %.4f",
             m["Accft"], m["Accrt"], m["H-Mean"], m["MIA"])


def run_single(args):
    cfg = build_config(args)
    if args.num_seeds > 1:
        m = run_multi_seed(cfg, seeds=list(range(42, 42 + args.num_seeds)))
    else:
        m = run_single_experiment(cfg, seed=42)
    fn = f"{cfg.backbone}_{cfg.eval_dataset}_{'_'.join(cfg.forget_classes)}.json"
    save_results(m, cfg.output_dir, fn)


def run_benchmark(args):
    log = logging.getLogger("CSE")
    probes = [
        {"dataset": "cifar10", "forget": ["airplane"], "nontarget": ["bird", "ship"]},
        {"dataset": "cifar10", "forget": ["truck"], "nontarget": ["automobile"]},
        {"dataset": "cifar10", "forget": ["cat"], "nontarget": ["dog", "deer"]},
    ]
    all_res = []
    for bb in ["resnet18", "efficientnet_b0", "swin_t"]:
        for pr in probes:
            log.info("\nBenchmark: %s | forget=%s on %s", bb, pr["forget"], pr["dataset"])
            nc = 10 if pr["dataset"] == "cifar10" else 100
            cfg = ExperimentConfig(
                backbone=bb, eval_dataset=pr["dataset"], forget_classes=pr["forget"],
                nontarget_classes=pr["nontarget"], num_classes=nc,
                data_root=args.data_root, batch_size=args.batch_size,
                num_workers=args.num_workers, output_dir=args.output_dir, device=args.device,
            )
            all_res.append(run_single_experiment(cfg, seed=42))
    save_results({"experiments": all_res}, args.output_dir, "benchmark_results.json")


def main():
    args = parse_args()
    setup_logging(verbose=not args.quiet)
    log = logging.getLogger("CSE")
    log.info("Device: %s", get_device(args.device))
    {"demo": run_demo, "single": run_single, "multi": run_single,
     "benchmark": run_benchmark}[args.mode](args)


if __name__ == "__main__":
    main()
