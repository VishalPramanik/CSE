"""
Cross-dataset evaluation orchestrator.

Coordinates: model building -> dataset prep -> CSE -> evaluation.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from configs.default import ExperimentConfig
from .models import build_model
from .datasets import build_datasets
from .cse import ContrastiveSubnetErasure
from .evaluate import Evaluator, evaluate_original
from .utils import set_seed, get_device, ensure_dir, format_metrics

logger = logging.getLogger("CSE")


def run_single_experiment(
    config: ExperimentConfig,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Run one CSE unlearning experiment end-to-end.

    1. Build model and datasets
    2. Evaluate original model (baseline)
    3. Apply CSE (Stages 1-3)
    4. Evaluate unlearned model
    """
    set_seed(seed)
    device = get_device(config.device)

    logger.info("Building %s (pretrained=%s)...", config.backbone, config.pretrained)
    model, layer_names = build_model(
        backbone=config.backbone,
        pretrained=config.pretrained,
        num_classes=config.num_classes,
        device=device,
    )

    logger.info("Building cross-dataset protocol: forget=%s on %s...",
                config.forget_classes, config.eval_dataset)
    protocol = build_datasets(
        forget_classes=config.forget_classes,
        dataset_name=config.eval_dataset,
        data_root=config.data_root,
        nontarget_classes=config.nontarget_classes,
        nontarget_fraction=config.nontarget_fraction,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=seed,
    )
    logger.info("Dataset splits: %s", protocol.summary())

    logger.info("Evaluating original (unedited) model...")
    orig = evaluate_original(
        model, protocol.get_forget_test_loader(),
        protocol.get_retain_test_loader(), device,
    )
    logger.info("Original: %s", format_metrics(orig))

    t0 = time.time()
    cse = ContrastiveSubnetErasure(
        model=model, layer_names=layer_names,
        alpha=config.cse.alpha, k_max=config.cse.k_max,
        beta=config.cse.beta, tau_cov=config.cse.tau_cov,
        tau_0=config.cse.tau_0, lambda_0=config.cse.lambda_0,
        epsilon=config.cse.epsilon, device=device,
    )
    cse.fit_and_apply(
        target_loader=protocol.get_target_loader(),
        nontarget_loader=protocol.get_nontarget_loader(),
    )
    elapsed = time.time() - t0
    logger.info("CSE completed in %.1fs", elapsed)
    logger.info(cse.summary())

    evaluator = Evaluator(model, device)
    metrics = evaluator.evaluate(
        forget_train_loader=protocol.get_forget_train_loader(),
        forget_test_loader=protocol.get_forget_test_loader(),
        retain_train_loader=protocol.get_retain_train_loader(),
        retain_test_loader=protocol.get_retain_test_loader(),
    )
    metrics.update({
        "elapsed_seconds": elapsed,
        "backbone": config.backbone,
        "forget_classes": config.forget_classes,
        "dataset": config.eval_dataset,
        "seed": seed,
        "original_Accft": orig["Accft"],
        "original_Accrt": orig["Accrt"],
    })
    return metrics


def run_multi_seed(
    config: ExperimentConfig,
    seeds: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Run experiments across multiple seeds and aggregate."""
    seeds = seeds or [42, 123, 456]
    all_metrics = []
    for i, seed in enumerate(seeds):
        logger.info("\n" + "=" * 60)
        logger.info("Run %d/%d (seed=%d)", i + 1, len(seeds), seed)
        logger.info("=" * 60)
        all_metrics.append(run_single_experiment(config, seed=seed))

    numeric_keys = ["Accf", "Accft", "Accr", "Accrt", "H-Mean", "MIA", "elapsed_seconds"]
    agg: Dict[str, object] = {}
    for key in numeric_keys:
        vals = [m[key] for m in all_metrics if key in m]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))

    agg.update({"backbone": config.backbone, "forget_classes": config.forget_classes,
                "dataset": config.eval_dataset, "num_seeds": len(seeds)})

    logger.info("\n" + "=" * 60)
    logger.info("Aggregated Results (mean +/- std):")
    for key in numeric_keys:
        if f"{key}_mean" in agg:
            logger.info("  %s: %.4f +/- %.4f", key, agg[f"{key}_mean"], agg[f"{key}_std"])
    return agg


def save_results(metrics: Dict, output_dir: str, filename: str = "results.json") -> str:
    """Save metrics to JSON."""
    ensure_dir(output_dir)
    path = os.path.join(output_dir, filename)
    ser = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            ser[k] = float(v)
        elif isinstance(v, np.ndarray):
            ser[k] = v.tolist()
        else:
            ser[k] = v
    with open(path, "w") as f:
        json.dump(ser, f, indent=2)
    logger.info("Results saved to %s", path)
    return path
