"""
Default configuration for Contrastive Subnet Erasure (CSE).

All hyperparameters follow Section 4 and Appendix C.4 of the paper.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CSEConfig:
    """Hyperparameters for the CSE algorithm."""

    # Stage 2: Contrastive Subnet Discovery
    alpha: float = 0.01          # Regularization factor for background covariance (Eq. 6)
    k_max: int = 50              # Maximum number of eigenvectors per layer
    beta: float = 0.3            # Fraction of layer dimension for eigenvector budget
    tau_cov: float = 0.85        # Coverage threshold for subnet selection (Eq. 8)

    # Stage 3: Subnet Attenuation
    tau_0: float = 0.1           # Minimum score threshold for attenuation (Eq. 9)
    lambda_0: float = 0.5        # Transition smoothness for attenuation (Eq. 9)
    epsilon: float = 1e-6        # Numerical stability constant for standardization


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    # --- CSE hyperparameters ---
    cse: CSEConfig = field(default_factory=CSEConfig)

    # --- Model ---
    backbone: str = "resnet18"                # resnet18 | efficientnet_b0 | swin_t
    pretrained: bool = True                   # Use ImageNet pretrained weights
    num_classes: int = 10                     # Number of classes in evaluation dataset

    # --- Datasets ---
    source_dataset: str = "cifar10"           # Dataset for unlearning
    eval_dataset: str = "cifar10"             # Dataset for evaluation
    data_root: str = "./data"                 # Root directory for datasets
    imagenet_root: Optional[str] = None       # Path to ImageNet (if used)
    image_size: int = 224                     # Input image resolution

    # --- Cross-dataset protocol ---
    forget_classes: List[str] = field(default_factory=lambda: ["airplane"])
    nontarget_fraction: float = 0.10          # Fraction of non-target samples for Db
    nontarget_classes: Optional[List[str]] = None  # Semantically similar classes for Db

    # --- Training baselines ---
    baseline_epochs: int = 10
    baseline_lr: float = 1e-5
    baseline_momentum: float = 0.9
    baseline_batch_size: int = 64

    # --- Evaluation ---
    num_seeds: int = 3
    batch_size: int = 128
    num_workers: int = 4

    # --- Output ---
    output_dir: str = "./results"
    save_gradcam: bool = False
    device: str = "cuda"                      # cuda | cpu

    # --- Logging ---
    verbose: bool = True


# ─────────────────────────────────────────────────────────
# Preset configurations for reproducing paper experiments
# ─────────────────────────────────────────────────────────

def cifar10_airplane_config() -> ExperimentConfig:
    """CIFAR-10 -> ImageNet: forget airplane (Section 4, probe i)."""
    return ExperimentConfig(
        source_dataset="cifar10",
        eval_dataset="cifar10",
        forget_classes=["airplane"],
        nontarget_classes=["bird", "ship"],
        num_classes=10,
    )


def cifar100_multiclass_config(n_classes: int = 2) -> ExperimentConfig:
    """Multi-class forgetting on CIFAR-100 (Table 2)."""
    class_pool = ["castle", "keyboard", "telephone", "television", "lawn_mower"]
    return ExperimentConfig(
        source_dataset="cifar100",
        eval_dataset="cifar100",
        forget_classes=class_pool[:n_classes],
        num_classes=100,
    )
