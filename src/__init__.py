"""
Contrastive Subnet Erasure (CSE) -- source package.

Modules:
    cse         Core CSE algorithm (Stages 1-3)
    models      Backbone wrappers with feature hooks
    datasets    Dataset loaders and cross-dataset protocol
    evaluate    Metrics: accuracy, H-Mean, MIA
    cross_eval  Cross-dataset evaluation orchestrator
    gradcam     Grad-CAM visualization utilities
    utils       Seed management, logging, I/O helpers
"""

from .cse import ContrastiveSubnetErasure
from .models import build_model
from .datasets import build_datasets, CrossDatasetProtocol
from .evaluate import Evaluator

__all__ = [
    "ContrastiveSubnetErasure",
    "build_model",
    "build_datasets",
    "CrossDatasetProtocol",
    "Evaluator",
]
