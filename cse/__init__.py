"""Contrastive Subnet Erasure (CSE).

Training-free, encoder-centric class-level unlearning for vision models.

Reference
---------
Vishal Pramanik, Maisha Maliha, Susmit Jha, Alvaro Velasquez,
Olivera Kotevska, Sumit Kumar Jha.
"Selective Amnesia using Contrastive Subnet Erasure for Class-Level
Unlearning in Vision Models." CVPR 2026.
"""

from .config import CSEConfig
from .erasure import CSE, LayerEdit
from .features import FeatureExtractor
from .standardize import LayerStats, compute_joint_stats
from .subnet import SubnetResult, contrastive_subnet
from .attenuation import AttenuationParams, build_attenuation

__all__ = [
    "CSE",
    "CSEConfig",
    "LayerEdit",
    "FeatureExtractor",
    "LayerStats",
    "compute_joint_stats",
    "SubnetResult",
    "contrastive_subnet",
    "AttenuationParams",
    "build_attenuation",
]

__version__ = "1.0.0"
