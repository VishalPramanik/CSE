"""Evaluation utilities for CSE: accuracy/H-Mean metrics and MIA."""

from .metrics import (
    UnlearningReport,
    accuracy,
    forget_success,
    h_mean,
)
from .mia import membership_inference

__all__ = [
    "UnlearningReport",
    "accuracy",
    "forget_success",
    "h_mean",
    "membership_inference",
]
