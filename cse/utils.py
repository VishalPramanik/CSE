"""Miscellaneous utilities: reproducible seeding and subset sampling."""

from __future__ import annotations

import random
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


def set_seed(seed: int = 0) -> None:
    """Fix RNG seeds across ``random``, ``numpy`` and ``torch`` for fair runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def indices_for_labels(labels: Sequence[int], wanted: Sequence[int]) -> List[int]:
    """Indices of samples whose label is in ``wanted``."""
    wanted_set = set(int(w) for w in wanted)
    return [i for i, y in enumerate(labels) if int(y) in wanted_set]


def sample_fraction(indices: Sequence[int], fraction: float, seed: int = 0) -> List[int]:
    """Sample a fraction of ``indices`` without replacement (e.g. the 10% D_b)."""
    rng = random.Random(seed)
    idx = list(indices)
    rng.shuffle(idx)
    k = max(1, int(round(fraction * len(idx))))
    return sorted(idx[:k])


def loader_from_subset(
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int = 64,
    num_workers: int = 0,
) -> DataLoader:
    """Build a non-shuffling DataLoader over a subset of ``dataset``."""
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
