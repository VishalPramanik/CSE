from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class CSEConfig:
    """Hyperparameters for Contrastive Subnet Erasure (CSE).

    Defaults follow the main paper:
      alpha=0.01, k_max=50, eigen_fraction=0.3, coverage=0.85,
      tau0=0.1, lambda0=0.5, epsilon=1e-6, non_target_fraction=0.10.
    """

    alpha: float = 0.01
    k_max: int = 50
    eigen_fraction: float = 0.30
    coverage: float = 0.85
    tau0: float = 0.10
    lambda0: float = 0.50
    epsilon: float = 1e-6
    non_target_fraction: float = 0.10

    def validate(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0")
        if self.k_max < 1:
            raise ValueError("k_max must be >= 1")
        if not 0 < self.eigen_fraction <= 1:
            raise ValueError("eigen_fraction must be in (0, 1]")
        if not 0 < self.coverage <= 1:
            raise ValueError("coverage must be in (0, 1]")
        if self.tau0 < 0:
            raise ValueError("tau0 must be >= 0")
        if self.lambda0 <= 0:
            raise ValueError("lambda0 must be > 0")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if not 0 < self.non_target_fraction <= 1:
            raise ValueError("non_target_fraction must be in (0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
