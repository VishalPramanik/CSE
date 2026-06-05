"""Contrastive Subnet Erasure (CSE) - top-level orchestrator.

This module wires the three stages together into a single, training-free
edit (Alg. 1):

    Stage 1  feature extraction + joint standardization   (Sec. 3.2)
    Stage 2  contrastive subnet discovery                 (Sec. 3.3)
    Stage 3  subnet attenuation + runtime fold-in         (Sec. 3.4)

The result is an *edited encoder* that no longer recognizes the target
concept while leaving non-target representations essentially intact, with
negligible inference-time overhead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from .attenuation import AttenuationHook, build_attenuation
from .config import CSEConfig
from .features import FeatureExtractor, resolve_module
from .standardize import compute_joint_stats
from .subnet import contrastive_subnet


@dataclass
class LayerEdit:
    """Bookkeeping for the edit applied at a single layer."""

    layer: str
    num_channels: int
    num_selected: int
    salience: torch.Tensor = field(repr=False)
    selected: torch.Tensor = field(repr=False)
    scale: torch.Tensor = field(repr=False)
    bias: torch.Tensor = field(repr=False)

    @property
    def selected_fraction(self) -> float:
        return self.num_selected / max(self.num_channels, 1)


class CSE:
    """Contrastive Subnet Erasure editor.

    Example
    -------
    >>> editor = CSE(model, layers=["layer4"], config=CSEConfig())
    >>> editor.fit(target_loader, background_loader)   # discover the subnet
    >>> editor.apply()                                 # register runtime hooks
    >>> # model now forgets the target concept
    >>> editor.remove()                                # restore the original
    """

    def __init__(
        self,
        model: nn.Module,
        layers: List[str],
        config: Optional[CSEConfig] = None,
    ) -> None:
        self.model = model
        self.layers = list(layers)
        self.config = config or CSEConfig()
        self.extractor = FeatureExtractor(model, self.layers, self.config.channel_dim)
        self.edits: Dict[str, LayerEdit] = {}
        self._hooks: List[AttenuationHook] = []

    # ------------------------------------------------------------------ #
    # Stage 1-3: discover the contrastive subnet and its attenuation.
    # ------------------------------------------------------------------ #
    def fit(
        self,
        target_loader: DataLoader,
        background_loader: DataLoader,
        device: Optional[torch.device] = None,
    ) -> "CSE":
        """Run Stages 1-3 to compute per-layer attenuation parameters."""
        device = device or next(self.model.parameters()).device
        target_feats = self.extractor.extract(target_loader, device)
        background_feats = self.extractor.extract(background_loader, device)

        self.edits.clear()
        for layer in self.layers:
            t_feats = target_feats[layer]
            b_feats = background_feats[layer]

            # Stage 1: joint standardization.
            stats = compute_joint_stats(t_feats, b_feats, eps=self.config.eps)
            t_std = stats.standardize(t_feats.double())
            b_std = stats.standardize(b_feats.double())

            # Stage 2: contrastive subnet discovery.
            subnet = contrastive_subnet(
                t_std,
                b_std,
                alpha=self.config.alpha,
                k_max=self.config.k_max,
                beta=self.config.beta,
                tau_cov=self.config.tau_cov,
            )

            # Stage 3: attenuation parameters (runtime affine form).
            atten = build_attenuation(
                subnet.salience,
                subnet.selected,
                stats.mu,
                tau0=self.config.tau0,
                lambda0=self.config.lambda0,
            )

            self.edits[layer] = LayerEdit(
                layer=layer,
                num_channels=t_feats.shape[1],
                num_selected=subnet.num_selected,
                salience=subnet.salience,
                selected=subnet.selected,
                scale=atten.scale,
                bias=atten.bias,
            )
        return self

    # ------------------------------------------------------------------ #
    # Apply / remove the runtime edit.
    # ------------------------------------------------------------------ #
    def apply(self) -> "CSE":
        """Register attenuation hooks so the model forgets the target."""
        if not self.edits:
            raise RuntimeError("Call fit() before apply().")
        self.remove()
        for layer, edit in self.edits.items():
            module = resolve_module(self.model, layer)
            cdim = self.config.channel_dim.get(layer)
            from .attenuation import AttenuationParams

            params = AttenuationParams(scale=edit.scale, bias=edit.bias)
            hook = AttenuationHook(params, channel_dim=cdim).register(module)
            self._hooks.append(hook)
        return self

    def remove(self) -> "CSE":
        """Detach all attenuation hooks and restore the original encoder."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        return self

    # Context-manager sugar: `with editor.edited(): ...`
    def edited(self):
        editor = self

        class _Ctx:
            def __enter__(self):
                editor.apply()
                return editor.model

            def __exit__(self, *exc):
                editor.remove()
                return False

        return _Ctx()

    # ------------------------------------------------------------------ #
    # Reporting.
    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        lines = ["CSE edit summary", "-" * 48]
        for layer, edit in self.edits.items():
            lines.append(
                f"{layer:>20s} | channels={edit.num_channels:4d} "
                f"| selected={edit.num_selected:4d} "
                f"({100 * edit.selected_fraction:5.1f}%)"
            )
        return "\n".join(lines)
