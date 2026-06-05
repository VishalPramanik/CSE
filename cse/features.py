"""Stage 0 helper: hook-based feature extraction.

We attach forward hooks to a set of named modules ("layers") of a frozen
encoder and collect channel-wise representations. For spatial features
(convolutional maps or transformer patch tokens) we apply global average
pooling to obtain a single vector per channel, exactly as described in
Sec. 3.1 of the paper.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


def resolve_module(model: nn.Module, name: str) -> nn.Module:
    """Resolve a dotted module path (e.g. ``"layer4.1"``) to a submodule."""
    module = model
    for attr in name.split("."):
        if attr.isdigit():
            module = module[int(attr)]  # type: ignore[index]
        else:
            module = getattr(module, attr)
    return module


def infer_channel_dim(tensor: torch.Tensor) -> int:
    """Heuristic channel axis used when not explicitly configured.

    * 4-D ``(B, C, H, W)`` convolutional maps  -> axis 1
    * 3-D ``(B, N, C)`` transformer token maps -> axis -1
    * 2-D ``(B, C)`` pooled vectors            -> axis -1
    """
    if tensor.dim() == 4:
        return 1
    return tensor.dim() - 1


def channel_pool(tensor: torch.Tensor, channel_dim: int) -> torch.Tensor:
    """Global-average-pool everything except batch and channel dims.

    Returns a ``(B, C)`` tensor of channel-wise activations.
    """
    if tensor.dim() == 2:
        return tensor
    dims = [d for d in range(tensor.dim()) if d not in (0, channel_dim % tensor.dim())]
    pooled = tensor.mean(dim=dims)
    return pooled


class FeatureExtractor:
    """Collects pooled per-channel features from selected encoder layers."""

    def __init__(
        self,
        model: nn.Module,
        layers: Iterable[str],
        channel_dim: Optional[Dict[str, int]] = None,
    ) -> None:
        self.model = model
        self.layers: List[str] = list(layers)
        self.channel_dim = dict(channel_dim or {})
        self._modules = {name: resolve_module(model, name) for name in self.layers}

    @contextmanager
    def _capture(self):
        store: Dict[str, torch.Tensor] = {}
        handles = []

        def make_hook(layer_name: str):
            def hook(_module, _inp, out):
                if isinstance(out, (tuple, list)):
                    out = out[0]
                cdim = self.channel_dim.get(layer_name, infer_channel_dim(out))
                store[layer_name] = channel_pool(out.detach(), cdim).float().cpu()

            return hook

        for name, module in self._modules.items():
            handles.append(module.register_forward_hook(make_hook(name)))
        try:
            yield store
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def extract(
        self,
        loader: DataLoader,
        device: Optional[torch.device] = None,
        max_batches: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the encoder over ``loader`` and return ``{layer: (N, C)}``."""
        device = device or next(self.model.parameters()).device
        was_training = self.model.training
        self.model.eval()

        buffers: Dict[str, List[torch.Tensor]] = {n: [] for n in self.layers}
        with self._capture() as store:
            for i, batch in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                self.model(images.to(device))
                for name in self.layers:
                    buffers[name].append(store[name])

        if was_training:
            self.model.train()
        return {name: torch.cat(chunks, dim=0) for name, chunks in buffers.items()}
