from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models import LayerSpec, ModelSpec


def pool_channels(x: torch.Tensor, channel_dim: int) -> torch.Tensor:
    """Global-average-pool all non-batch, non-channel dimensions."""
    if x.ndim < 2:
        raise ValueError(f"expected tensor with batch and channel dimensions, got {tuple(x.shape)}")
    dim = channel_dim if channel_dim >= 0 else x.ndim + channel_dim
    if dim == 0:
        raise ValueError("channel dimension cannot be the batch dimension")
    reduce_dims = tuple(i for i in range(1, x.ndim) if i != dim)
    return x.mean(dim=reduce_dims) if reduce_dims else x


@torch.inference_mode()
def collect_features(
    model: nn.Module,
    dataloader: DataLoader,
    spec: ModelSpec,
    device: torch.device | str,
    max_samples: int | None = None,
    show_progress: bool = True,
) -> Dict[str, torch.Tensor]:
    """Collect pooled block-output features h^(l)(x) for CSE fitting."""
    device = torch.device(device)
    model = model.to(device).eval()
    chunks: Dict[str, list[torch.Tensor]] = defaultdict(list)
    current: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(layer: LayerSpec):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                raise TypeError(f"layer {layer.name} produced a non-tensor output")
            current[layer.name] = pool_channels(output.detach(), layer.channel_dim).cpu()
        return hook

    for layer in spec.layers:
        handles.append(model.get_submodule(layer.name).register_forward_hook(make_hook(layer)))

    seen = 0
    try:
        iterator = tqdm(dataloader, desc="Collecting CSE features", disable=not show_progress)
        for batch in iterator:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device, non_blocking=True)
            current.clear()
            _ = model(images)

            if set(current) != {layer.name for layer in spec.layers}:
                missing = {layer.name for layer in spec.layers}.difference(current)
                raise RuntimeError(f"feature hooks did not fire for: {sorted(missing)}")

            take = images.shape[0]
            if max_samples is not None:
                take = min(take, max_samples - seen)
            for layer in spec.layers:
                chunks[layer.name].append(current[layer.name][:take])
            seen += take
            if max_samples is not None and seen >= max_samples:
                break
    finally:
        for handle in handles:
            handle.remove()

    if seen == 0:
        raise ValueError("dataloader produced no samples")
    return {name: torch.cat(parts, dim=0) for name, parts in chunks.items()}
