from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(train: bool) -> transforms.Compose:
    """Paper-consistent 224x224 preprocessing.

    The manuscript specifies random crops + horizontal flips for training and
    center crops for evaluation, but not the pre-crop resize rule. This uses the
    standard 256 -> 224 convention while preserving those stated operations.
    """
    ops = [transforms.Resize(256)]
    if train:
        ops += [transforms.RandomCrop(224), transforms.RandomHorizontalFlip()]
    else:
        ops += [transforms.CenterCrop(224)]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


def build_dataset(
    name: str,
    root: str | Path,
    split: str = "train",
    train_transform: bool | None = None,
    download: bool = False,
) -> Dataset:
    """Create a dataset interface for CIFAR-10, CIFAR-100, ImageNet-1K or LFW."""
    name = name.lower()
    root = str(root)
    is_train = split.lower() in {"train", "training"} if train_transform is None else train_transform
    transform = build_transform(is_train)

    if name == "cifar10":
        return datasets.CIFAR10(root=root, train=split.lower() == "train", transform=transform, download=download)
    if name == "cifar100":
        return datasets.CIFAR100(root=root, train=split.lower() == "train", transform=transform, download=download)
    if name in {"imagenet", "imagenet1k", "imagenet-1k"}:
        mapped_split = "val" if split.lower() in {"test", "val", "validation"} else "train"
        # torchvision.datasets.ImageNet expects an ILSVRC2012-compatible root.
        return datasets.ImageNet(root=root, split=mapped_split, transform=transform)
    if name == "lfw":
        lfw_split = "train" if split.lower() == "train" else "test"
        return datasets.LFWPeople(
            root=root,
            split=lfw_split,
            image_set="deepfunneled",
            transform=transform,
            download=download,
        )
    raise ValueError("dataset must be one of: cifar10, cifar100, imagenet, lfw")


def _normalize_label(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").replace("-", " ").split())


def class_names(dataset: Dataset) -> list[list[str]]:
    """Return each class as a list of normalized aliases."""
    raw = getattr(dataset, "classes", None)
    if raw is None:
        class_to_idx = getattr(dataset, "class_to_idx", None)
        if class_to_idx is None:
            raise ValueError("dataset does not expose classes or class_to_idx")
        raw = [None] * len(class_to_idx)
        for name, idx in class_to_idx.items():
            raw[int(idx)] = name
    aliases: list[list[str]] = []
    for item in raw:
        if isinstance(item, (tuple, list)):
            aliases.append([_normalize_label(str(x)) for x in item])
        else:
            aliases.append([_normalize_label(str(item))])
    return aliases


def resolve_class_indices(dataset: Dataset, requested: Sequence[str]) -> list[int]:
    """Resolve human-readable class labels against torchvision dataset classes."""
    aliases = class_names(dataset)
    resolved: list[int] = []
    for query in requested:
        q = _normalize_label(query)
        exact = [i for i, names in enumerate(aliases) if q in names]
        if not exact:
            # Useful for ImageNet synonyms such as 'garbage truck, dustcart'.
            exact = [i for i, names in enumerate(aliases) if any(q in name or name in q for name in names)]
        if len(exact) == 0:
            raise ValueError(f"class '{query}' was not found in dataset classes")
        if len(exact) > 1:
            raise ValueError(f"class '{query}' is ambiguous; matched indices {exact}")
        resolved.append(exact[0])
    return sorted(set(resolved))


def subset_by_classes(dataset: Dataset, requested: Sequence[str]) -> Subset:
    indices = set(resolve_class_indices(dataset, requested))
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("dataset does not expose targets; provide a compatible torchvision-style dataset")
    selected = [i for i, y in enumerate(targets) if int(y) in indices]
    if not selected:
        raise ValueError(f"no samples found for requested classes: {requested}")
    return Subset(dataset, selected)


def sample_background_by_classes(
    dataset: Dataset,
    requested: Sequence[str],
    fraction: float = 0.10,
    seed: int = 0,
) -> Subset:
    """Sample the paper's default fraction independently within each Db class."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    class_ids = resolve_class_indices(dataset, requested)
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("dataset does not expose targets")

    rng = random.Random(seed)
    selected: list[int] = []
    for class_id in class_ids:
        candidates = [i for i, y in enumerate(targets) if int(y) == class_id]
        if not candidates:
            raise ValueError(f"class index {class_id} has no samples")
        n = max(1, int(round(fraction * len(candidates))))
        selected.extend(rng.sample(candidates, min(n, len(candidates))))
    return Subset(dataset, sorted(selected))


def split_lfw_identities(dataset: Dataset, train_fraction: float = 0.80, seed: int = 0):
    """Identity-disjoint 80/20 split used by the paper's LFW case study."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("LFW dataset does not expose targets")
    identities = sorted(set(int(y) for y in targets))
    rng = random.Random(seed)
    rng.shuffle(identities)
    cut = int(round(train_fraction * len(identities)))
    train_ids = set(identities[:cut])
    train_idx = [i for i, y in enumerate(targets) if int(y) in train_ids]
    test_idx = [i for i, y in enumerate(targets) if int(y) not in train_ids]
    return Subset(dataset, train_idx), Subset(dataset, test_idx)
