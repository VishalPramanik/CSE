"""
Dataset loading and cross-dataset evaluation protocol.

Implements the semantic class mappings from Table 3 and the
cross-dataset unlearning protocol described in Section 4.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision.transforms as T
import torchvision.datasets as datasets

logger = logging.getLogger("CSE")


# ─────────────────────────────────────────────────────────
# CIFAR-10 / CIFAR-100 class name <-> index mappings
# ─────────────────────────────────────────────────────────

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee",
    "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus",
    "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch",
    "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant",
    "flatfish", "forest", "fox", "girl", "hamster", "house",
    "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter",
    "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate",
    "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray",
    "road", "rocket", "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider", "squirrel",
    "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf",
    "woman", "worm",
]


# ─────────────────────────────────────────────────────────
# Cross-dataset semantic class mappings (Table 3)
# ─────────────────────────────────────────────────────────

CROSS_DATASET_MAPPINGS = {
    "airplane": {
        "cifar10": ["airplane"],
        "cifar100": [],
        "imagenet_ids": [404],
        "similar_nontarget": {"cifar10": ["bird", "ship"]},
    },
    "truck": {
        "cifar10": ["truck"],
        "cifar100": ["pickup_truck"],
        "imagenet_ids": [569, 864, 867],
        "similar_nontarget": {"cifar10": ["automobile"]},
    },
    "ship": {
        "cifar10": ["ship"],
        "cifar100": [],
        "imagenet_ids": [510],
        "similar_nontarget": {"cifar10": ["airplane", "automobile"]},
    },
    "cat": {
        "cifar10": ["cat"],
        "cifar100": [],
        "imagenet_ids": [281],
        "similar_nontarget": {"cifar10": ["dog", "deer"]},
    },
    "frog": {
        "cifar10": ["frog"],
        "cifar100": [],
        "imagenet_ids": [30],
        "similar_nontarget": {"cifar10": ["bird", "deer"]},
    },
    "shark": {
        "cifar10": [],
        "cifar100": ["shark"],
        "imagenet_ids": [2, 3],
        "similar_nontarget": {"cifar100": ["aquarium_fish", "flatfish", "ray", "trout"]},
    },
    "castle": {
        "cifar10": [],
        "cifar100": ["castle"],
        "imagenet_ids": [483],
        "similar_nontarget": {"cifar100": ["bridge", "house", "skyscraper"]},
    },
    "keyboard": {
        "cifar10": [],
        "cifar100": ["keyboard"],
        "imagenet_ids": [508],
        "similar_nontarget": {"cifar100": ["telephone", "television"]},
    },
    "telephone": {
        "cifar10": [],
        "cifar100": ["telephone"],
        "imagenet_ids": [487, 528],
        "similar_nontarget": {"cifar100": ["keyboard", "television"]},
    },
    "television": {
        "cifar10": [],
        "cifar100": ["television"],
        "imagenet_ids": [851],
        "similar_nontarget": {"cifar100": ["keyboard", "telephone"]},
    },
    "lawn_mower": {
        "cifar10": [],
        "cifar100": ["lawn_mower"],
        "imagenet_ids": [621],
        "similar_nontarget": {"cifar100": ["tractor", "bicycle"]},
    },
}


def _get_class_index(class_name: str, dataset_name: str) -> int:
    """Resolve a class name to its integer label."""
    if dataset_name == "cifar10":
        return CIFAR10_CLASSES.index(class_name)
    elif dataset_name == "cifar100":
        return CIFAR100_CLASSES.index(class_name)
    raise ValueError(f"Unknown dataset: {dataset_name}")


def _get_class_indices(class_names: List[str], dataset_name: str) -> List[int]:
    return [_get_class_index(n, dataset_name) for n in class_names]


# ─────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────

def get_train_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.RandomCrop(image_size, padding=image_size // 8),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────
# Dataset loaders (with download fallback)
# ─────────────────────────────────────────────────────────

def _load_dataset(dataset_cls, root: str, train: bool, transform):
    """Try to load a torchvision dataset; return None on failure."""
    for download in (True, False):
        try:
            return dataset_cls(root=root, train=train, download=download, transform=transform)
        except Exception:
            continue
    return None


def _get_subset_by_classes(
    dataset,
    class_indices: List[int],
    invert: bool = False,
    fraction: float = 1.0,
    seed: int = 42,
) -> Subset:
    """
    Extract samples belonging to (or NOT belonging to) specific classes.
    """
    targets = np.array(dataset.targets)
    mask = np.isin(targets, class_indices)
    if invert:
        mask = ~mask
    indices = np.where(mask)[0]

    if fraction < 1.0:
        rng = np.random.RandomState(seed)
        n_select = max(1, int(len(indices) * fraction))
        indices = rng.choice(indices, size=n_select, replace=False)

    return Subset(dataset, indices.tolist())


class CrossDatasetProtocol:
    """
    Implements the cross-dataset evaluation protocol from Section 4.

    Prepares:
        - Dt: Target set (samples of the class to forget)
        - Db: Non-target set (semantically similar classes, subsampled)
        - Evaluation splits: forget-train, forget-test, retain-train, retain-test
    """

    def __init__(
        self,
        forget_classes: List[str],
        dataset_name: str = "cifar10",
        data_root: str = "./data",
        nontarget_classes: Optional[List[str]] = None,
        nontarget_fraction: float = 0.10,
        image_size: int = 224,
        batch_size: int = 128,
        num_workers: int = 4,
        seed: int = 42,
    ):
        self.forget_classes = forget_classes
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.nontarget_fraction = nontarget_fraction
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.forget_indices = _get_class_indices(forget_classes, dataset_name)

        if nontarget_classes is not None:
            self.nontarget_indices = _get_class_indices(nontarget_classes, dataset_name)
        else:
            self.nontarget_indices = self._auto_nontarget_indices()

        self._build()

    def _auto_nontarget_indices(self) -> List[int]:
        """Auto-select non-target classes based on cross-dataset mappings."""
        for mapping in CROSS_DATASET_MAPPINGS.values():
            ds_classes = mapping.get(self.dataset_name, [])
            if any(c in ds_classes for c in self.forget_classes):
                similar = mapping.get("similar_nontarget", {}).get(self.dataset_name, [])
                if similar:
                    return _get_class_indices(similar, self.dataset_name)
        # Fallback: all non-forget classes
        n = 10 if self.dataset_name == "cifar10" else 100
        return [i for i in range(n) if i not in self.forget_indices]

    def _build(self):
        transform = get_eval_transform(self.image_size)

        if self.dataset_name == "cifar10":
            ds_cls = datasets.CIFAR10
        else:
            ds_cls = datasets.CIFAR100

        full_train = _load_dataset(ds_cls, self.data_root, train=True, transform=transform)
        full_test = _load_dataset(ds_cls, self.data_root, train=False, transform=transform)

        if full_train is None or full_test is None:
            raise RuntimeError(
                f"Could not load {self.dataset_name}. Download it manually to '{self.data_root}' "
                f"or ensure network access."
            )

        self.target_set = _get_subset_by_classes(full_train, self.forget_indices)
        self.nontarget_set = _get_subset_by_classes(
            full_train, self.nontarget_indices,
            fraction=self.nontarget_fraction, seed=self.seed,
        )
        self.forget_train = _get_subset_by_classes(full_train, self.forget_indices)
        self.forget_test = _get_subset_by_classes(full_test, self.forget_indices)
        self.retain_train = _get_subset_by_classes(full_train, self.forget_indices, invert=True)
        self.retain_test = _get_subset_by_classes(full_test, self.forget_indices, invert=True)

    def _loader(self, dataset) -> DataLoader:
        return DataLoader(dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def get_target_loader(self) -> DataLoader:
        return self._loader(self.target_set)

    def get_nontarget_loader(self) -> DataLoader:
        return self._loader(self.nontarget_set)

    def get_forget_train_loader(self) -> DataLoader:
        return self._loader(self.forget_train)

    def get_forget_test_loader(self) -> DataLoader:
        return self._loader(self.forget_test)

    def get_retain_train_loader(self) -> DataLoader:
        return self._loader(self.retain_train)

    def get_retain_test_loader(self) -> DataLoader:
        return self._loader(self.retain_test)

    def summary(self) -> Dict[str, int]:
        return {
            "target_set (Dt)": len(self.target_set),
            "nontarget_set (Db)": len(self.nontarget_set),
            "forget_train": len(self.forget_train),
            "forget_test": len(self.forget_test),
            "retain_train": len(self.retain_train),
            "retain_test": len(self.retain_test),
        }


def build_datasets(
    forget_classes: List[str],
    dataset_name: str = "cifar10",
    data_root: str = "./data",
    nontarget_classes: Optional[List[str]] = None,
    nontarget_fraction: float = 0.10,
    image_size: int = 224,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 42,
) -> CrossDatasetProtocol:
    """Convenience wrapper to build the cross-dataset protocol."""
    return CrossDatasetProtocol(
        forget_classes=forget_classes,
        dataset_name=dataset_name,
        data_root=data_root,
        nontarget_classes=nontarget_classes,
        nontarget_fraction=nontarget_fraction,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
