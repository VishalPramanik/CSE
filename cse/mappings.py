"""Semantic class mappings reported in Table 3 of the paper."""

SEMANTIC_FAMILIES = {
    "airplane": {
        "cifar10": ["airplane"],
        "imagenet": ["airliner", "aircraft"],
    },
    "truck": {
        "cifar10": ["truck"],
        "imagenet": ["garbage truck", "tow truck", "trailer truck"],
    },
    "ship": {
        "cifar10": ["ship"],
        "imagenet": ["container ship"],
    },
    "cat": {
        "cifar10": ["cat"],
        "imagenet": ["tabby cat"],
    },
    "frog": {
        "cifar10": ["frog"],
        "imagenet": ["bullfrog"],
    },
    "shark": {
        "cifar100": ["shark"],
        "imagenet": ["white shark", "tiger shark"],
    },
    "castle": {
        "cifar100": ["castle"],
        "imagenet": ["castle"],
    },
    "keyboard": {
        "cifar100": ["keyboard"],
        "imagenet": ["computer keyboard"],
    },
    "telephone": {
        "cifar100": ["telephone"],
        "imagenet": ["cellular telephone", "dial telephone"],
    },
    "television": {
        "cifar100": ["television"],
        "imagenet": ["television"],
    },
    "lawn_mower": {
        "cifar100": ["lawn mower"],
        "imagenet": ["lawn mower"],
    },
}

# The main paper explicitly uses bird and ship as the semantic non-target set
# for CIFAR-10 airplane forgetting. Other families are intentionally not
# auto-filled because the manuscript does not uniquely specify their Db sets.
DEFAULT_BACKGROUND_CLASSES = {
    ("cifar10", "airplane"): ["bird", "ship"],
}
