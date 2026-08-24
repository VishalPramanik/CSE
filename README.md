# Contrastive Subnet Erasure (CSE)

A clean research implementation of **Contrastive Subnet Erasure**, a training-free, encoder-centric method for class-level visual unlearning. CSE discovers a compact set of target-salient channels with contrastive generalized eigenanalysis and applies calibrated channel attenuation while preserving the rest of the representation.

The repository is organized as a research codebase focused on method components rather than experiment-table scripts. Model interfaces, dataset interfaces, the method, evaluation utilities, configuration, tests, and paper-to-code alignment are separated into focused modules.

## Highlights

- **Paper-aligned three-stage CSE:** joint standardization, contrastive subnet discovery, calibrated attenuation.
- **Three paper backbones:** ResNet-18, EfficientNet-B0, and Swin-T with ImageNet-1K initialization support.
- **Paper datasets:** CIFAR-10, CIFAR-100, ImageNet-1K, and LFW interfaces.
- **Cross-dataset semantic mappings:** the class families reported in the paper are centralized in `cse/mappings.py`.
- **Block-output editing:** CSE is applied after complete encoder stages/blocks to prevent residual bypass.
- **Safe algebraic fold utilities:** exact fold-in helpers are provided for directly adjacent affine operators where the rewrite is mathematically valid.
- **Metrics:** forget/retain accuracy helpers, the paper's H-Mean, and the loss-threshold membership-inference attack.
- **Integrity checks:** a local synthetic smoke test and unit tests run without downloading datasets or pretrained weights.

## Repository layout

```text
.
├── cse/
│   ├── config.py       # centralized CSE hyperparameters
│   ├── method.py       # Stage 1-3 CSE mathematics
│   ├── models.py       # backbones, layer specs, block-output affine edit
│   ├── features.py     # intermediate feature collection + global pooling
│   ├── datasets.py     # CIFAR-10/100, ImageNet-1K, LFW interfaces
│   ├── mappings.py     # paper semantic class mappings
│   ├── evaluation.py   # accuracy, H-Mean, loss-threshold MIA
│   ├── fold.py         # exact-safe affine fold-in primitives
│   └── utils.py
├── configs/default.yaml
├── docs/METHOD_ALIGNMENT.md
├── tests/
├── main.py
├── requirements.txt
└── pyproject.toml
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For editable development installation:

```bash
pip install -e ".[dev]"
```

## Quick integrity check

```bash
python main.py --smoke-test
python -m pytest
```

The smoke test executes the full CSE mathematics on synthetic target/background features and checks that the discovered edit is finite and non-empty. It does not download a dataset or model checkpoint.

## Applying CSE to image datasets

`main.py` fits a CSE edit from a target source set and a semantically related non-target/background set, applies the edit at the model's block outputs, and saves the edited model package.

Example using locally available or torchvision-downloadable CIFAR-10 data:

```bash
python main.py \
  --model resnet18 \
  --weights imagenet1k \
  --target-dataset cifar10 \
  --target-root ./data \
  --target-classes airplane \
  --background-dataset cifar10 \
  --background-root ./data \
  --background-classes bird ship \
  --download \
  --output outputs/cse_resnet18_airplane.pt
```

This command is an interface to the method itself; it is not a script that hard-codes paper tables, seeds, or claimed result values.

### Dataset notes

- **CIFAR-10 / CIFAR-100:** can be downloaded automatically by torchvision.
- **ImageNet-1K:** must be available in an ILSVRC2012-compatible torchvision layout; the dataset cannot be redistributed by this repository.
- **LFW:** torchvision's `LFWPeople` interface is provided, together with a utility for the paper's identity-disjoint 80/20 split.
- The paper's default non-target sampling rate is **10% per semantically related class**.

## Default CSE configuration

```yaml
alpha: 0.01
k_max: 50
eigen_fraction: 0.30
coverage: 0.85
tau0: 0.10
lambda0: 0.50
epsilon: 1.0e-6
non_target_fraction: 0.10
```

These values are centralized in `configs/default.yaml`. The code intentionally does not scatter experiment-specific constants across scripts.

## Method implementation

At each selected encoder block, CSE:

1. extracts and globally pools channel features for the target set and non-target set;
2. jointly standardizes the features;
3. forms target and non-target standardized second-moment matrices;
4. solves the regularized generalized eigenproblem;
5. computes eigenvalue-weighted channel salience;
6. greedily selects the smallest channel subset covering the configured discriminative mass;
7. applies the paper's calibrated attenuation only to the selected compact subnet;
8. transforms the attenuation to the original feature coordinates as a per-channel scale plus mean-compensating bias.

For equation-by-equation correspondence and the few manuscript ambiguities that required an explicit implementation choice, see [`docs/METHOD_ALIGNMENT.md`](docs/METHOD_ALIGNMENT.md).

## Model stages

The default edit points follow the block/stage-level formulation used by the paper:

- **ResNet-18:** `layer1`, `layer2`, `layer3`, `layer4`
- **EfficientNet-B0:** the seven MBConv stages `features.1` through `features.7`
- **Swin-T:** the four transformer stages `features.1`, `features.3`, `features.5`, `features.7`

Classifier heads are not modified by CSE.

## Engineering principles

- No training loop is hidden inside the CSE method.
- Dataset samples are never bundled into the repository.
- No result table is encoded into the implementation.
- Numerical generalized eigenanalysis runs in float64 on CPU for stability; learned scale/bias tensors are stored as float32 and follow the model device/dtype at runtime.
- Method assumptions and manuscript tensions are documented instead of silently patched.
- CI executes both a dependency-light smoke test and unit tests.

## Citation

If this codebase is used in academic work, cite the accompanying paper **“Selective Amnesia using Contrastive Subnet Erasure for Class Level Unlearning in Vision Models.”**
