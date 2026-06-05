<div align="center">

# Contrastive Subnet Erasure (CSE)

### Selective Amnesia using Contrastive Subnet Erasure for Class-Level Unlearning in Vision Models

[![Conference](https://img.shields.io/badge/CVPR-2026-1b3a5c.svg)](https://openaccess.thecvf.com/content/CVPR2026/html/Pramanik_Selective_Amnesia_using_Contrastive_Subnet_Erasure_for_Class_Level_Unlearning_CVPR_2026_paper.html)
[![Paper](https://img.shields.io/badge/Paper-CVF_Open_Access-b31b1b.svg)](https://openaccess.thecvf.com/content/CVPR2026/html/Pramanik_Selective_Amnesia_using_Contrastive_Subnet_Erasure_for_Class_Level_Unlearning_CVPR_2026_paper.html)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**[Vishal Pramanik](mailto:vishalpramanik@ufl.edu)**<sup>1</sup> &nbsp;·&nbsp;
**Maisha Maliha**<sup>2</sup> &nbsp;·&nbsp;
**Susmit Jha**<sup>3</sup> &nbsp;·&nbsp;
**Alvaro Velasquez**<sup>4</sup> &nbsp;·&nbsp;
**Olivera Kotevska**<sup>5</sup> &nbsp;·&nbsp;
**Sumit Kumar Jha**<sup>1</sup>

<sup>1</sup>University of Florida &nbsp;·&nbsp; <sup>2</sup>University of Oklahoma &nbsp;·&nbsp; <sup>3</sup>SRI &nbsp;·&nbsp;
<sup>4</sup>University of Colorado Boulder &nbsp;·&nbsp; <sup>5</sup>Oak Ridge National Laboratory

**[📄 Paper (CVF Open Access)](https://openaccess.thecvf.com/content/CVPR2026/html/Pramanik_Selective_Amnesia_using_Contrastive_Subnet_Erasure_for_Class_Level_Unlearning_CVPR_2026_paper.html)**

</div>

---

## Overview

**Contrastive Subnet Erasure (CSE)** is a **training-free, encoder-centric** method for
**class-level unlearning** in pretrained vision models. Given a small set of *target* images
(the concept to forget) and a small set of related *non-target* images (concepts to preserve),
CSE identifies a **compact subnet of channels** most responsible for the target concept and
applies a **calibrated, per-channel attenuation** to them. Because the edit is a diagonal
affine map, it can be **algebraically folded into the following layer** — adding **no
inference-time overhead** and leaving task heads unchanged.

CSE directly targets the two recurring failure modes of prior encoder edits:

- **Locality** — it edits only a small, target-salient channel set rather than broad subspaces or full parameter blocks.
- **Geometry preservation** — shared directions that are not *uniquely* target-salient are retained, so non-target structure is left intact.

The method proceeds in three stages:

1. **Standardization** — joint mean/std over `D_t ∪ D_b` for stable covariance estimation *(Sec. 3.2)*.
2. **Contrastive subnet discovery** — a regularized **generalized eigenproblem** `Σ_t v = ρ (Σ_b + δI) v` scores channels by eigenvalue-weighted participation; a greedy rule selects the minimal subnet covering a fraction `τ_cov` of the discriminative mass *(Sec. 3.3)*.
3. **Subnet attenuation** — selected channels are damped by `β_c = clip₀₁((s_c − τ₀)/(s_c + λ₀))`, applied at the block output as `h_att = scale ⊙ h + bias` *(Sec. 3.4)*.

---

## Installation

```bash
git clone https://github.com/VishalPramanik/CSE.git
cd CSE
pip install -e .            # or: pip install -r requirements.txt
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

---

## Quickstart

A self-contained sanity check that needs **no dataset downloads** and runs in seconds on CPU.
It builds a small frozen encoder with a known target concept and verifies that CSE (i) recovers
the exact target channels, (ii) collapses the target concept's variance, and (iii) preserves the
non-target representation.

```bash
python examples/quickstart.py
```

Expected output (abridged):

```
CSE edit summary
                proj | channels=  64 | selected=   8 ( 12.5%)
------------------------------------------------------------------
Metric                                      before       after
------------------------------------------------------------------
Subnet recovered (target = 0..7)                          True
Target-channel variance (target)            6.3384      0.0014
Non-target-channel variance (kept)          2.3265      2.3265
Non-target task accuracy                     1.000       1.000
------------------------------------------------------------------
Target concept variance collapsed by 100.0%  | non-target variance retained 100.0%

Result: PASS - exact subnet recovered, target concept erased, non-target preserved.
```

### Minimal API

```python
from cse import CSE, CSEConfig

editor = CSE(model, layers=["layer4"], config=CSEConfig(channel_dim={"layer4": 1}))
editor.fit(target_loader, background_loader)   # Stages 1–3 (discover the subnet)
editor.apply()                                 # register the runtime edit (forgets target)
# ... evaluate the now-unlearned model ...
editor.remove()                                # restore the original encoder

# or scope the edit with a context manager:
with editor.edited() as unlearned_model:
    evaluate(unlearned_model)
```

---

## Reproducing the paper

The cross-dataset single-class protocol *(Sec. 4)* on real pretrained backbones
(downloads CIFAR-10 and ImageNet weights via `torchvision`; run on a machine with internet/GPU):

```bash
python scripts/reproduce_cifar10.py \
    --backbone resnet18 \
    --forget-class airplane \
    --related bird ship \
    --data ./data
```

This forgets the CIFAR-10 `airplane` class using a non-target set `D_b` of 10% of related
classes (`bird`, `ship`) and reports `Acc_f`, `Acc_ft`, `Acc_r`, `Acc_rt`, `H-Mean`, and `MIA`.
Supported backbones: `resnet18`, `efficientnet_b0`, `swin_t`.

To run CSE on your **own** model and image folders, use the general runner:

```bash
python scripts/run_unlearning.py \
    --model torchvision.models:resnet18 --layer layer4 --channel-dim 1 \
    --target-dir data/forget --background-dir data/retain_related \
    --forget-test data/forget_test --retain-test data/retain_test
```

---

## Results

Single-class cross-dataset unlearning (forget-test `Acc_ft ↓`, retain-test `Acc_rt ↑`,
H-Mean `↑`, MIA `↓`). Selected rows from the paper (Table 1):

| Backbone | Method | `Acc_ft` ↓ | `Acc_rt` ↑ | H-Mean ↑ | MIA ↓ |
|---|---|:--:|:--:|:--:|:--:|
| ResNet-18 | Original | 0.94 | 0.93 | 0.50 | 0.22 |
| ResNet-18 | ESC | 0.10 | 0.92 | 0.90 | 0.05 |
| ResNet-18 | DELETE | 0.12 | 0.91 | 0.89 | 0.06 |
| ResNet-18 | **CSE (ours)** | **0.01** | **0.95** | **0.96** | **0.01** |
| EfficientNet-B0 | **CSE (ours)** | **0.01** | **0.96** | **0.97** | **0.01** |
| Swin-T | **CSE (ours)** | **0.01** | **0.97** | **0.98** | **0.01** |

Across CIFAR-10 / CIFAR-100 / ImageNet and three backbones, CSE drives `Acc_ft` to 0.01–0.02
while preserving the highest `Acc_rt` and attaining the strongest H-Mean (0.96–0.98 on CIFAR-10,
0.84–0.86 on CIFAR-100, 0.73–0.75 on ImageNet) and the lowest MIA (0.01). See the paper for the
full tables, multi-class results, ablations, and the LFW identity-forgetting case study.

---

## Repository structure

```
CSE/
├── cse/                      # Core library
│   ├── config.py             # CSEConfig (paper hyperparameters)
│   ├── features.py           # Stage 0: hook-based feature extraction + GAP
│   ├── standardize.py        # Stage 1: joint standardization        (Sec. 3.2)
│   ├── subnet.py             # Stage 2: contrastive subnet discovery (Sec. 3.3)
│   ├── attenuation.py        # Stage 3: per-channel attenuation       (Sec. 3.4)
│   ├── erasure.py            # CSE orchestrator (fit / apply / remove)
│   ├── utils.py              # seeding, subset sampling
│   └── eval/                 # metrics (Acc, H-Mean) and the MIA attack
├── examples/
│   └── quickstart.py         # self-contained CPU sanity check
├── scripts/
│   ├── reproduce_cifar10.py  # cross-dataset reproduction on real backbones
│   └── run_unlearning.py     # general CLI for arbitrary models/folders
├── tests/
│   └── test_cse.py           # pytest unit + integration tests
├── configs/default.yaml      # paper hyperparameters
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

---

## Tests

```bash
pytest -q
```

The suite validates each stage (standardization statistics, exact subnet recovery via the
generalized eigenproblem, attenuation scale/bias correctness, apply/remove round-trip, and an
end-to-end variance-collapse integration test).

---

## Citation

```bibtex
@InProceedings{Pramanik_2026_CVPR,
    author    = {Pramanik, Vishal and Maliha, Maisha and Jha, Susmit and Velasquez, Alvaro and Kotevska, Olivera and Jha, Sumit Kumar},
    title     = {Selective Amnesia using Contrastive Subnet Erasure for Class Level Unlearning in Vision Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {31662-31671}
}
```

**Paper:** <https://openaccess.thecvf.com/content/CVPR2026/html/Pramanik_Selective_Amnesia_using_Contrastive_Subnet_Erasure_for_Class_Level_Unlearning_CVPR_2026_paper.html>

---

## License

Released under the [MIT License](LICENSE).
