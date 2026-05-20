# Contrastive Subnet Erasure (CSE)

<p align="center">
  <strong>Selective Amnesia using Contrastive Subnet Erasure for Class-Level Unlearning in Vision Models</strong>
</p>

<p align="center">
  <b>CVPR 2026</b>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#reproducing-results">Reproducing Results</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

**CSE** is a **training-free**, encoder-centric method for class-level unlearning in vision models. Given a pretrained encoder, CSE identifies a compact set of channels most responsible for encoding a target class via contrastive generalized eigenanalysis, then attenuates them in a calibrated manner — all without any gradient updates.

**Key properties:**
- **Training-free**: No gradient computation or fine-tuning required
- **Zero inference overhead**: Attenuation is algebraically folded into existing weights
- **Architecture-agnostic**: Works on CNNs (ResNet, EfficientNet) and Transformers (Swin-T)
- **Cross-dataset generalization**: Forgetting transfers to unseen distributions

### Method at a Glance

| Stage | Operation | Reference |
|-------|-----------|-----------|
| **1. Standardization** | Extract features from target (D_t) and non-target (D_b) sets; compute joint mean and std | Eq. 1–3 |
| **2. Subnet Discovery** | Solve generalized eigenvalue problem Σ_t v = ρ Σ̃_b v; score channels by eigenvalue-weighted participation; greedily select minimal subnet meeting coverage τ_cov | Eq. 4–8 |
| **3. Attenuation** | Compute per-channel attenuation factors; fold scale and bias into model weights | Eq. 9–14 |

## Installation

```bash
git clone https://github.com/VishalPramanik/CSE.git
cd CSE

# Create environment (recommended)
conda create -n cse python=3.10 -y
conda activate cse

# Install dependencies
pip install -r requirements.txt
```

**Requirements**: Python ≥ 3.9, PyTorch ≥ 2.0, CUDA (optional, for GPU acceleration)

## Quick Start

### Verify Installation

```bash
python test_pipeline.py
```

Runs all 5 test modules with synthetic data (no downloads needed).

### Demo (< 2 minutes on GPU)

```bash
python main.py --mode demo
```

Forgets `airplane` on CIFAR-10 with ResNet-18 using `bird` and `ship` as the semantically similar non-target set.

### Single-Class Forgetting

```bash
# Forget 'airplane' on CIFAR-10
python main.py --mode single --backbone resnet18 --dataset cifar10 \
               --forget airplane --nontarget bird ship

# Forget 'shark' on CIFAR-100
python main.py --mode single --backbone efficientnet_b0 --dataset cifar100 \
               --forget shark --nontarget aquarium_fish flatfish ray trout

# Forget 'cat' on CIFAR-10 with Swin-T
python main.py --mode single --backbone swin_t --dataset cifar10 \
               --forget cat --nontarget dog deer
```

### Multi-Class Forgetting

```bash
python main.py --mode multi --backbone resnet18 --dataset cifar100 \
               --forget castle keyboard telephone --num-seeds 3
```

### Full Benchmark

```bash
python main.py --mode benchmark --device cuda
```

### CPU-Only

```bash
python main.py --mode demo --device cpu
```

## Reproducing Results

### Table 1: Single-Class Cross-Dataset Unlearning

```bash
bash scripts/run_single_class.sh resnet18 cuda
bash scripts/run_single_class.sh efficientnet_b0 cuda
bash scripts/run_single_class.sh swin_t cuda
```

### Table 2: Multi-Class Forgetting on CIFAR-100

```bash
bash scripts/run_multi_class.sh resnet18 cuda
```

### Tables 5–6: Ablation Studies

```bash
bash scripts/run_ablation.sh cuda
```

## Hyperparameters

All hyperparameters follow Section 4 and Appendix C.4 of the paper. They are fixed across backbones and datasets.

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `--alpha` | α | 0.01 | Regularization for background covariance |
| `--k-max` | k_max | 50 | Max eigenvectors per layer |
| `--beta` | β | 0.3 | Eigenvector budget as fraction of d_ℓ |
| `--tau-cov` | τ_cov | 0.85 | Coverage threshold for subnet selection |
| `--tau-0` | τ_0 | 0.1 | Minimum score threshold for attenuation |
| `--lambda-0` | λ_0 | 0.5 | Transition smoothness for attenuation |
| `--nontarget-fraction` | — | 0.10 | Fraction of non-target samples for D_b |

## Project Structure

```
CSE/
├── main.py                    # CLI entry point (demo / single / multi / benchmark)
├── test_pipeline.py           # Self-contained test suite (no data needed)
├── configs/
│   ├── __init__.py
│   └── default.py             # CSEConfig and ExperimentConfig dataclasses
├── src/
│   ├── __init__.py
│   ├── cse.py                 # Core algorithm: Stages 1–3
│   ├── models.py              # Backbone wrappers with feature hooks
│   ├── datasets.py            # Dataset loading & cross-dataset protocol (Table 3)
│   ├── evaluate.py            # Metrics: Acc, H-Mean, MIA
│   ├── cross_eval.py          # Experiment orchestrator
│   ├── gradcam.py             # Grad-CAM visualization (Figures 3, 6)
│   └── utils.py               # Seed management, logging, I/O
├── scripts/
│   ├── run_single_class.sh    # Table 1 experiments
│   ├── run_multi_class.sh     # Table 2 experiments
│   └── run_ablation.sh        # Ablation studies (Tables 5–6)
├── requirements.txt
└── README.md
```

### Module Responsibilities

| Module | Role |
|--------|------|
| `src/cse.py` | All three CSE stages. `fit()` runs Stages 1–2; `apply()` runs Stage 3 via forward hooks. |
| `src/models.py` | Wraps torchvision backbones with `FeatureExtractor` that intercepts intermediate layer outputs. Handles spatial pooling for conv and transformer features. |
| `src/datasets.py` | `CrossDatasetProtocol` builds D_t, D_b, and four evaluation splits. Contains all semantic class mappings from Table 3. |
| `src/evaluate.py` | `Evaluator` computes all metrics. MIA uses loss-threshold attack with calibration/test splitting. |
| `src/cross_eval.py` | `run_single_experiment()` orchestrates the full pipeline; `run_multi_seed()` aggregates across seeds. |
| `src/gradcam.py` | `GradCAM` generates attention heatmaps; `visualize_gradcam()` produces the comparison figures. |

## Acknowledgments

This material is based upon work co-supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research under Contract No. DE-AC05-00OR22725.

## Citation

```bibtex
@inproceedings{pramanik2026cse,
  title={Selective Amnesia using Contrastive Subnet Erasure for Class Level Unlearning in Vision Models},
  author={Pramanik, Vishal and Maliha, Maisha and Jha, Susmit and Velasquez, Alvaro and Kotevska, Olivera and Jha, Sumit Kumar},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

