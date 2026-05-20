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

Deep neural networks memorize information that may later need to be removed — to comply with privacy regulations like GDPR, to excise unsafe content, or to neutralize data poisoning. **Machine unlearning** aims to remove this information without retraining from scratch.

The core challenge is **feature entanglement**: the class you want to forget shares channels and directions with classes you want to keep, so naïve removal damages unrelated capabilities.

**CSE** solves this by identifying *which specific channels* encode the target class and attenuating *only those*, leaving everything else intact:

1. **Extract & standardize** features from target and non-target image sets
2. **Discover the subnet** — solve a generalized eigenvalue problem to find channels where target variance dominates background variance, then greedily select the minimal set covering 85% of discriminative signal
3. **Attenuate** — scale down the selected channels and fold the edit into existing weights

The result is a **training-free**, single-pass edit with **zero inference overhead** that works across CNNs (ResNet-18, EfficientNet-B0) and Transformers (Swin-T).

### Cross-Dataset Evaluation Protocol

Most unlearning methods are evaluated on the *same* dataset used to specify the forget class, which risks overfitting to dataset-specific patterns rather than truly removing the concept. We introduce a **cross-dataset protocol**: unlearn a class using data from one dataset (e.g., CIFAR-10 airplane), then test whether the model still fails to recognize that concept on a *completely different* dataset (e.g., ImageNet airliner). If the model still recognizes airliners in ImageNet images it has never seen, the concept was not truly erased.

### Results at a Glance

Single-class forgetting (Table 1, averaged across backbones):

| Dataset | Accft ↓ | Accrt ↑ | H-Mean ↑ | MIA ↓ |
|---------|---------|---------|----------|-------|
| CIFAR-10 | **0.01** | **0.96** | **0.97** | **0.01** |
| CIFAR-100 | **0.02** | **0.80** | **0.85** | **0.01** |
| ImageNet | **0.02** | **0.62** | **0.74** | **0.01** |

All baselines (ESC, DELETE, BU, SCAR, SCRUB) remain at Accft ≥ 0.10.

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

**Data**: CIFAR-10 and CIFAR-100 are downloaded automatically on first run. ImageNet requires manual setup — set `--imagenet-root /path/to/imagenet` if using ImageNet experiments.

## Quick Start

### Verify Installation

```bash
python test_pipeline.py
```

Runs 5 self-contained tests using synthetic data (no downloads or GPU needed). Expected output:

```
  5 passed, 0 failed / 5 total
  ALL TESTS PASSED -- pipeline is ready
```

### Demo

```bash
python main.py --mode demo
```

Forgets `airplane` on CIFAR-10 with ResNet-18, using `bird` and `ship` as the semantically similar non-target set (D_b). Expected output:

```
Forget-test acc (Accft ↓): ~0.02
Retain-test acc (Accrt ↑): ~0.93
H-Mean (↑):                ~0.95
MIA (↓):                   ~0.01
```

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

### Full Benchmark (All Backbones)

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

All hyperparameters follow Section 4 and Appendix C.4 of the paper. They are **fixed across all backbones and datasets** — no per-experiment tuning.

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `--alpha` | α | 0.01 | Regularization for background covariance |
| `--k-max` | k_max | 50 | Max eigenvectors per layer |
| `--beta` | β | 0.3 | Eigenvector budget as fraction of layer dimension |
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
│   ├── cse.py                 # Core CSE algorithm (Stages 1–3)
│   ├── models.py              # ResNet-18, EfficientNet-B0, Swin-T with feature hooks
│   ├── datasets.py            # Cross-dataset protocol and semantic class mappings
│   ├── evaluate.py            # Accf, Accft, Accr, Accrt, H-Mean, MIA
│   ├── cross_eval.py          # Experiment orchestrator (single-seed and multi-seed)
│   ├── gradcam.py             # Grad-CAM visualization
│   └── utils.py               # Seed management, logging, I/O helpers
├── scripts/
│   ├── run_single_class.sh    # Reproduce Table 1
│   ├── run_multi_class.sh     # Reproduce Table 2
│   └── run_ablation.sh        # Reproduce Tables 5–6
├── requirements.txt
└── README.md
```

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
