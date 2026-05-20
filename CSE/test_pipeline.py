#!/usr/bin/env python3
"""
End-to-end test for CSE pipeline using synthetic data.

Verifies all modules work correctly without network access.

Usage:
    python test_pipeline.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.models import build_model, FeatureExtractor
from src.cse import ContrastiveSubnetErasure
from src.evaluate import Evaluator
from src.utils import set_seed, setup_logging


def make_data(n_t=200, n_b=400, nc=10, tc=0, sz=224, seed=42):
    """Synthetic data: target class has a channel bias."""
    torch.manual_seed(seed)
    t_img = torch.randn(n_t, 3, sz, sz) * 0.5
    t_img[:, 0] += 0.3                                # target signal
    t_lab = torch.full((n_t,), tc, dtype=torch.long)
    b_img = torch.randn(n_b, 3, sz, sz) * 0.5
    b_lab = torch.randint(1, nc, (n_b,))
    return {
        "target": TensorDataset(t_img, t_lab),
        "nontarget": TensorDataset(b_img, b_lab),
        "forget_train": TensorDataset(t_img[: n_t // 2], t_lab[: n_t // 2]),
        "forget_test":  TensorDataset(t_img[n_t // 2:],  t_lab[n_t // 2:]),
        "retain_train": TensorDataset(b_img[: n_b // 2], b_lab[: n_b // 2]),
        "retain_test":  TensorDataset(b_img[n_b // 2:],  b_lab[n_b // 2:]),
    }


def dl(ds, bs=32):
    return DataLoader(ds, batch_size=bs, shuffle=False)


def test_model_building():
    print("\n[TEST 1] Model Building")
    print("-" * 40)
    for bb in ["resnet18", "efficientnet_b0", "swin_t"]:
        m, layers = build_model(bb, pretrained=False, num_classes=10)
        assert isinstance(m, FeatureExtractor)
        x = torch.randn(2, 3, 224, 224)
        o = m(x)
        assert o.shape == (2, 10), f"{bb}: shape {o.shape}"
        assert len(m.features) == len(layers)
        print(f"  OK  {bb}: {len(layers)} layers, out={o.shape}")
    print("  PASS")
    return True


def test_cse_algorithm():
    print("\n[TEST 2] CSE Algorithm (Stages 1-3)")
    print("-" * 40)
    set_seed(42)
    m, ly = build_model("resnet18", pretrained=False, num_classes=10)
    d = make_data(n_t=100, n_b=200, sz=224)
    cse = ContrastiveSubnetErasure(m, ly)
    info = cse.fit(dl(d["target"]), dl(d["nontarget"]))
    assert len(info) == len(ly)
    for n, si in info.items():
        assert 0 < si.n_selected <= si.n_channels
        assert (si.attenuation_factors >= 0).all() and (si.attenuation_factors <= 1).all()
        print(f"  OK  {n}: {si.n_selected}/{si.n_channels} ch, rho={si.eigenvalues[0]:.4f}")
    cse.apply()
    o = m(torch.randn(4, 3, 224, 224))
    assert o.shape == (4, 10) and not o.isnan().any() and not o.isinf().any()
    print("  PASS")
    return True


def test_evaluation():
    print("\n[TEST 3] Evaluation Metrics")
    print("-" * 40)
    set_seed(42)
    m, _ = build_model("resnet18", pretrained=False, num_classes=10)
    d = make_data(n_t=100, n_b=200, sz=224)
    ev = Evaluator(m, torch.device("cpu"))

    acc = ev.compute_accuracy(dl(d["retain_test"]))
    assert 0 <= acc <= 1
    print(f"  OK  accuracy={acc:.4f}")

    hm = Evaluator.compute_hmean(0.05, 0.93)
    assert abs(hm - 2 * 0.95 * 0.93 / (0.95 + 0.93)) < 1e-6
    print(f"  OK  H-Mean={hm:.4f}")

    mia = ev.compute_mia(dl(d["forget_train"]), dl(d["forget_test"]))
    assert 0 <= mia <= 1
    print(f"  OK  MIA={mia:.4f}")

    met = ev.evaluate(dl(d["forget_train"]), dl(d["forget_test"]),
                      dl(d["retain_train"]), dl(d["retain_test"]))
    for k in ("Accf", "Accft", "Accr", "Accrt", "H-Mean", "MIA"):
        assert k in met
    print(f"  OK  full eval: {met}")
    print("  PASS")
    return True


def test_end_to_end():
    print("\n[TEST 4] End-to-End Pipeline")
    print("-" * 40)
    set_seed(42)
    m, ly = build_model("resnet18", pretrained=False, num_classes=10)
    d = make_data(n_t=100, n_b=200, sz=224)

    ev = Evaluator(m, torch.device("cpu"))
    before = ev.evaluate(dl(d["forget_train"]), dl(d["forget_test"]),
                         dl(d["retain_train"]), dl(d["retain_test"]))
    print(f"  Before CSE: Accft={before['Accft']:.4f} Accrt={before['Accrt']:.4f}")

    cse = ContrastiveSubnetErasure(m, ly)
    cse.fit_and_apply(dl(d["target"]), dl(d["nontarget"]))

    after = ev.evaluate(dl(d["forget_train"]), dl(d["forget_test"]),
                        dl(d["retain_train"]), dl(d["retain_test"]))
    print(f"  After CSE:  Accft={after['Accft']:.4f} Accrt={after['Accrt']:.4f}")
    for k in ("Accf", "Accft", "Accr", "Accrt", "H-Mean", "MIA"):
        assert 0 <= after[k] <= 1
    print("  PASS")
    return True


def test_gradcam():
    print("\n[TEST 5] Grad-CAM Visualization")
    print("-" * 40)
    from src.gradcam import GradCAM
    m, ly = build_model("resnet18", pretrained=False, num_classes=10)
    gc = GradCAM(m, ly[-1])
    hm = gc.generate(torch.randn(1, 3, 224, 224))
    assert hm.shape == (224, 224) and hm.min() >= 0 and hm.max() <= 1
    gc.remove_hooks()
    print(f"  OK  heatmap shape={hm.shape} range=[{hm.min():.3f}, {hm.max():.3f}]")
    print("  PASS")
    return True


def main():
    setup_logging(verbose=True)
    print("=" * 60)
    print("  CSE Pipeline -- End-to-End Test Suite")
    print("=" * 60)

    tests = [
        ("Model Building", test_model_building),
        ("CSE Algorithm", test_cse_algorithm),
        ("Evaluation Metrics", test_evaluation),
        ("End-to-End", test_end_to_end),
        ("Grad-CAM", test_gradcam),
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            if fn():
                passed += 1
        except Exception as e:
            print(f"\n  FAIL  {name}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"  {passed} passed, {failed} failed / {len(tests)} total")
    print("=" * 60)
    if failed:
        print("\n  SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("\n  ALL TESTS PASSED -- pipeline is ready")
        print("  Next: python main.py --mode demo")
        sys.exit(0)


if __name__ == "__main__":
    main()
