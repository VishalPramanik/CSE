"""Unit and integration tests for CSE.

Run with:  pytest -q
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cse import (
    CSE,
    CSEConfig,
    build_attenuation,
    compute_joint_stats,
    contrastive_subnet,
)
from cse.attenuation import AttenuationHook, AttenuationParams
from cse.eval import accuracy, h_mean
from cse.utils import set_seed


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _synthetic(dim=32, n=400, target_ch=range(0, 4), seed=0):
    g = torch.Generator().manual_seed(seed)
    tgt = 0.3 * torch.randn(n, dim, generator=g)
    tgt[:, list(target_ch)] += 4.0 + 2.0 * torch.randn(n, len(list(target_ch)), generator=g)
    bg = 0.3 * torch.randn(n, dim, generator=g)
    return tgt, bg


class _Enc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.proj.weight)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.proj(x)


# --------------------------------------------------------------------------- #
# Stage 1: standardization
# --------------------------------------------------------------------------- #
def test_standardization_zero_mean_unit_std():
    tgt, bg = _synthetic()
    stats = compute_joint_stats(tgt, bg, eps=1e-6)
    joint = torch.cat([tgt, bg]).double()
    std = stats.standardize(joint)
    assert torch.allclose(std.mean(0), torch.zeros(joint.shape[1]).double(), atol=1e-6)
    assert torch.allclose(std.std(0, unbiased=False), torch.ones(joint.shape[1]).double(), atol=1e-3)


# --------------------------------------------------------------------------- #
# Stage 2: subnet discovery
# --------------------------------------------------------------------------- #
def test_subnet_recovers_target_channels():
    target_ch = list(range(0, 4))
    tgt, bg = _synthetic(dim=32, target_ch=target_ch)
    stats = compute_joint_stats(tgt, bg)
    res = contrastive_subnet(
        stats.standardize(tgt.double()),
        stats.standardize(bg.double()),
        alpha=0.01, k_max=50, beta=0.3, tau_cov=0.85,
    )
    selected = sorted(res.selected.nonzero().flatten().tolist())
    assert selected == target_ch
    # Target channels must carry the largest salience.
    assert res.salience[target_ch].min() > res.salience[10:].max()


def test_eigenvalues_descending():
    tgt, bg = _synthetic()
    stats = compute_joint_stats(tgt, bg)
    res = contrastive_subnet(stats.standardize(tgt.double()), stats.standardize(bg.double()))
    ev = res.eigenvalues
    assert torch.all(ev[:-1] >= ev[1:] - 1e-8)


# --------------------------------------------------------------------------- #
# Stage 3: attenuation
# --------------------------------------------------------------------------- #
def test_attenuation_factors_and_preservation():
    salience = torch.tensor([10.0, 0.0, 5.0, 0.0])
    selected = torch.tensor([True, False, True, False])
    mu = torch.tensor([2.0, -1.0, 3.0, 0.5])
    params = build_attenuation(salience, selected, mu, tau0=0.1, lambda0=0.5)
    # Non-selected channels are preserved exactly.
    assert torch.isclose(params.scale[1], torch.tensor(1.0).double())
    assert torch.isclose(params.bias[1], torch.tensor(0.0).double())
    # Selected channels are attenuated: 0 < scale < 1.
    assert 0.0 < params.scale[0] < 1.0
    # beta_c = (s - tau0)/(s + lambda0); bias = beta * mu.
    beta0 = (10.0 - 0.1) / (10.0 + 0.5)
    assert torch.isclose(params.scale[0], torch.tensor(1.0 - beta0).double(), atol=1e-6)
    assert torch.isclose(params.bias[0], torch.tensor(beta0 * 2.0).double(), atol=1e-6)


def test_attenuation_hook_pulls_toward_mean():
    dim = 8
    enc = _Enc(dim)
    scale = torch.ones(dim); scale[0] = 0.0      # fully remove channel 0
    bias = torch.zeros(dim); bias[0] = 5.0       # ... toward mean 5.0
    hook = AttenuationHook(AttenuationParams(scale, bias), channel_dim=-1)
    hook.register(enc.proj)
    x = torch.randn(3, dim)
    out = enc(x)
    assert torch.allclose(out[:, 0], torch.full((3,), 5.0), atol=1e-5)
    hook.remove()


# --------------------------------------------------------------------------- #
# Integration: apply / remove round-trip and end-to-end edit.
# --------------------------------------------------------------------------- #
def test_apply_remove_restores_model():
    set_seed(0)
    dim = 32
    enc = _Enc(dim)
    tgt, bg = _synthetic(dim=dim)
    x = torch.randn(5, dim)
    before = enc(x).clone()

    editor = CSE(enc, layers=["proj"], config=CSEConfig(channel_dim={"proj": -1}))
    editor.fit(
        DataLoader(TensorDataset(tgt, torch.zeros(len(tgt), dtype=torch.long)), batch_size=128),
        DataLoader(TensorDataset(bg, torch.ones(len(bg), dtype=torch.long)), batch_size=128),
    )
    editor.apply()
    edited = enc(x)
    assert not torch.allclose(before, edited)        # the edit changed the output
    editor.remove()
    restored = enc(x)
    assert torch.allclose(before, restored, atol=1e-6)  # removal restores exactly


def test_end_to_end_variance_collapse():
    set_seed(0)
    dim = 64
    target_ch = list(range(0, 8))
    tgt, bg = _synthetic(dim=dim, n=600, target_ch=target_ch)
    enc = _Enc(dim)
    editor = CSE(enc, layers=["proj"], config=CSEConfig(channel_dim={"proj": -1}))
    editor.fit(
        DataLoader(TensorDataset(tgt, torch.zeros(len(tgt), dtype=torch.long)), batch_size=128),
        DataLoader(TensorDataset(bg, torch.ones(len(bg), dtype=torch.long)), batch_size=128),
    )
    var_before = enc(tgt)[:, target_ch].var(0).mean().item()
    editor.apply()
    var_after = enc(tgt)[:, target_ch].var(0).mean().item()
    assert var_after / var_before < 0.05            # >95% variance collapse


def test_context_manager():
    dim = 16
    enc = _Enc(dim)
    tgt, bg = _synthetic(dim=dim)
    editor = CSE(enc, layers=["proj"], config=CSEConfig(channel_dim={"proj": -1}))
    editor.fit(
        DataLoader(TensorDataset(tgt, torch.zeros(len(tgt), dtype=torch.long)), batch_size=128),
        DataLoader(TensorDataset(bg, torch.ones(len(bg), dtype=torch.long)), batch_size=128),
    )
    x = torch.randn(4, dim)
    base = enc(x).clone()
    with editor.edited():
        assert not torch.allclose(enc(x), base)
    assert torch.allclose(enc(x), base, atol=1e-6)  # auto-removed on exit


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_h_mean():
    # Perfect forgetting (acc_ft=0) and perfect retention (acc_rt=1) -> H=1.
    assert abs(h_mean(0.0, 1.0) - 1.0) < 1e-9
    # No forgetting -> H=0.
    assert abs(h_mean(1.0, 1.0) - 0.0) < 1e-9


def test_conv_feature_map_pooling():
    # A 4-D conv output should be pooled over spatial dims to (N, C).
    from cse.features import FeatureExtractor

    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Conv2d(3, 6, 3, padding=1)

        def forward(self, x):
            return self.block(x)

    net = ConvNet().eval()
    loader = DataLoader(TensorDataset(torch.randn(10, 3, 8, 8)), batch_size=5)
    feats = FeatureExtractor(net, ["block"]).extract(loader)
    assert feats["block"].shape == (10, 6)
