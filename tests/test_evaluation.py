import numpy as np

from cse.evaluation import harmonic_mean, loss_threshold_mia


def test_harmonic_mean():
    h = harmonic_mean(0.02, 0.93)
    assert 0.9 < h < 1.0


def test_mia_returns_probability():
    member = np.linspace(0.1, 1.0, 100)
    nonmember = np.linspace(0.8, 2.0, 70)
    score = loss_threshold_mia(member, nonmember, seed=0)
    assert 0.0 <= score <= 1.0
