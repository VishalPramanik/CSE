import torch

from cse import CSEConfig, ContrastiveSubnetErasure, apply_pooled_edit


def test_cse_selects_and_preserves_unselected_channels():
    torch.manual_seed(0)
    background = torch.randn(128, 12)
    target = torch.randn(128, 12)
    target[:, :2] *= 5.0

    edit = ContrastiveSubnetErasure(CSEConfig()).fit_layer(target, background)
    assert 0 < edit.selected.numel() <= 12
    mask = torch.ones(12, dtype=torch.bool)
    mask[edit.selected] = False
    assert torch.allclose(edit.attenuation[mask], torch.zeros_like(edit.attenuation[mask]))

    edited = apply_pooled_edit(target, edit)
    assert edited.shape == target.shape
    assert torch.isfinite(edited).all()


def test_mean_compensation_fixed_point():
    torch.manual_seed(1)
    background = torch.randn(64, 8)
    target = torch.randn(64, 8) * torch.tensor([4.0, 3.0, 1, 1, 1, 1, 1, 1])
    edit = ContrastiveSubnetErasure().fit_layer(target, background)
    mean = edit.mean.unsqueeze(0)
    assert torch.allclose(apply_pooled_edit(mean, edit), mean, atol=1e-6)
