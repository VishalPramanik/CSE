import torch

from cse.models import ChannelAffine, MODEL_SPECS, build_model


def test_model_constructors_without_network():
    for name in MODEL_SPECS:
        model = build_model(name, weights="none")
        assert model is not None


def test_channel_affine_nchw_and_nhwc():
    scale = torch.tensor([0.5, 1.0, 0.0])
    bias = torch.tensor([1.0, 2.0, 3.0])

    nchw = torch.zeros(2, 3, 4, 4)
    y = ChannelAffine(scale, bias, 1)(nchw)
    assert y.shape == nchw.shape
    assert torch.allclose(y[:, 0], torch.ones_like(y[:, 0]))

    nhwc = torch.zeros(2, 4, 4, 3)
    y2 = ChannelAffine(scale, bias, -1)(nhwc)
    assert y2.shape == nhwc.shape
    assert torch.allclose(y2[..., 2], torch.full_like(y2[..., 2], 3.0))
