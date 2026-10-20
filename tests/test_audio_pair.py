"""Centralised dry/wet pair contract (shape, finiteness, mono, alignment, crop, tiny inputs)."""
import pytest
import torch

from common.audio_pair import prepare_audio_pair
from models.inverse_fx_net import InverseFXNet
from common.losses import MultiResolutionSTFTLoss


def test_shapes_mono_and_alignment():
    d, w = prepare_audio_pair(torch.randn(3000), torch.randn(2, 5000))      # 1-D and stereo (C,T)
    assert d.shape == w.shape == (1, 1, 3000)                              # aligned to the shorter, mono


def test_crop_to_rl_window_and_system_limit():
    d, w = prepare_audio_pair(torch.randn(1, 44100 * 5), torch.randn(1, 44100 * 5))
    assert d.shape[-1] == 88200                                            # 2 s RL crop
    d, _ = prepare_audio_pair(torch.randn(1, 44100 * 5), torch.randn(1, 44100 * 5), duration=4.0)
    assert d.shape[-1] == 176400
    with pytest.raises(ValueError):
        prepare_audio_pair(torch.randn(1, 100), torch.randn(1, 100), duration=11.0)   # beyond the 10 s system limit


@pytest.mark.parametrize("n", [1, 2, 10, 511, 1024])
def test_very_short_inputs_are_safe_for_every_stft_frontend(n):
    d, w = prepare_audio_pair(torch.randn(1, n), torch.randn(1, n))
    assert d.shape[-1] >= 1025
    assert torch.isfinite(MultiResolutionSTFTLoss()(d, w))
    m = InverseFXNet().eval()
    assert m.predict_chain(d, w)["chain"]


@pytest.mark.parametrize("bad", [torch.tensor([float("nan")] * 2000), torch.tensor([float("inf")] * 2000),
                                 torch.zeros(0), torch.zeros(1, 0), torch.zeros(2, 3, 4, 5), torch.zeros(2, 1, 100),
                                 torch.arange(2000)])
def test_invalid_audio_is_rejected(bad):
    with pytest.raises(ValueError):
        prepare_audio_pair(bad, torch.randn(1, 2000))
    with pytest.raises(ValueError):
        prepare_audio_pair(torch.randn(1, 2000), bad)


def test_wrong_sample_rate_is_rejected_not_silently_used():
    with pytest.raises(ValueError, match="sample rate"):
        prepare_audio_pair(torch.randn(1, 2000), torch.randn(1, 2000), sample_rate=48000)
