import math
import torch
import pytest
from dapn.effects.eq import EQ
from dapn.effects.distortion import Distortion
from dapn.effects.reverb import Reverb


def test_eq_neutral_impulse_and_waveform():
    x = torch.randn(1, 4096)
    p = {k: torch.tensor(.5) for k in ['frequency', 'gain', 'q']}
    assert torch.allclose(EQ()(x, p), x, atol=2e-6)
    x.zero_(); x[0, 0] = 1
    assert EQ()(x, p).abs().argmax() == 0


@pytest.mark.parametrize('length', [1, 32, 4096])
def test_eq_boundaries(length):
    x = torch.randn(1, length)
    for v in [0., 1.]:
        p = {k: torch.tensor(v, requires_grad=True) for k in ['frequency', 'gain', 'q']}
        y = EQ()(x, p)
        assert y.shape == x.shape and torch.isfinite(y).all()
        y.square().mean().backward()
        assert all(t.grad is not None and torch.isfinite(t.grad) for t in p.values())


def test_eq_center_frequency_gain():
    sr = 44100; freq = 1000.
    x = torch.sin(2 * math.pi * freq * torch.arange(sr) / sr)[None]
    p = dict(frequency=torch.tensor((freq-20)/19980), gain=torch.tensor(.75), q=torch.tensor(.2))
    y = EQ()(x, p)
    measured = 20 * torch.log10(y[...,4096:].square().mean().sqrt()/x[...,4096:].square().mean().sqrt())
    assert abs(measured.item()-6) < .1


def test_distortion_bright_impulse_is_aligned():
    x = torch.zeros(1, 256); x[0,0] = .1
    p = {k: torch.tensor(v) for k,v in dict(drive=0.,tone=1.,mix=1.,output_gain=2/3).items()}
    assert Distortion()(x,p).abs().argmax() == 0


def test_reverb_tail_decay_and_damping():
    x = torch.zeros(1, 44100); x[0,0] = 1
    def render(decay,damping):
        return Reverb()(x,{k:torch.tensor(v) for k,v in dict(room_size=.6,damping=damping,wet_mix=1.,dry_mix=0.,decay=decay).items()})
    short=render(0.,0.); long=render(.5,0.); damp=render(.5,1.)
    assert long[...,22050:].square().sum() > 100 * short[...,22050:].square().sum()
    assert long[...,22050:].abs().max() > 1e-4
    assert long[...,22050:].square().sum() < long[...,:22050].square().sum()
    # First differences measure high-frequency energy suppression in the late tail.
    assert torch.diff(damp[...,10000:]).square().sum() < torch.diff(long[...,10000:]).square().sum()
    assert torch.isfinite(long).all() and long.abs().max() < 1
