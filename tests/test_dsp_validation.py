"""DSP validation: EQ vs analytic biquad, reverb decay/damping physics, alignment, lengths, stability, gradients."""
import math

import pytest
import torch
import yaml

from dapn.effects import EFFECT_REGISTRY
from dapn.effects.eq import EQ
from dapn.effects.reverb import Reverb
from dataset.dapn_dataset import PARAM_SCHEMA

SR = 44100
CFG = yaml.safe_load(open("config/effects_config.yaml"))["effects"]


def phys(eff, name, v):
    lo, hi = CFG[eff]["parameters"][name]["physical_range"]
    return lo + v * (hi - lo)


def biquad_gain_db(f, fc, g, q):
    A, w0 = 10 ** (g / 40), 2 * math.pi * fc / SR
    al = math.sin(w0) / (2 * q)
    b = [1 + al * A, -2 * math.cos(w0), 1 - al * A]; a = [1 + al / A, -2 * math.cos(w0), 1 - al / A]
    w = 2 * math.pi * f / SR; z = complex(math.cos(w), -math.sin(w))
    return 20 * math.log10(abs((b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)))


@pytest.mark.parametrize("fn,gn,qn", [(.05, .9, .3), (.5, .2, .5), (.1, .85, .8), (.03, .9, .1)])
def test_eq_frequency_response_matches_rbj_biquad(fn, gn, qn):
    fc, g, q = phys("EQ", "frequency", fn), phys("EQ", "gain", gn), phys("EQ", "q", qn)
    t = torch.arange(SR) / SR
    for f in [50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000]:
        x = torch.sin(2 * math.pi * f * t)[None]
        y = EQ()(x, {"frequency": torch.tensor(fn), "gain": torch.tensor(gn), "q": torch.tensor(qn)})
        meas = 20 * math.log10(float(y[..., 8192:].square().mean().sqrt() / x[..., 8192:].square().mean().sqrt()))
        assert abs(meas - biquad_gain_db(f, fc, g, q)) < 0.1, (f, meas)


def test_eq_neutral_has_no_delay_or_coloration_at_any_length():
    p = {k: torch.tensor(.5) for k in ("frequency", "gain", "q")}
    for n in (1, 2, 17, 2047, 2048, 2049, 44101):
        x = torch.randn(1, n)
        assert torch.allclose(EQ()(x, p), x, atol=2e-6)
    imp = torch.zeros(1, 8192); imp[0, 0] = 1
    y = EQ()(imp, p)
    assert int(y.abs().argmax()) == 0 and abs(float(y[0, 0]) - 1) < 1e-6 and y[0, 1:].abs().max() < 1e-6


def test_eq_is_causal():
    p = {"frequency": torch.tensor(.1), "gain": torch.tensor(.9), "q": torch.tensor(.5)}
    imp = torch.zeros(1, 4096); imp[0, 1000] = 1
    y = EQ()(imp, p)
    assert y[0, :1000].abs().max() == 0          # nothing before the impulse (no look-ahead/pre-ringing)


def edc_rt60(y):
    e = y.double().square().flip(-1).cumsum(-1).flip(-1)[0]
    db = 10 * torch.log10(e / e[0] + 1e-30)
    i5, i25 = int((db < -5).nonzero()[0]), int((db < -25).nonzero()[0])
    return -60.0 / float((db[i25] - db[i5]) / ((i25 - i5) / SR))


def render_reverb(imp, **kw):
    base = dict(room_size=.6, damping=.2, wet_mix=1., dry_mix=0., decay=.5); base.update(kw)
    return Reverb()(imp, {k: torch.tensor(float(v)) for k, v in base.items()})


def test_reverb_rt60_increases_monotonically_with_decay_and_tail_length_follows():
    imp = torch.zeros(1, SR * 6); imp[0, 0] = 1
    rts = [edc_rt60(render_reverb(imp, decay=d)) for d in (0., .25, .5, .75, 1.)]
    assert all(a < b for a, b in zip(rts, rts[1:])), rts
    assert rts[0] < 0.5 and rts[-1] > 6.0   # min decay: short; max decay: seconds-long tail
    tail = [float(render_reverb(imp, decay=d)[..., 3 * SR:].square().sum()) for d in (0., .5, 1.)]
    assert tail[0] < tail[1] < tail[2]


def test_reverb_energy_envelope_decays_and_is_stable():
    imp = torch.zeros(1, SR * 10); imp[0, 0] = 1
    for decay in (.2, 1.0):
        for room in (0., 1.):
            y = render_reverb(imp, decay=decay, room_size=room)
            assert torch.isfinite(y).all() and y.abs().max() < 1.0
            sec = [float(y[..., i * SR:(i + 1) * SR].square().sum()) for i in range(10)]
            assert sec[-1] < 1e-2 * sec[1] and sec[5] < sec[1]        # decays, never grows (max feedback g<1)


def test_reverb_damping_attenuates_highs_more_than_lows_in_the_tail():
    imp = torch.zeros(1, SR * 3); imp[0, 0] = 1
    def band_ratio(damping):
        tail = render_reverb(imp, damping=damping, decay=.5)[0, SR // 2: SR * 2]
        spec = torch.fft.rfft(tail).abs().square()
        f = torch.fft.rfftfreq(tail.numel(), 1 / SR)
        return float(spec[f > 4000].sum() / spec[f < 1000].sum())
    r = [band_ratio(d) for d in (0., .5, 1.)]
    assert r[0] > r[1] > r[2] and r[2] < 0.2 * r[0], r


def test_reverb_wet_dry_mix_controls_the_direct_signal():
    imp = torch.zeros(1, 4410); imp[0, 0] = 1
    y = Reverb()(imp, {"room_size": torch.tensor(.5), "damping": torch.tensor(.5), "wet_mix": torch.tensor(0.), "dry_mix": torch.tensor(1.), "decay": torch.tensor(.5)})
    assert torch.allclose(y, imp, atol=1e-6)


@pytest.mark.parametrize("name", list(PARAM_SCHEMA))
def test_every_effect_preserves_length_and_alignment_at_neutral(name):
    cls = EFFECT_REGISTRY[name]
    for n in (1, 3, 100, 4411):
        x = torch.randn(1, n) * .1
        y = cls()(x, {k: torch.tensor(.5) for k in PARAM_SCHEMA[name]})
        assert y.shape == x.shape and torch.isfinite(y).all()
    imp = torch.zeros(1, 8820); imp[0, 0] = 1
    y = cls()(imp, {k: torch.tensor(.5) for k in PARAM_SCHEMA[name]})
    assert int(y.abs().argmax()) == 0, f"{name}: unexpected latency (peak at {int(y.abs().argmax())} samples)"


@pytest.mark.parametrize("name", list(PARAM_SCHEMA))
def test_every_effect_is_finite_and_bounded_on_long_audio_at_corners(name):
    torch.manual_seed(0)
    x = torch.randn(1, SR * 10) * .1
    cls = EFFECT_REGISTRY[name]
    for v in (0., 1.):
        y = cls()(x, {k: torch.tensor(v) for k, v in dict.fromkeys(PARAM_SCHEMA[name], v).items()})
        assert y.shape == x.shape and torch.isfinite(y).all() and y.abs().max() < 20


@pytest.mark.parametrize("name", list(PARAM_SCHEMA))
def test_gradients_are_finite_for_all_parameters_inside_the_clip(name):
    torch.manual_seed(1)
    x = torch.randn(1, 22050) * .1
    p = {k: torch.tensor(.3, requires_grad=True) for k in PARAM_SCHEMA[name]}
    if name == "Delay":
        p["delay_time"] = torch.tensor(.02, requires_grad=True)   # taps must fall inside the clip to carry gradient
    EFFECT_REGISTRY[name]()(x, p).square().mean().backward()
    for k, v in p.items():
        assert v.grad is not None and torch.isfinite(v.grad), (name, k)
