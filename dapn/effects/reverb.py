"""Causal Schroeder reverb: eight damped feedback combs, two all-pass filters.

Comb H(z)=D(z)/(1-g L(z)D(z)), L(z)=(1-a)/(1-a z^-1).
Fractional delays use linear interpolation. g=10^(-3 delay_seconds/RT60)
sets the undamped low-frequency decay; damping shortens high-frequency decay.

We recover a causal impulse response using frequency sampling on a circle
inside the unit disk (q=z^-1). Radius exp(-32/N) suppresses periodic wrap by
exp(-32); double precision and N>=4*T keep inverse weighting well conditioned.
This evaluates the feedback transfer function, not a fixed number of echoes.
Linear FFT convolution then returns the first T samples. Output length stays
unchanged: append silence to the input when measuring or exporting full tails.
"""
import math
import torch
from dapn.base_effect import BaseEffect
from common.config_loader import get_sample_rate

_BASE_COMB_DELAYS_MS = [29.7, 37.1, 41.1, 44.1, 48.7, 51.3, 55.7, 60.0]
_BASE_AP_DELAYS_MS = [12.6, 10.0]


class Reverb(BaseEffect):
    effect_name = "Reverb"

    def forward(self, audio, parameters):
        x, was_2d = self.ensure_3d(audio)
        length = x.shape[-1]
        if length == 0:
            raise ValueError("Reverb requires nonempty audio")
        p = {k: self.map_to_physical(k, v).to(dtype=torch.float64, device=x.device)
             for k, v in parameters.items()}
        n = 1 << max(2, (4 * length - 1).bit_length())
        log_radius = -32.0 / n
        omega = 2 * math.pi * torch.arange(n // 2 + 1, device=x.device, dtype=torch.float64) / n
        log_q = torch.complex(torch.full_like(omega, log_radius), -omega)
        q = log_q.exp()

        def delay(samples):
            integer = samples.floor()
            fraction = samples - integer
            return (log_q * integer).exp() * (1 - fraction + fraction * q)

        a = 0.9 * p['damping']
        lowpass = (1 - a) / (1 - a * q)
        response = torch.zeros_like(q)
        sr = get_sample_rate()
        for ms in _BASE_COMB_DELAYS_MS:
            seconds = ms * .001 * (.5 + .5 * p['room_size'])
            d = delay(seconds * sr)
            g = torch.exp(-3 * math.log(10) * seconds / p['decay'])
            response = response + d / (1 - g * lowpass * d)
        response = response / len(_BASE_COMB_DELAYS_MS)
        for ms in _BASE_AP_DELAYS_MS:
            d = delay(p['room_size'].new_tensor(ms * .001 * sr))
            response = response * (d - .5) / (1 - .5 * d)
        impulse = torch.fft.irfft(response, n=n)[:length]
        impulse = impulse * torch.exp(-log_radius * torch.arange(length, device=x.device, dtype=torch.float64))
        conv_n = 1 << (2 * length - 1).bit_length()
        wet = torch.fft.irfft(torch.fft.rfft(x.double(), n=conv_n) *
                              torch.fft.rfft(impulse, n=conv_n), n=conv_n)[..., :length]
        out = (p['dry_mix'] * x + p['wet_mix'] * wet).to(audio.dtype)
        return out.squeeze(0) if was_2d else out
