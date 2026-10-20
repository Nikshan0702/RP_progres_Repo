"""Shared pair contract: finite mono float32, aligned crop, safe STFT length.

Two distinct duration limits (both explicit):
  * system limit  : ``audio.max_duration_seconds`` (10 s) in config/system_config.yaml - the longest clip DAPN/IO accept
                    (audio_io.load_audio truncates longer files with a logged warning).
  * RL/predictor crop: ``duration`` (default 2.0 s, must be <= the system limit) - ONLY the first ``duration`` seconds of
                    an aligned dry/wet pair are used for training, evaluation, rewards and the dashboard. Longer input is
                    cropped, not resampled; shorter input is zero-padded only up to the minimum STFT length (1025 samples).
All of 44.1 kHz / mono are enforced by audio_io (resampling + downmix at load time) and validated here.
"""
import torch
import torch.nn.functional as F
from common.config_loader import get_sample_rate, get_max_duration_seconds


def prepare_audio_pair(dry, wet, *, device='cpu', sample_rate=None, duration=2.0):
    sr = get_sample_rate()
    if sample_rate is not None and sample_rate != sr:
        raise ValueError(f'Tensor sample rate {sample_rate} differs from configured {sr}; resample at I/O')
    if not 0 < duration <= get_max_duration_seconds():
        raise ValueError('Invalid audio crop duration')
    def mono(x):
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise ValueError('Audio must be a floating-point tensor')
        x=x.to(device=device,dtype=torch.float32)
        if x.ndim==1:x=x[None,None]
        elif x.ndim==2:x=x[None]
        if x.ndim!=3 or x.shape[0]!=1 or x.shape[1]<1 or x.shape[-1]<1 or not torch.isfinite(x).all():
            raise ValueError('Expected one nonempty finite audio clip (C,T) or (1,C,T)')
        return x.mean(dim=1,keepdim=True)
    dry,wet=mono(dry),mono(wet)
    length=min(dry.shape[-1],wet.shape[-1],int(sr*duration))
    # Shared right zero-padding makes all feature frontends safe on tiny clips.
    return tuple(F.pad(x[...,:length],(0,max(0,1025-length))) for x in (dry,wet))
