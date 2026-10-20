"""Shared pair contract: finite mono float32, aligned crop, safe STFT length."""
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
