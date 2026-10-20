"""
rl/visualization/audio_utils.py
===============================
Audio I/O, downsampled waveform extraction, and metrics computation for the
RL audio processing visualization tool.
"""

from __future__ import annotations

import base64
import io
import math
import struct
import wave
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from audio_io import load_audio, save_audio


def load_audio_from_path(
    path: str | Path,
    target_sample_rate: int = 44100,
    target_channels: int = 1,
) -> tuple[torch.Tensor, int]:
    """Load WAV file and format into standardized tensor shape (1, 1, T)."""
    waveform, sr = load_audio(
        path,
        target_sample_rate=target_sample_rate,
        target_channels=target_channels,
    )
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2:
        if waveform.shape[0] != 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.unsqueeze(0)  # (1, 1, T)
    return waveform, sr


def save_audio_to_path(
    waveform: torch.Tensor,
    path: str | Path,
    sample_rate: int = 44100,
) -> Path:
    """Save audio tensor to WAV file using standard audio_io utilities."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio_to_save = waveform.detach().cpu()
    while audio_to_save.ndim > 2:
        audio_to_save = audio_to_save.squeeze(0)
    save_audio(audio_to_save, out_path, sample_rate=sample_rate)
    return out_path


def compute_waveform_peaks(
    audio_tensor: torch.Tensor,
    num_points: int = 120,
) -> dict[str, list[float]]:
    """Downsample audio tensor to min/max amplitude envelopes for crisp canvas rendering.

    Returns
    -------
    dict with:
      - 'min_peaks': list of negative peak floats in [-1.0, 0.0]
      - 'max_peaks': list of positive peak floats in [0.0, 1.0]
      - 'rms': list of RMS values per time bucket
    """
    if audio_tensor is None or audio_tensor.numel() == 0:
        return {"min_peaks": [0.0] * num_points, "max_peaks": [0.0] * num_points, "rms": [0.0] * num_points}

    tensor = audio_tensor.detach().cpu().float()
    while tensor.ndim > 1:
        tensor = tensor.mean(dim=0)
    tensor = tensor.flatten()

    total_samples = tensor.shape[0]
    if total_samples < num_points:
        padded = F.pad(tensor, (0, num_points - total_samples))
        vals = padded.tolist()
        return {
            "min_peaks": [min(0.0, v) for v in vals],
            "max_peaks": [max(0.0, v) for v in vals],
            "rms": [abs(v) for v in vals],
        }

    # Use adaptive pooling across time buckets
    x_pos = torch.clamp(tensor, min=0.0).unsqueeze(0).unsqueeze(0)
    x_neg = torch.clamp(-tensor, min=0.0).unsqueeze(0).unsqueeze(0)
    x_sq = (tensor ** 2).unsqueeze(0).unsqueeze(0)

    max_peaks = F.adaptive_max_pool1d(x_pos, num_points).squeeze().tolist()
    min_peaks = (-F.adaptive_max_pool1d(x_neg, num_points).squeeze()).tolist()
    rms_vals = torch.sqrt(F.adaptive_avg_pool1d(x_sq, num_points).squeeze()).tolist()

    if isinstance(max_peaks, float):
        max_peaks = [max_peaks]
        min_peaks = [min_peaks]
        rms_vals = [rms_vals]

    # Clean any potential non-finite values
    max_peaks = [round(max(0.0, min(1.0, float(v))), 4) if math.isfinite(v) else 0.0 for v in max_peaks]
    min_peaks = [round(max(-1.0, min(0.0, float(v))), 4) if math.isfinite(v) else 0.0 for v in min_peaks]
    rms_vals = [round(max(0.0, min(1.0, float(v))), 4) if math.isfinite(v) else 0.0 for v in rms_vals]

    return {
        "min_peaks": min_peaks,
        "max_peaks": max_peaks,
        "rms": rms_vals,
    }


def compute_audio_metrics(
    audio_tensor: torch.Tensor,
    sample_rate: int = 44100,
) -> dict[str, Any]:
    """Compute summary acoustic metrics for an audio clip."""
    if audio_tensor is None or audio_tensor.numel() == 0:
        return {
            "duration_sec": 0.0,
            "sample_rate": sample_rate,
            "total_samples": 0,
            "peak_amplitude": 0.0,
            "peak_db": -99.0,
            "rms_amplitude": 0.0,
            "rms_db": -99.0,
        }

    flat = audio_tensor.detach().cpu().float().flatten()
    total_samples = flat.shape[0]
    duration_sec = total_samples / sample_rate

    peak = float(torch.max(torch.abs(flat)).item()) if total_samples > 0 else 0.0
    rms = float(torch.sqrt(torch.mean(flat ** 2)).item()) if total_samples > 0 else 0.0

    peak_db = 20.0 * math.log10(max(peak, 1e-5))
    rms_db = 20.0 * math.log10(max(rms, 1e-5))

    return {
        "duration_sec": round(duration_sec, 3),
        "sample_rate": sample_rate,
        "total_samples": total_samples,
        "peak_amplitude": round(peak, 4),
        "peak_db": round(peak_db, 1),
        "rms_amplitude": round(rms, 4),
        "rms_db": round(rms_db, 1),
    }


def tensor_to_wav_bytes(
    waveform: torch.Tensor,
    sample_rate: int = 44100,
) -> bytes:
    """Encode audio tensor into standard 16-bit PCM WAV in-memory bytes."""
    tensor = waveform.detach().cpu().float()
    while tensor.ndim > 2:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)  # (1, T)
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise ValueError("Cannot encode empty or non-finite audio")

    channels, total_frames = tensor.shape
    clamped = torch.clamp(tensor, -1.0, 1.0)
    # Same quantisation as audio_io.save_audio (round, not truncate) so downloads match saved files.
    int_samples = (clamped * 32767.0).round().to(torch.int16).t().flatten().tolist()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        raw = struct.pack(f"<{len(int_samples)}h", *int_samples)
        wf.writeframes(raw)

    return buf.getvalue()


def tensor_to_base64_wav(
    waveform: torch.Tensor,
    sample_rate: int = 44100,
) -> str:
    """Return Data URL string (data:audio/wav;base64,...) for immediate browser playback."""
    wav_bytes = tensor_to_wav_bytes(waveform, sample_rate)
    b64 = base64.b64encode(wav_bytes).decode("ascii")
    return f"data:audio/wav;base64,{b64}"
