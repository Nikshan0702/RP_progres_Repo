"""
audio_io.py
===========
Audio file I/O utilities — intentionally kept *outside* the
differentiable DSP chain.

Uses a small RIFF parser (integer PCM 8/16/24/32-bit, float 32/64-bit,
WAVE_FORMAT_EXTENSIBLE) and the stdlib `wave` module for writing. This avoids heavy external codec dependencies like
torchcodec / soundfile, while guaranteeing standard 16-bit or 32-bit float
PCM WAV compatibility across all platforms.

Functions
---------
load_audio  — WAV → torch.Tensor (C, T)
save_audio  — torch.Tensor → WAV
generate_test_tone — create a synthetic test tone without a file
"""

from __future__ import annotations

import logging
import math
import struct
import wave
from pathlib import Path

import torch

from common.config_loader import get_sample_rate, get_channels, get_max_duration_seconds

logger = logging.getLogger(__name__)


_WAVE_FORMAT_PCM, _WAVE_FORMAT_FLOAT, _WAVE_FORMAT_EXTENSIBLE = 1, 3, 0xFFFE


def _decode_wav(path: Path) -> tuple[torch.Tensor, int]:
    """Decode a RIFF/WAVE file using its *declared* encoding.

    Supported: integer PCM 8/16/24/32-bit and IEEE float 32/64-bit
    (including WAVE_FORMAT_EXTENSIBLE). Everything else raises ``ValueError``.
    Returns ``(C, T)`` float32 and the file sample rate; checks finiteness.
    """
    import numpy as np

    data = path.read_bytes()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path.name}: not a RIFF/WAVE file")
    fmt = None
    pcm = None
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            pcm = body  # truncated final chunk tolerated (streamed writers)
            break
        pos += 8 + size + (size & 1)
    if fmt is None or len(fmt) < 16 or pcm is None:
        raise ValueError(f"{path.name}: missing fmt/data chunk")
    tag, channels, sr, _, block_align, bits = struct.unpack("<HHIIHH", fmt[:16])
    if tag == _WAVE_FORMAT_EXTENSIBLE:
        if len(fmt) < 26:
            raise ValueError(f"{path.name}: malformed WAVE_FORMAT_EXTENSIBLE header")
        tag = struct.unpack("<H", fmt[24:26])[0]  # sub-format GUID starts with the real tag
    if channels < 1 or sr < 1:
        raise ValueError(f"{path.name}: invalid channel count/sample rate ({channels}, {sr})")
    width = bits // 8
    if bits % 8 or width < 1 or block_align != width * channels:
        raise ValueError(f"{path.name}: inconsistent block alignment")
    frames = len(pcm) // block_align
    if frames == 0:
        raise ValueError(f"{path.name}: contains no audio samples")
    pcm = pcm[:frames * block_align]
    if tag == _WAVE_FORMAT_PCM:
        if width == 1:
            x = (np.frombuffer(pcm, np.uint8).astype(np.float32) - 128.0) / 128.0
        elif width == 2:
            x = np.frombuffer(pcm, "<i2").astype(np.float32) / 32768.0
        elif width == 3:
            b = np.frombuffer(pcm, np.uint8).reshape(-1, 3).astype(np.int32)
            v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            x = np.where(v >= 1 << 23, v - (1 << 24), v).astype(np.float32) / 8388608.0
        elif width == 4:
            x = (np.frombuffer(pcm, "<i4").astype(np.float64) / 2147483648.0).astype(np.float32)
        else:
            raise ValueError(f"{path.name}: unsupported PCM width {bits}-bit")
    elif tag == _WAVE_FORMAT_FLOAT:
        if width not in (4, 8):
            raise ValueError(f"{path.name}: unsupported float width {bits}-bit")
        x = np.frombuffer(pcm, "<f4" if width == 4 else "<f8").astype(np.float32)
    else:
        raise ValueError(f"{path.name}: unsupported WAV format tag {tag}")
    if not np.isfinite(x).all():
        raise ValueError(f"{path.name}: contains non-finite samples")
    return torch.from_numpy(np.ascontiguousarray(x.reshape(frames, channels).T)), int(sr)


def load_audio(
    path: str | Path,
    target_sample_rate: int | None = None,
    target_channels: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Load a WAV audio file and return a float32 tensor.

    Parameters
    ----------
    path : str | Path
        Path to the WAV audio file.
    target_sample_rate : int, optional
        Target sample rate. Resampled if different.
    target_channels : int, optional
        Target channel count. Mono downmixed if needed.

    Returns
    -------
    audio : torch.Tensor
        Shape ``(C, T)``, dtype float32, values in [-1, 1].
    sample_rate : int
        Sample rate in Hz.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sr = _decode_wav(path)  # (C, T) float32 in native layout

    target_sr = target_sample_rate or get_sample_rate()
    target_ch = target_channels    or get_channels()

    # Mix down to target channels
    if target_ch == 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif target_ch > 1 and waveform.shape[0] == 1:
        waveform = waveform.expand(target_ch, -1)

    # Resample to target sample rate if needed
    if sr != target_sr:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
        logger.info("Resampled '%s' from %d Hz to %d Hz", path.name, sr, target_sr)

    # Check duration limit
    dur_s = waveform.shape[-1] / target_sr
    max_dur = get_max_duration_seconds()
    if dur_s > max_dur:
        # system limit (config/system_config.yaml: audio.max_duration_seconds); never silent
        logger.warning("'%s' is %.2f s; truncating to the configured system limit of %.1f s", path.name, dur_s, max_dur)
        waveform = waveform[..., :int(max_dur * target_sr)]

    logger.info("Loaded '%s' — shape=%s, sr=%d", path.name, tuple(waveform.shape), target_sr)
    return waveform, target_sr


def save_audio(
    audio: torch.Tensor,
    path: str | Path,
    sample_rate: int | None = None,
) -> None:
    """Save an audio tensor to a 16-bit PCM WAV file.

    Parameters
    ----------
    audio : torch.Tensor
        Shape ``(C, T)`` or ``(B, C, T)``.
    path : str | Path
        Output file path.
    sample_rate : int, optional
        Sample rate in Hz.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sr = sample_rate or get_sample_rate()
    waveform = audio.detach().cpu()

    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim != 2:
        raise ValueError(
            f"Expected (C, T) or (B, C, T) tensor, got {tuple(audio.shape)}"
        )

    if not torch.isfinite(waveform).all():
        raise ValueError("Refusing to save non-finite audio")
    # Clamp to [-1, 1]
    waveform = waveform.clamp(-1.0, 1.0)
    n_channels, n_frames = waveform.shape

    # Convert to 16-bit PCM
    int_samples = (waveform * 32767.0).round().to(torch.int16)
    # Interleave channels: (C, T) -> (T, C) -> flatten
    interleaved = int_samples.t().contiguous().view(-1).tolist()

    raw_bytes = struct.pack(f"<{len(interleaved)}h", *interleaved)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)  # 2 bytes = 16-bit
        wf.setframerate(sr)
        wf.writeframes(raw_bytes)

    logger.info("Saved audio to '%s' — shape=%s, sr=%d", path, tuple(waveform.shape), sr)


def generate_test_tone(
    frequency_hz: float = 440.0,
    duration_s: float = 1.0,
    amplitude: float = 0.5,
    sample_rate: int | None = None,
    channels: int | None = None,
    duration_seconds: float | None = None,
) -> torch.Tensor:
    """Generate a pure sine-wave test tone.

    Returns
    -------
    torch.Tensor
        Shape ``(C, T)``, dtype float32.
    """
    if duration_seconds is not None:
        duration_s = duration_seconds
    sr  = sample_rate or get_sample_rate()
    ch  = channels    or get_channels()
    T   = int(duration_s * sr)
    t   = torch.linspace(0.0, duration_s, T)
    tone = amplitude * torch.sin(2.0 * math.pi * frequency_hz * t)
    return tone.unsqueeze(0).expand(ch, -1)
