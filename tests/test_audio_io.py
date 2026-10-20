"""WAV decoding must follow the declared encoding (PCM width / IEEE float)."""
import struct
import numpy as np
import pytest
import torch
from audio_io import load_audio, save_audio

SR = 44100


def _wav(path, tag, bits, samples, channels=1, sr=SR, extensible=False):
    width = bits // 8
    if tag == 1 and bits == 8:
        raw = (samples * 127 + 128).astype(np.uint8).tobytes()
    elif tag == 1 and bits == 16:
        raw = (samples * 32767).astype("<i2").tobytes()
    elif tag == 1 and bits == 24:
        v = (samples * 8388607).astype(np.int32)
        raw = b"".join(struct.pack("<i", int(i))[:3] for i in v)
    elif tag == 1 and bits == 32:
        raw = (samples.astype(np.float64) * 2147483647).astype("<i4").tobytes()
    else:
        raw = samples.astype("<f4" if bits == 32 else "<f8").tobytes()
    ba = width * channels
    if extensible:
        fmt = struct.pack("<HHIIHH", 0xFFFE, channels, sr, sr * ba, ba, bits)
        fmt += struct.pack("<HHI", 22, bits, 4) + struct.pack("<H", tag) + b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
    else:
        fmt = struct.pack("<HHIIHH", tag, channels, sr, sr * ba, ba, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(raw)) + raw
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


@pytest.fixture
def ref():
    t = np.arange(2048) / SR
    return (0.6 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.mark.parametrize("tag,bits,tol", [(1, 16, 1e-4), (1, 24, 1e-6), (1, 32, 1e-6), (3, 32, 0), (3, 64, 1e-7), (1, 8, 2e-2)])
@pytest.mark.parametrize("extensible", [False, True])
def test_decodes_declared_encoding(tmp_path, ref, tag, bits, tol, extensible):
    p = tmp_path / "a.wav"
    _wav(p, tag, bits, ref, extensible=extensible)
    x, sr = load_audio(p)
    assert sr == SR and x.shape == (1, len(ref))
    assert np.abs(x[0].numpy() - ref).max() <= tol + 1e-9


def test_int32_pcm_is_not_read_as_float(tmp_path, ref):
    p = tmp_path / "i32.wav"
    _wav(p, 1, 32, ref)
    assert load_audio(p)[0].abs().max() < 1.0  # float reinterpretation would give ~1e-3/NaN garbage


def test_stereo_downmix_and_resample(tmp_path, ref):
    st = np.stack([ref, -ref], axis=1).reshape(-1)
    p = tmp_path / "s.wav"
    _wav(p, 1, 16, st, channels=2)
    x, _ = load_audio(p)
    assert x.shape[0] == 1 and x.abs().max() < 1e-3
    q = tmp_path / "r.wav"
    _wav(q, 1, 16, ref, sr=22050)
    y, sr = load_audio(q)
    assert sr == SR and abs(y.shape[-1] - 2 * len(ref)) <= 2


def test_rejects_bad_files(tmp_path):
    (tmp_path / "x.wav").write_bytes(b"not a wav")
    with pytest.raises(ValueError):
        load_audio(tmp_path / "x.wav")
    p = tmp_path / "nan.wav"
    _wav(p, 3, 32, np.array([0.0, np.nan], dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        load_audio(p)
    p = tmp_path / "empty.wav"
    _wav(p, 1, 16, np.zeros(0, dtype=np.float32))
    with pytest.raises(ValueError):
        load_audio(p)
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "missing.wav")


def test_save_roundtrip_and_nonfinite(tmp_path, ref):
    p = tmp_path / "o.wav"
    save_audio(torch.from_numpy(ref)[None], p)
    assert torch.allclose(load_audio(p)[0][0], torch.from_numpy(ref), atol=1e-4)
    with pytest.raises(ValueError):
        save_audio(torch.tensor([[float("nan")]]), p)
