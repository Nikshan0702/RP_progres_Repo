
cat > generate_ground_truth_pair.py <<'PY'
from pathlib import Path
import json
import torch

from audio_io import generate_test_tone, save_audio, load_audio
from dapn.dapn import DAPN

output_dir = Path("Testing/dapn_ground_truth")
output_dir.mkdir(parents=True, exist_ok=True)

sample_rate = 44100
duration = 3.0

tone1 = generate_test_tone(
    frequency_hz=220.0,
    duration_s=duration,
    amplitude=0.35,
    sample_rate=sample_rate,
)
tone2 = generate_test_tone(
    frequency_hz=440.0,
    duration_s=duration,
    amplitude=0.20,
    sample_rate=sample_rate,
)

dry_audio = tone1 + tone2
dry_path = output_dir / "ground_truth_dry.wav"
wet_path = output_dir / "ground_truth_wet.wav"

save_audio(dry_audio, dry_path, sample_rate=sample_rate)
audio, loaded_sr = load_audio(dry_path)

ground_truth = {
    "chain": [
        {
            "effect": "EQ",
            "parameters": {
                "frequency": 0.35,
                "gain": 0.75,
                "q": 0.40,
            },
        },
        {
            "effect": "Compressor",
            "parameters": {
                "threshold": 0.50,
                "ratio": 0.40,
                "attack": 0.20,
                "release": 0.30,
                "makeup_gain": 0.35,
            },
        },
        {
            "effect": "Reverb",
            "parameters": {
                "room_size": 0.65,
                "damping": 0.35,
                "decay": 0.60,
                "wet_mix": 0.55,
                "dry_mix": 0.45,
            },
        },
    ]
}

if loaded_sr != sample_rate:
    raise ValueError(
        f"Expected {sample_rate} Hz, got {loaded_sr} Hz"
    )

dapn = DAPN()

with torch.no_grad():
    wet_audio = dapn.process(audio, ground_truth)

if not torch.isfinite(wet_audio).all():
    raise ValueError("Wet audio contains NaN or infinity")

save_audio(wet_audio, wet_path, sample_rate=loaded_sr)

metadata = {
    "sample_rate": loaded_sr,
    "dry_file": str(dry_path),
    "wet_file": str(wet_path),
    "ground_truth": ground_truth,
    "note": (
        "This records the chain and parameters supplied to DAPN. "
        "It does not guarantee exact recovery by the predictor."
    ),
}

(output_dir / "ground_truth.json").write_text(
    json.dumps(metadata, indent=2)
)

print("Ground-truth audio generated successfully.")
print(f"Dry audio: {dry_path}")
print(f"Wet audio: {wet_path}")
print(f"Metadata: {output_dir / 'ground_truth.json'}")
PY
