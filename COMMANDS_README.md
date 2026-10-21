# DAPN & RL Execution Commands Reference Manual

This guide contains the complete command-line interface (CLI) reference for running:
1. **DAPN Proof Suite**: Processing test audio through each DAPN effect individually and extracting acoustic verification metrics.
2. **Inverse FX Chain Predictor**: Detecting audio effect chains and predicting knob parameters (normalized and physical DSP values).
3. **Reinforcement Learning Agent (GRRL)**: Running goal-oriented RL reconstruction to match wet audio.
4. **Audio Analysis & Diagnostics**: Extracting dynamic and spectral metrics from audio files.
5. **Autograd & Unit Verification**: Validating differentiability and DSP behavior.

---

## Quick Navigation

- [1. DAPN Proof Suite (All 7 Effects)](#1-dapn-proof-suite-all-7-effects)
- [2. Apply Individual DAPN Effects (One by One)](#2-apply-individual-dapn-effects-one-by-one)
- [3. Predict FX Chain & Knob Values (InverseFXNet)](#3-predict-fx-chain--knob-values-inversefxnet)
- [4. Execute RL Agent (GRRL Reconstruction)](#4-execute-rl-agent-grrl-reconstruction)
- [5. Audio Signal & Acoustic Feature Extraction](#5-audio-signal--acoustic-feature-extraction)
- [6. Mathematical & Differentiability Checks](#6-mathematical--differentiability-checks)
- [7. Directory & Output Structure](#7-directory--output-structure)

---

## 1. DAPN Proof Suite (All 7 Effects)

Run the all-in-one verification script to process clean dry audio through all 7 DAPN effects (`EQ`, `Compressor`, `Delay`, `Reverb`, `Distortion`, `Chorus`, `Flanger`), verify finite math, compute Multi-Resolution STFT distance, and export 7 WAV files and a JSON report:

```bash
python3 prove_dapn_all_effects.py \
  --input Testing/clear_test_dry.wav \
  --output-dir results/dapn_proof
```

### What this produces:
- `results/dapn_proof/dapn_eq.wav`
- `results/dapn_proof/dapn_compressor.wav`
- `results/dapn_proof/dapn_delay.wav`
- `results/dapn_proof/dapn_reverb.wav`
- `results/dapn_proof/dapn_distortion.wav`
- `results/dapn_proof/dapn_chorus.wav`
- `results/dapn_proof/dapn_flanger.wav`
- `results/dapn_proof/dapn_proof_summary.json` (Full metric report)

---

## 2. Apply Individual DAPN Effects (One by One)

Use `examples/apply_fx.py` to process dry audio through any individual effect with custom output destinations.

### Reverb Only
```bash
python3 examples/apply_fx.py \
  --input Testing/clear_test_dry.wav \
  --effects Reverb \
  --output results/fx_reverb.wav
```

### Delay Only
```bash
python3 examples/apply_fx.py \
  --input Testing/clear_test_dry.wav \
  --effects Delay \
  --output results/fx_delay.wav
```

### EQ Only
```bash
python3 examples/apply_fx.py \
  --input Testing/clear_test_dry.wav \
  --effects EQ \
  --output results/fx_eq.wav
```

### Compressor Only
```bash
python3 examples/apply_fx.py \
  --input Testing/clear_test_dry.wav \
  --effects Compressor \
  --output results/fx_compressor.wav
```

### Chorus Only
```bash
python3 examples/apply_fx.py \
  --input Testing/clear_test_dry.wav \
  --effects Chorus \
  --output results/fx_chorus.wav
```

### Distortion Only
```bash
python3 examples/apply_fx.py \
  --input Testing/clear_test_dry.wav \
  --effects Distortion \
  --output results/fx_distortion.wav
```

### Flanger Only
```bash
python3 examples/apply_fx.py \
  --input Testing/clear_test_dry.wav \
  --effects Flanger \
  --output results/fx_flanger.wav
```

### Chaining Multiple Effects (e.g. EQ + Compressor + Reverb)
```bash
python3 examples/apply_fx.py \
  --input Testing/clear_test_dry.wav \
  --effects EQ Compressor Reverb \
  --output results/fx_chain_output.wav
```

### Applying Built-in Presets
Presets: `ambient_space`, `rock_drive`, `clean_chorus`, `vocal_polish`
```bash
python3 examples/apply_fx.py \
  --input Testing/clear_test_dry.wav \
  --preset ambient_space \
  --output results/fx_ambient_preset.wav
```

---

## 3. Predict FX Chain & Knob Values (InverseFXNet)

### Predict Complete Chain & Order Sequence
Evaluates the (Dry, Wet) audio pair, decodes effect presence, order sequence, and knob settings:

```bash
python3 predict_fx.py \
  --dry Testing/clear_test_dry.wav \
  --wet results/dapn_proof/dapn_reverb.wav \
  --checkpoint checkpoints_v2/best_model.pth
```

### Inspect Specific Effect Knobs (Normalized & Physical Units)
Isolates a specific effect and prints its knob values with unit conversions (dB, Hz, ms, ratio):

```bash
python3 predict_fx.py \
  --dry Testing/clear_test_dry.wav \
  --wet results/dapn_proof/dapn_reverb.wav \
  --effect Reverb \
  --checkpoint checkpoints_v2/best_model.pth
```
*(Options for `--effect`: `EQ`, `Compressor`, `Delay`, `Reverb`, `Distortion`, `Chorus`, `Flanger`)*

---

## 4. Execute RL Agent (GRRL Reconstruction)

Executes the Graph-Generating Reinforcement Learning (GRRL) agent to iteratively match the target wet sound.

### Standard Execution (Warm-Start with Predictor Baseline)
```bash
python3 rl/process_pair.py \
  --dry Testing/clear_test_dry.wav \
  --wet Testing/clear_test_wet.wav \
  --output results/clear_test_output.wav \
  --warm-start \
  --max-steps 10 \
  --predictor-ckpt checkpoints_v2/best_model.pth \
  --grrl-ckpt checkpoints_v2/grrl_warm_edit/best_grrl_model.pth
```

### Fast Execution (Skip Per-Step PNG Generation)
```bash
python3 rl/process_pair.py \
  --dry Testing/clear_test_dry.wav \
  --wet Testing/clear_test_wet.wav \
  --output results/clear_test_output.wav \
  --warm-start \
  --max-steps 10 \
  --no-step-images \
  --predictor-ckpt checkpoints_v2/best_model.pth \
  --grrl-ckpt checkpoints_v2/grrl_warm_edit/best_grrl_model.pth
```

### RL Output Files Generated:
1. `results/clear_test_output.wav` (Synthesized audio)
2. `results/clear_test_output_details.json` (Per-step action & loss trajectory report)
3. `results/clear_test_output_charts.svg` (High-resolution loss & reward analytics vector chart)
4. `results/clear_test_output_steps/` (Step-by-step waveform/spectrogram diagnostic PNGs)

---

## 5. Audio Signal & Acoustic Feature Extraction

### Comprehensive Feature Extraction (Pure PyTorch)
Extracts Peak dBFS, RMS dBFS, Crest Factor, Zero Crossing Rate, Active/Silence ratio, and Spectral Centroid:

```bash
python3 examples/analyze_audio.py \
  --input results/dapn_proof/dapn_reverb.wav \
  --output results/reverb_acoustic_features.json
```

### Process Preset & Generate Signal Quality Report
```bash
python3 examples/process_file.py \
  --input Testing/clear_test_dry.wav \
  --output results/dapn_processed.wav \
  --preset full_chain \
  --report results/dapn_full_chain_report.json
```
*(Preset choices: `full_chain`, `reverb_only`, `delay_only`, `comp_only`, `eq_only`)*

---

## 6. Mathematical & Differentiability Checks

### Autograd Gradient Backpropagation Test
Verifies that gradients pass non-zero through all DSP components without exploding or vanishing:
```bash
python3 examples/run_basic_dapn.py
```

### Differentiable Parameter Optimization (Gradient Descent Match)
Proves that DAPN parameters smoothly converge toward a target sound via Adam optimizer and Multi-Resolution STFT Loss:
```bash
python3 examples/optimize_fx_match.py \
  --input Testing/clear_test_dry.wav \
  --steps 30 \
  --lr 0.04
```

### Run Full Test Suite (Pytest)
Executes all 78 tests covering DSP equations, chain non-commutativity, and Autograd graphs:
```bash
pytest tests/test_gradients.py tests/test_fx_chain.py tests/test_eq.py tests/test_compressor.py tests/test_delay.py tests/test_reverb.py -v
```

---

## 7. Directory & Output Structure

```
RP2_A/
├── checkpoints_v2/
│   ├── best_model.pth                      # InverseFXNet checkpoint (v2)
│   └── grrl_warm_edit/
│       └── best_grrl_model.pth             # Trained GRRL agent checkpoint
├── Testing/
│   ├── clear_test_dry.wav                  # Input clean dry audio
│   └── clear_test_wet.wav                  # Reference wet target audio
├── results/
│   ├── dapn_proof/                         # Output of prove_dapn_all_effects.py
│   │   ├── dapn_eq.wav
│   │   ├── dapn_compressor.wav
│   │   ├── dapn_delay.wav
│   │   ├── dapn_reverb.wav
│   │   ├── dapn_distortion.wav
│   │   ├── dapn_chorus.wav
│   │   ├── dapn_flanger.wav
│   │   └── dapn_proof_summary.json
│   ├── clear_test_output.wav               # Output from RL agent
│   ├── clear_test_output_details.json      # Step-by-step metrics
│   ├── clear_test_output_charts.svg        # SVG analytics charts
│   └── clear_test_output_steps/            # Per-step PNGs
```
