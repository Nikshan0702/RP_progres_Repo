# DAPN RL Audio Synthesis Studio

An interactive, clean, and modern dashboard visualizing the sequential reinforcement learning (RL) audio processing workflow in the **Differentiable Audio Processing Network (DAPN)** project.

---

## Architecture & Workflow Overview

The visualization displays the complete end-to-end 11-stage audio processing pipeline:

```
[1. Audio Pair Upload] ➔ [2. Audio Preprocessing] ➔ [3. RL Agent State] ➔ [4. Action Selection]
        ➔ [5. FX Chain Generation] ➔ [6. DAPN Audio Processing] ➔ [7. Loss Calculation]
        ➔ [8. Reward Calculation] ➔ [9. Next State / Action] ➔ [10. STOP] ➔ [11. Final Audio Output]
```

### Complete 11-Stage Workflow Breakdown

1. **Audio Pair Upload (`upload`)**: Loads the unprocessed input (Dry) and target reference sound (Wet). Displays filenames, durations, channels, and sample rates.
2. **Audio Preprocessing (`preprocess`)**: Formats both audio signals into standardized $44.1\,\text{kHz}$ mono tensors $(1, 1, T)$ bounded to max duration (2.0s). Computes baseline initial spectral loss $L_0$.
3. **RL Agent State (`state`)**: Constructs the 148-dimensional observation vector:
   - Acoustic STFT spectral distance embedding (64 bins)
   - Ordered chain history matrix (8 slots $\times$ 8 one-hot classes = 64 dims)
   - Progress and scalar metrics: step norm, chain occupancy, loss norm, improvement ratio (4 dims)
   - PTA taste conditioning vector (16 dims)
4. **Action Selection (`action`)**: Evaluates syntactic action masks (enforcing min/max chain lengths and prohibiting unauthorized repeats); samples hybrid discrete effect from Actor-Critic policy head and continuous parameters.
5. **FX Chain Generation (`chain`)**: Appends the chosen effect to the sequence; maps normalized $[0, 1]$ parameters to exact physical DSP ranges (Hz, dB, ms, ratios) using `config/effects_config.yaml`.
6. **DAPN Audio Processing (`dsp`)**: Streams dry audio through the ordered differentiable DSP chain using PyTorch DAPN filters.
7. **Loss Calculation (`loss`)**: Calculates the multi-resolution STFT loss (spectral convergence + log-magnitude distance) between the intermediate synthesized audio and the target reference.
8. **Reward Calculation (`reward`)**: Computes step reward:
   $$R_t = 10 \cdot (L_{t-1} - L_t) - 0.05 - \text{penalty}_{\text{mask}} + \text{bonus}_{\text{terminal}}$$
9. **Next State / Next Action (`next`)**: Checks stopping criteria (target loss $\le 0.40$, max chain length reached, or STOP action).
10. **STOP (`stop`)**: Explicit termination of chain synthesis when STOP is chosen or stopping conditions are met.
11. **Final Audio Output (`output`)**: Generates final wet audio, downsamples waveform envelopes, provides A/B/C player controls, and generates downloadable WAV output.

---

## Key Features

- **Action Graph & Node Progression**: Every action is visualized as a distinct node in the sequential FX chain graph with previous vs. new parameter deltas ($\Delta$), normalized knob sliders, and physical values with units.
- **Interactive SVG Charts**:
  - **MR-STFT Loss Trajectory**: Live curve tracking loss across step iterations with baseline reference and $0.40$ target threshold lines.
  - **Reward Progression**: Positive/negative color-coded step reward bars with cumulative return polyline overlay.
- **Waveform Visualizers & Synced Audio Playback**: Three synchronized audio channels (**Dry Input**, **Target Reference**, and **RL Generated Output**) with interactive seek, play/pause, scrubbers, and download button.
- **Real RL vs. Demo Mode**:
  - **Real RL Mode**: Uses PyTorch DAPN and the trained GRRL Actor-Critic checkpoint (`checkpoints/grrl/best_grrl_model.pth`).
  - **Demo Mode**: Clearly labeled simulated mode for testing and UI inspection without presenting synthetic data as real results.
- **Zero External Dependencies**: Implemented using the Python standard library `http.server` backend and vanilla HTML5 / CSS3 / ES6 / Canvas / SVG frontend. Runs out of the box on Python 3 without needing npm or pip web frameworks.

---

## How to Run

### 1. Launch Web Server

From the project root:

```bash
# Default (http://127.0.0.1:8080)
python3 rl/visualize.py

# Custom port
python3 rl/visualize.py --port 8085

# Headless / No auto-browser
python3 rl/visualize.py --no-browser

# Launch directly in Demo Mode
python3 rl/visualize.py --mode demo
```

Alternatively:
```bash
python3 rl/visualization/server.py --port 8080
```

### 2. Run Automated Tests

To test the visualization components and server routing:

```bash
pytest rl/test_visualization.py
```

To run the full project test suite (165 tests):

```bash
pytest
```

---

## REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Web dashboard single-page interface |
| `GET` | `/api/status` | Server health, model status, and presets |
| `GET` | `/api/presets` | Available audio preset pairs |
| `GET` | `/api/summary` | Current session state and stage diagnostics |
| `POST` | `/api/initialize` | Initialize session with dry/wet audio |
| `POST` | `/api/step` | Advance one RL action step forward |
| `POST` | `/api/run_all` | Execute workflow to completion |
| `POST` | `/api/reset` | Reset episode to start |
| `POST` | `/api/upload` | Upload custom dry/wet audio pair |
| `GET` | `/api/audio?type={dry\|wet\|output}` | Stream WAV audio for playback |
| `GET` | `/api/download?type=output` | Download synthesized audio file |
| `GET` | `/api/export_json` | Export full execution report and FX chain JSON |
