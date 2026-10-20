"""
rl/visualization/config.py
==========================
Configuration and constants for RL Audio Processing Visualization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common.config_loader import load_effects_config, load_system_config

# Project root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default network settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080

# Checkpoint paths
DEFAULT_PREDICTOR_CKPT = PROJECT_ROOT / "checkpoints" / "best_model.pth"
DEFAULT_GRRL_CKPT = PROJECT_ROOT / "checkpoints" / "grrl" / "best_grrl_model.pth"
DEFAULT_PPO_CKPT = PROJECT_ROOT / "checkpoints" / "ppo" / "best_ppo_model.pth"

# Preset audio file pairs
DEFAULT_PRESETS = [
    {
        "id": "lead_guitar",
        "name": "Preset 1: Synth/Guitar Lead (0.5s)",
        "dry_path": str(PROJECT_ROOT / "audio" / "dry.wav"),
        "wet_path": str(PROJECT_ROOT / "audio" / "wet.wav"),
        "description": "Standard benchmark test pair for DAPN FX synthesis.",
    },
    {
        "id": "acoustic_reverb",
        "name": "Preset 2: Acoustic Clean + Reverb/Delay",
        "dry_path": str(PROJECT_ROOT / "audio" / "testFile.wav"),
        "wet_path": str(PROJECT_ROOT / "audio" / "test_wet_reverb_delay.wav"),
        "description": "Clean acoustic guitar processed with spatial reverb and delay.",
    },
    {
        "id": "acoustic_rock",
        "name": "Preset 3: Acoustic Clean + Target Rock Tone",
        "dry_path": str(PROJECT_ROOT / "audio" / "testFile.wav"),
        "wet_path": str(PROJECT_ROOT / "audio" / "testFile_target_rock.wav"),
        "description": "Clean acoustic guitar matching an aggressive target rock profile.",
    },
]

# Audio processing defaults
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_AUDIO_DURATION = 2.0  # seconds
DEFAULT_TARGET_LOSS_THRESHOLD = 0.40
DEFAULT_MAX_CHAIN_LENGTH = 8
DEFAULT_STEP_PENALTY = 0.05

# Workflow Stages
WORKFLOW_STAGES = [
    {
        "id": "upload",
        "name": "Audio Upload",
        "title": "Audio Pair Upload (Dry + Wet Reference)",
        "description": "Receives the unprocessed input (Dry) and target reference sound (Wet).",
        "icon": "upload",
    },
    {
        "id": "preprocess",
        "name": "Preprocessing",
        "title": "Audio Preprocessing",
        "description": "Standardizes sample rate (44.1kHz), channels (mono), and duration.",
        "icon": "audio",
    },
    {
        "id": "state",
        "name": "RL Agent State",
        "title": "RL Agent State Representation",
        "description": "Constructs acoustic spectral distance, chain history, and progress features (148 dims).",
        "icon": "state",
    },
    {
        "id": "action",
        "name": "Action Selection",
        "title": "Action Selection & Policy Sampling",
        "description": "Masks invalid actions; samples discrete effect and continuous parameters from Actor-Critic.",
        "icon": "target",
    },
    {
        "id": "chain",
        "name": "FX Chain Generation",
        "title": "FX Chain Generation & Parameter Mapping",
        "description": "Appends the effect to sequence and maps normalized [0, 1] knobs to physical DSP ranges.",
        "icon": "chain",
    },
    {
        "id": "dsp",
        "name": "DAPN Processing",
        "title": "DAPN Audio Processing",
        "description": "Executes differentiable audio DSP filters sequentially on dry audio waveform.",
        "icon": "dsp",
    },
    {
        "id": "loss",
        "name": "Loss Calculation",
        "title": "Loss Calculation (Multi-Resolution STFT)",
        "description": "Computes spectral convergence and log-STFT distance between synthesized audio and target.",
        "icon": "chart",
    },
    {
        "id": "reward",
        "name": "Reward Calculation",
        "title": "Reward Calculation & Regularization",
        "description": "Calculates delta-loss improvement reward, step length penalty, and terminal bonus.",
        "icon": "trophy",
    },
    {
        "id": "next",
        "name": "Next State / Action",
        "title": "Next State Transition & Check",
        "description": "Evaluates stopping criteria (target loss reached, max steps, or STOP action).",
        "icon": "repeat",
    },
    {
        "id": "stop",
        "name": "STOP Action",
        "title": "Chain Synthesis Termination (STOP)",
        "description": "Synthesizer finishes; chain graph finalized without redundant trailing effects.",
        "icon": "stop",
    },
    {
        "id": "output",
        "name": "Final Audio Output",
        "title": "Final Audio Output & A/B Comparison",
        "description": "Generates final WAV, waveform visualizations, and downloadable audio artifact.",
        "icon": "speaker",
    },
]


def get_effects_spec_map() -> dict[str, Any]:
    """Retrieve effect parameter specs and physical ranges from configuration."""
    raw_cfg = load_effects_config()
    effects = raw_cfg.get("effects", {})
    return effects
