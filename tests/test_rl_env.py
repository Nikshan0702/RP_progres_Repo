"""
tests/test_rl_env.py
====================
Unit tests for the Goal-Conditioned Audio Effects RL Environment (AudioFXEnv).
"""

import pytest
import torch
from pathlib import Path
from rl.dapn_env import AudioFXEnv


@pytest.fixture
def dummy_audio():
    """Generates synthetic 2-second audio pair."""
    sr = 44100
    t = torch.linspace(0, 2.0, sr * 2)
    # Dry: pure tone
    dry = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)
    # Target: tone + harmonics
    target = (torch.sin(2 * 3.14159 * 440 * t) + 0.5 * torch.sin(2 * 3.14159 * 880 * t)).unsqueeze(0)
    return dry, target


def test_env_init():
    """Test environment initialization and spaces."""
    env = AudioFXEnv(predictor_checkpoint=None)
    assert env.action_dim == 36
    assert env.obs_dim == 100
    assert env.num_effects == 7
    assert env.num_params == 29


def test_env_reset(dummy_audio):
    """Test reset without warm-start."""
    dry, target = dummy_audio
    env = AudioFXEnv(predictor_checkpoint=None)

    obs, info = env.reset(dry, target, use_warm_start=False)

    assert obs.shape == (100,)
    assert isinstance(info["initial_loss"], float)
    assert info["num_active_effects"] == 0
    assert env.current_step == 0


def test_env_reset_warm_start(dummy_audio):
    """Test reset with trained model checkpoint warm-start."""
    dry, target = dummy_audio
    ckpt_path = Path("checkpoints/best_model.pth")
    if not ckpt_path.exists():
        pytest.skip("Checkpoint not found for warm-start test")

    env = AudioFXEnv(predictor_checkpoint=ckpt_path)
    obs, info = env.reset(dry, target, use_warm_start=True)

    assert obs.shape == (100,)
    assert info["warm_started"] is True
    # The predictor should have initialized candidate presence/params
    assert env.current_presence.shape == (7,)
    assert env.current_params.shape == (29,)


def test_env_step(dummy_audio):
    """Test step execution, action effects, and reward computation."""
    dry, target = dummy_audio
    env = AudioFXEnv(predictor_checkpoint=None, max_steps_per_episode=5)
    obs, _ = env.reset(dry, target, use_warm_start=False)

    # Action: Turn on Reverb (index 3 > 0), adjust first param
    action = torch.zeros(36)
    action[3] = 1.0  # Turn on Reverb
    action[7] = 0.5  # Adjust first parameter

    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.shape == (100,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["step"] == 1
    assert "improvement_pct" in info
    assert env.current_presence[3].item() == 1.0


def test_env_truncation(dummy_audio):
    """Test episode truncation after max steps."""
    dry, target = dummy_audio
    env = AudioFXEnv(predictor_checkpoint=None, max_steps_per_episode=3)
    env.reset(dry, target, use_warm_start=False)

    action = torch.zeros(36)
    for _ in range(2):
        _, _, _, truncated, _ = env.step(action)
        assert truncated is False

    _, _, _, truncated, _ = env.step(action)
    assert truncated is True
