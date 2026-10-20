"""
rl/dapn_env.py
==============
Goal-Conditioned Audio Effects Chain Environment for Reinforcement Learning.

Provides an environment for training an RL agent to:
1. Interactively fine-tune DSP knob parameters.
2. Optimize effect chain ordering and prune unnecessary effects (sparsity).
3. Maximize acoustic spectral matching against a goal target sound.

State Space (Observations):
---------------------------
- Feature 1: Acoustic distance embedding between Current Audio and Target Audio.
- Feature 2: Effect Presence Vector (7 binary flags for EQ, Comp, Delay, Reverb, Distortion, Chorus, Flanger).
- Feature 3: Current Continuous Parameter Values (29 continuous values in [0, 1]).

Action Space:
-------------
Continuous Action vector of dimension 36:
- Indices 0..6: Presence logits for the 7 effects (> 0.0 activates, <= 0.0 bypasses).
- Indices 7..35: Knob parameter adjustments Delta in [-0.05, +0.05] applied to current knob values.

Reward Function:
----------------
R_t = (Loss_{t-1} - Loss_t) * scale - lambda_sparse * N_active_effects + R_bonus
"""

from __future__ import annotations

import copy
import logging
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from common.audio_pair import prepare_audio_pair
from common.config_loader import get_sample_rate
from common.losses import MultiResolutionSTFTLoss
from dapn.dapn import DAPN
from dataset.dapn_dataset import PARAM_OFFSETS, PARAM_SCHEMA, SUPPORTED_EFFECTS
from models.inverse_fx_net import InverseFXNet


class AudioFXEnv:
    """Goal-Conditioned Reinforcement Learning Environment for Audio FX Chain Synthesis."""

    def __init__(
        self,
        predictor_checkpoint: str | Path | None = None,
        device: str = "cpu",
        max_steps_per_episode: int = 15,
        sparsity_weight: float = 0.02,
        target_threshold_loss: float = 0.40,
        sample_rate: int = 44100,
        audio_duration_sec: float = 2.0,
    ) -> None:
        self.device = torch.device(device)
        self.max_steps = max_steps_per_episode
        self.sparsity_weight = sparsity_weight
        self.target_threshold_loss = target_threshold_loss
        self.sample_rate = get_sample_rate()
        if sample_rate != self.sample_rate: raise ValueError("Environment/DAPN sample rates differ")
        self.audio_duration_sec = audio_duration_sec
        self._done = True
        self.max_samples = int(sample_rate * audio_duration_sec)

        # DSP Engine and Loss Function
        self.dapn = DAPN()
        self.loss_fn = MultiResolutionSTFTLoss()

        # Load warm-start predictor model if checkpoint provided
        self.predictor: InverseFXNet | None = None
        if predictor_checkpoint is not None:
            self.predictor = InverseFXNet().to(self.device)
            self.predictor.load_checkpoint(predictor_checkpoint)
            self.predictor.eval()

        # Dimensions
        self.num_effects = len(SUPPORTED_EFFECTS)
        self.num_params = sum(len(p) for p in PARAM_SCHEMA.values())  # 29
        self.action_dim = self.num_effects + self.num_params  # 36
        self.obs_dim = 64 + self.num_effects + self.num_params  # Acoustic embedding (64) + Presence (7) + Params (29) = 100

        # State tracking
        self.dry_audio: torch.Tensor | None = None
        self.target_audio: torch.Tensor | None = None
        self.current_audio: torch.Tensor | None = None
        self.current_presence: torch.Tensor = torch.zeros(self.num_effects, device=self.device)
        self.current_params: torch.Tensor = torch.full((self.num_params,), 0.5, device=self.device)
        self.current_loss: float = 999.0
        self.initial_loss: float = 999.0
        self.current_step: int = 0

    def reset(
        self,
        dry_audio: torch.Tensor,
        target_audio: torch.Tensor,
        use_warm_start: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset environment with new dry and target audio pair."""
        if use_warm_start and self.predictor is None:
            raise ValueError("Warm-start requires a compatible predictor checkpoint")
        self.dry_audio, self.target_audio = prepare_audio_pair(dry_audio, target_audio, device=self.device,
            sample_rate=self.sample_rate, duration=self.audio_duration_sec)
        self.current_audio = None
        self.current_step = 0
        self._done = False
        # Initial baseline loss (dry vs target without any effects)
        with torch.no_grad():
            self.initial_loss = float(self.loss_fn(self.dry_audio, self.target_audio).item())

        # Warm-start policy initialization using InverseFXNet
        if use_warm_start and self.predictor is not None:
            with torch.no_grad():
                logits, pred_params = self.predictor(self.dry_audio, self.target_audio)
                probs = torch.sigmoid(logits)[0]
                self.current_presence = (probs >= 0.5).float()
                self.current_params = torch.clamp(pred_params[0], 0.0, 1.0)
        else:
            self.current_presence = torch.zeros(self.num_effects, device=self.device)
            self.current_params = torch.full((self.num_params,), 0.5, device=self.device)

        # Render current sound
        self._render_current_sound()
        with torch.no_grad():
            self.current_loss = float(self.loss_fn(self.current_audio, self.target_audio).item())

        self.start_loss = self.current_loss
        self.predictor_loss = self.current_loss if use_warm_start else None
        obs = self._get_observation()
        info = {
            "start_loss": self.start_loss, "predictor_loss": self.predictor_loss,
            "initial_loss": self.initial_loss,
            "current_loss": self.current_loss,
            "num_active_effects": int(self.current_presence.sum().item()),
            "warm_started": use_warm_start and (self.predictor is not None),
        }
        return obs, info

    def step(self, action):
        if self._done: raise RuntimeError("Episode finished; call reset")
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action.shape != (36,) or not torch.isfinite(action).all():
            raise ValueError("Expected 36 finite action values")
        saved = (self.current_presence.clone(), self.current_params.clone(), self.current_audio,
                 self.current_loss, self.current_step)
        try:
            result = self._step(action)
        except Exception:
            self.current_presence,self.current_params,self.current_audio,self.current_loss,self.current_step=saved
            raise
        self._done = result[2] or result[3]
        return result

    def _step(
        self,
        action: torch.Tensor | list[float],
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        """Execute one RL action step."""
        if isinstance(action, (list, tuple)):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        else:
            action = action.to(self.device)

        prev_loss = self.current_loss
        self.current_step += 1

        # Decompose action
        presence_actions = action[:self.num_effects]
        param_deltas = action[self.num_effects:]  # delta in [-0.05, +0.05]

        # Update presence: toggle on if action > 0.0, off if action <= 0.0
        self.current_presence = (presence_actions > 0.0).float()

        # Update continuous knob parameters with small adjustments: clamped in [0, 1]
        self.current_params = torch.clamp(self.current_params + param_deltas * 0.10, 0.0, 1.0)

        # Render audio through DAPN DSP
        self._render_current_sound()

        with torch.no_grad():
            new_loss = float(self.loss_fn(self.current_audio, self.target_audio))
        if not math.isfinite(new_loss): raise RuntimeError("Nonfinite audio loss")
        self.current_loss = new_loss

        # Calculate Reward
        # 1. Spectral improvement reward: positive when loss decreases
        spectral_gain = (prev_loss - new_loss) * 10.0

        # 2. Chain sparsity penalty: encourages using fewer effects
        num_active = float(self.current_presence.sum().item())
        sparsity_penalty = self.sparsity_weight * num_active

        # 3. Target achievement bonus
        target_bonus = 5.0 if new_loss <= self.target_threshold_loss else 0.0

        reward = spectral_gain - sparsity_penalty + target_bonus
        if not math.isfinite(reward):
            reward = -1.0

        # Check termination
        terminated = bool(new_loss <= self.target_threshold_loss)
        truncated = bool(not terminated and self.current_step >= self.max_steps)

        obs = self._get_observation()
        info = {
            "reward_breakdown": {"loss_improvement_reward": spectral_gain, "length_penalty": -sparsity_penalty, "terminal_bonus": target_bonus, "total_step_reward": reward},
            "step": self.current_step,
            "prev_loss": prev_loss,
            "current_loss": new_loss,
            "spectral_gain": spectral_gain,
            "num_active_effects": int(num_active),
            "improvement_pct": (self.start_loss-new_loss) / max(self.start_loss, 1e-8) * 100.0,
        }

        return obs, reward, terminated, truncated, info

    def _render_current_sound(self) -> None:
        """Render audio using current presence flags and parameter settings through DAPN."""
        chain_steps = []
        for e_idx, effect_name in enumerate(SUPPORTED_EFFECTS):
            if self.current_presence[e_idx] > 0.5:
                start_idx, end_idx = PARAM_OFFSETS[effect_name]
                param_names = PARAM_SCHEMA[effect_name]
                eff_params = self.current_params[start_idx:end_idx]

                params_dict = {
                    p_name: float(eff_params[i].item())
                    for i, p_name in enumerate(param_names)
                }
                chain_steps.append({"effect": effect_name, "parameters": params_dict})

        chain_dict = {"chain": chain_steps}
        with torch.no_grad():
            rendered = self.dapn.process(self.dry_audio, chain_dict) if chain_steps else self.dry_audio.clone()
            if not torch.isfinite(rendered).all(): raise RuntimeError("Nonfinite DAPN output")
            self.current_audio = rendered

    def _get_observation(self) -> torch.Tensor:
        """Construct compact state representation for RL agent."""
        with torch.no_grad():
            # Acoustic difference representation: downsampled spectral difference
            n_fft = 512
            win = torch.hann_window(n_fft, device=self.device)
            c_stft = torch.stft(self.current_audio.squeeze(0), n_fft=n_fft, hop_length=256, window=win, return_complex=True).abs()
            t_stft = torch.stft(self.target_audio.squeeze(0), n_fft=n_fft, hop_length=256, window=win, return_complex=True).abs()

            # Mean frequency profile difference: 64 features
            diff = (t_stft - c_stft).mean(dim=-1).squeeze(0)
            if diff.shape[-1] >= 64:
                acoustic_feats = F.adaptive_avg_pool1d(diff.unsqueeze(0).unsqueeze(0), 64).squeeze()
            else:
                acoustic_feats = F.pad(diff, (0, 64 - diff.shape[-1]))

        # Concatenate acoustic embedding + current presence (7) + current params (29) = 100 dims
        obs = torch.cat([acoustic_feats, self.current_presence, self.current_params], dim=0)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def get_current_chain_summary(self) -> dict[str, Any]:
        """Returns clean dictionary summary of the currently configured chain."""
        active_chain = []
        for e_idx, effect_name in enumerate(SUPPORTED_EFFECTS):
            if self.current_presence[e_idx] > 0.5:
                start_idx, end_idx = PARAM_OFFSETS[effect_name]
                param_names = PARAM_SCHEMA[effect_name]
                params_dict = {
                    p_name: round(float(self.current_params[start_idx + i].item()), 4)
                    for i, p_name in enumerate(param_names)
                }
                active_chain.append({"effect": effect_name, "parameters": params_dict})
        return {
            "chain": active_chain,
            "loss": round(self.current_loss, 5),
            "num_effects": len(active_chain),
        }
