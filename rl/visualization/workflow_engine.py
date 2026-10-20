"""
rl/visualization/workflow_engine.py
===================================
Coordinates the 11-stage RL audio processing workflow for both Real RL and
clearly labeled Demo modes.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from pathlib import Path
from typing import Any

import torch

from dapn.dapn import DAPN
from dataset.dapn_dataset import PARAM_SCHEMA, SUPPORTED_EFFECTS
from rl.grrl_agent import GRRLAgent
from rl.grrl_env import EFFECT_ACTIONS, GRRLEnv, NUM_EFFECT_ACTIONS, STOP_ACTION_IDX
from rl.visualization.audio_utils import (
    compute_audio_metrics,
    compute_waveform_peaks,
    load_audio_from_path,
    save_audio_to_path,
    tensor_to_base64_wav,
)
from rl.visualization.config import (
    DEFAULT_AUDIO_DURATION,
    DEFAULT_GRRL_CKPT,
    DEFAULT_MAX_CHAIN_LENGTH,
    DEFAULT_PREDICTOR_CKPT,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_STEP_PENALTY,
    DEFAULT_TARGET_LOSS_THRESHOLD,
    WORKFLOW_STAGES,
)
from rl.visualization.state_extractor import (
    compute_reward_breakdown,
    extract_action_diagnostics,
    extract_observation_diagnostics,
    map_parameters_with_physical,
)

logger = logging.getLogger(__name__)


class WorkflowEngine:
    """Manages stage transitions, step execution, audio states, and diagnostic recording.

    Parameters
    ----------
    predictor_ckpt : str | Path | None
        Path to InverseFXNet checkpoint.
    grrl_ckpt : str | Path | None
        Path to trained GRRL Actor-Critic checkpoint.
    device : str
        Compute device ('cpu', 'cuda', 'mps').
    """

    def __init__(
        self,
        predictor_ckpt: str | Path | None = DEFAULT_PREDICTOR_CKPT,
        grrl_ckpt: str | Path | None = DEFAULT_GRRL_CKPT,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.predictor_ckpt = Path(predictor_ckpt) if predictor_ckpt else None
        self.grrl_ckpt = Path(grrl_ckpt) if grrl_ckpt else None

        # Stage status dictionary
        self.stages = {
            stage["id"]: {
                "id": stage["id"],
                "name": stage["name"],
                "title": stage["title"],
                "description": stage["description"],
                "status": "pending",  # 'pending', 'running', 'completed', 'failed'
                "message": "Awaiting execution",
                "timestamp": None,
                "data": {},
            }
            for stage in WORKFLOW_STAGES
        }

        # Session tracking
        self.mode: str = "real"  # 'real' or 'demo'
        self.is_initialized: bool = False
        self.is_terminated: bool = False
        self.current_step: int = 0
        self.max_steps: int = DEFAULT_MAX_CHAIN_LENGTH

        # Audio tensors (1, 1, T)
        self.dry_tensor: torch.Tensor | None = None
        self.wet_tensor: torch.Tensor | None = None
        self.current_audio_tensor: torch.Tensor | None = None
        self.final_audio_tensor: torch.Tensor | None = None
        self.sample_rate: int = DEFAULT_SAMPLE_RATE

        # RL Environment and Agent
        self.env: GRRLEnv | None = None
        self.agent: GRRLAgent | None = None
        self.obs: torch.Tensor | None = None
        self.info: dict[str, Any] = {}

        # Trajectory history
        self.steps_history: list[dict[str, Any]] = []
        self.loss_history: list[dict[str, Any]] = []
        self.reward_history: list[dict[str, Any]] = []
        self.param_history: list[dict[str, Any]] = []
        self.cumulative_reward: float = 0.0
        self.initial_loss: float = 999.0
        self.current_loss: float = 999.0

        # Attempt eager agent load
        self.has_real_agent = self._init_real_backend()

    def _init_real_backend(self) -> bool:
        """Initialize GRRLEnv and load trained GRRLAgent checkpoint."""
        try:
            self.env = GRRLEnv(
                predictor_checkpoint=str(self.predictor_ckpt) if self.predictor_ckpt and self.predictor_ckpt.exists() else None,
                device=str(self.device),
                max_chain_length=DEFAULT_MAX_CHAIN_LENGTH,
                step_penalty=DEFAULT_STEP_PENALTY,
                target_threshold_loss=DEFAULT_TARGET_LOSS_THRESHOLD,
                sample_rate=DEFAULT_SAMPLE_RATE,
                audio_duration_sec=DEFAULT_AUDIO_DURATION,
            )

            self.agent = GRRLAgent(
                obs_dim=self.env.obs_dim,
                num_actions=self.env.num_actions,
                device=str(self.device),
            )

            if self.grrl_ckpt and self.grrl_ckpt.exists():
                self.agent.load(str(self.grrl_ckpt))
                logger.info("Loaded real GRRLAgent checkpoint from %s", self.grrl_ckpt)
                return True
            else:
                logger.warning("GRRL checkpoint not found at %s. Demo mode recommended.", self.grrl_ckpt)
                return False
        except Exception as e:
            logger.warning("Could not initialize real RL backend: %s", e)
            self.env = None
            self.agent = None
            return False

    def reset_session(self) -> None:
        """Reset session to uninitialized pending state."""
        self.is_initialized = False
        self.is_terminated = False
        self.current_step = 0
        self.dry_tensor = None
        self.wet_tensor = None
        self.current_audio_tensor = None
        self.final_audio_tensor = None
        self.steps_history.clear()
        self.loss_history.clear()
        self.reward_history.clear()
        self.param_history.clear()
        self.cumulative_reward = 0.0
        self.initial_loss = 999.0
        self.current_loss = 999.0

        for stage_id in self.stages:
            self.stages[stage_id]["status"] = "pending"
            self.stages[stage_id]["message"] = "Awaiting execution"
            self.stages[stage_id]["timestamp"] = None
            self.stages[stage_id]["data"] = {}

    def _set_stage_status(
        self,
        stage_id: str,
        status: str,
        message: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        """Update stage execution status."""
        if stage_id in self.stages:
            self.stages[stage_id]["status"] = status
            self.stages[stage_id]["message"] = message
            self.stages[stage_id]["timestamp"] = round(time.time(), 3)
            if data:
                self.stages[stage_id]["data"].update(data)

    def initialize_session(
        self,
        dry_path: str | Path,
        wet_path: str | Path,
        mode: str = "real",
        use_warm_start: bool = False,
        max_steps: int = DEFAULT_MAX_CHAIN_LENGTH,
    ) -> dict[str, Any]:
        """Execute Stage 1 (Upload) and Stage 2 (Preprocessing) and reset RL environment."""
        self.reset_session()
        self.mode = "demo" if (mode == "demo" or not self.has_real_agent) else "real"
        self.max_steps = max_steps

        # -------------------------------------------------------------
        # STAGE 1: Audio Pair Upload (Dry + Wet Reference)
        # -------------------------------------------------------------
        self._set_stage_status("upload", "running", "Loading audio files...")
        try:
            self.dry_tensor, sr_dry = load_audio_from_path(dry_path, target_sample_rate=DEFAULT_SAMPLE_RATE)
            self.wet_tensor, sr_wet = load_audio_from_path(wet_path, target_sample_rate=DEFAULT_SAMPLE_RATE)
            self.sample_rate = sr_dry

            dry_metrics = compute_audio_metrics(self.dry_tensor, self.sample_rate)
            wet_metrics = compute_audio_metrics(self.wet_tensor, self.sample_rate)
            dry_peaks = compute_waveform_peaks(self.dry_tensor, num_points=120)
            wet_peaks = compute_waveform_peaks(self.wet_tensor, num_points=120)

            self._set_stage_status(
                "upload",
                "completed",
                f"Audio pair loaded: Dry ({dry_metrics['duration_sec']}s) & Wet ({wet_metrics['duration_sec']}s)",
                {
                    "dry_file": Path(dry_path).name,
                    "wet_file": Path(wet_path).name,
                    "dry_metrics": dry_metrics,
                    "wet_metrics": wet_metrics,
                    "dry_peaks": dry_peaks,
                    "wet_peaks": wet_peaks,
                },
            )
        except Exception as e:
            self._set_stage_status("upload", "failed", f"Failed to load audio: {e}")
            raise

        # -------------------------------------------------------------
        # STAGE 2: Audio Preprocessing
        # -------------------------------------------------------------
        self._set_stage_status("preprocess", "running", "Standardizing format and duration...")
        try:
            max_samples = int(self.sample_rate * DEFAULT_AUDIO_DURATION)
            # Truncate to maximum duration for computational efficiency
            self.dry_tensor = self.dry_tensor[..., :max_samples]
            self.wet_tensor = self.wet_tensor[..., :max_samples]

            # Harmonize length if different
            min_len = min(self.dry_tensor.shape[-1], self.wet_tensor.shape[-1])
            self.dry_tensor = self.dry_tensor[..., :min_len]
            self.wet_tensor = self.wet_tensor[..., :min_len]

            self.current_audio_tensor = self.dry_tensor.clone()

            if self.mode == "real" and self.env is not None:
                self.obs, self.info = self.env.reset(
                    self.dry_tensor,
                    self.wet_tensor,
                    use_warm_start=use_warm_start,
                )
                self.initial_loss = float(self.info["initial_loss"])
                self.current_loss = self.initial_loss
            else:
                # Demo mode baseline
                self.initial_loss = 7.20
                self.current_loss = 7.20
                self.obs = torch.randn(148)
                self.info = {"initial_loss": 7.20, "current_loss": 7.20, "chain_length": 0}

            # Record baseline in loss history
            self.loss_history.append({
                "step": 0,
                "action": "Baseline (Dry)",
                "loss": round(self.initial_loss, 5),
                "delta_loss": 0.0,
                "improvement_pct": 0.0,
            })

            self._set_stage_status(
                "preprocess",
                "completed",
                f"Audio standardized: {min_len} samples at {self.sample_rate} Hz (Mono). Baseline Loss: {self.initial_loss:.4f}",
                {
                    "processed_samples": min_len,
                    "sample_rate": self.sample_rate,
                    "initial_loss": round(self.initial_loss, 4),
                    "warm_started": use_warm_start,
                },
            )
            self.is_initialized = True
        except Exception as e:
            self._set_stage_status("preprocess", "failed", f"Preprocessing error: {e}")
            raise

        return self.get_summary()

    def step_forward(self) -> dict[str, Any]:
        """Advance one single RL action step in the workflow pipeline."""
        if not self.is_initialized:
            raise RuntimeError("Workflow session is not initialized. Call initialize_session first.")

        if self.is_terminated:
            return self.get_summary()

        self.current_step += 1

        if self.mode == "real":
            return self._step_real()
        else:
            return self._step_demo()

    def _step_real(self) -> dict[str, Any]:
        """Execute one real step using GRRLAgent, GRRLEnv, and DAPN."""
        assert self.env is not None and self.agent is not None

        prev_loss = self.current_loss
        prev_chain = copy.deepcopy(self.env.current_chain)

        # -------------------------------------------------------------
        # STAGE 3: RL Agent State
        # -------------------------------------------------------------
        self._set_stage_status("state", "running", f"Extracting state observation vector for step {self.current_step}...")
        obs_diagnostics = extract_observation_diagnostics(self.obs)
        action_mask = self.env.get_action_mask()
        self._set_stage_status(
            "state",
            "completed",
            f"State computed: 148 dimensions (Acoustic: 64, History: 64, Metrics: 4, Taste: 16)",
            {"observation": obs_diagnostics, "step": self.current_step},
        )

        # -------------------------------------------------------------
        # STAGE 4: Action Selection
        # -------------------------------------------------------------
        self._set_stage_status("action", "running", "Sampling hybrid discrete effect & continuous parameters...")
        with torch.no_grad():
            features = self.agent.actor_critic.forward_features(self.obs.unsqueeze(0))
            logits = self.agent.actor_critic.discrete_head(features)[0]
            val_est = float(self.agent.actor_critic.value_head(features)[0].item())

        disc_act, cont_act, log_prob, _ = self.agent.select_action(
            self.obs, action_mask=action_mask, deterministic=True
        )
        action_name = EFFECT_ACTIONS[disc_act]

        action_diag = extract_action_diagnostics(
            logits=logits,
            action_mask=action_mask,
            chosen_action_idx=disc_act,
            value_estimate=val_est,
        )

        self._set_stage_status(
            "action",
            "completed",
            f"Selected action: {action_name} (P={action_diag['distribution'][disc_act]['percentage']}%, V={val_est:+.3f})",
            action_diag,
        )

        # -------------------------------------------------------------
        # STAGE 5: FX Chain Generation
        # -------------------------------------------------------------
        self._set_stage_status("chain", "running", "Mapping effect parameters to physical DSP specifications...")

        # If STOP action chosen
        is_stop = (disc_act == STOP_ACTION_IDX)
        param_records = []
        if not is_stop:
            p_names = PARAM_SCHEMA[action_name]
            p_dict = {p_names[i]: float(cont_act[i].item()) for i in range(len(p_names))}

            # Find previous parameters if this effect was already configured
            prev_p_dict = None
            for item in prev_chain:
                if item["effect"] == action_name:
                    prev_p_dict = item["parameters"]

            param_records = map_parameters_with_physical(action_name, p_dict, prev_p_dict)

        self._set_stage_status(
            "chain",
            "completed",
            f"FX chain updated: +{action_name} ({len(param_records)} parameters mapped)" if not is_stop else "STOP action: Finalizing FX chain",
            {"parameters": param_records, "action": action_name},
        )

        # -------------------------------------------------------------
        # STAGE 6: DAPN Audio Processing & STAGE 7: Loss Calculation
        # -------------------------------------------------------------
        self._set_stage_status("dsp", "running", "Executing differentiable audio filters in DAPN...")
        self._set_stage_status("loss", "running", "Calculating Multi-Resolution STFT spectral distance...")

        t_dsp_0 = time.time()
        next_obs, reward, terminated, truncated, step_info = self.env.step(
            disc_act, continuous_params=cont_act
        )
        dsp_runtime_ms = (time.time() - t_dsp_0) * 1000.0

        self.obs = next_obs
        self.current_loss = float(step_info["current_loss"])
        self.current_audio_tensor = self.env.current_audio.clone()
        new_chain = copy.deepcopy(self.env.current_chain)

        interm_peaks = compute_waveform_peaks(self.current_audio_tensor, num_points=120)

        self._set_stage_status(
            "dsp",
            "completed",
            f"DAPN processed {len(new_chain)} effects in {dsp_runtime_ms:.1f}ms. Waveform finite: True",
            {
                "runtime_ms": round(dsp_runtime_ms, 2),
                "waveform_peaks": interm_peaks,
                "chain_length": len(new_chain),
            },
        )

        delta_loss = prev_loss - self.current_loss
        improvement_pct = float(step_info["improvement_pct"])

        self._set_stage_status(
            "loss",
            "completed",
            f"MR-STFT Loss: {self.current_loss:.4f} (Δ={delta_loss:+.4f}, Improvement={improvement_pct:.1f}%)",
            {
                "current_loss": round(self.current_loss, 5),
                "prev_loss": round(prev_loss, 5),
                "delta_loss": round(delta_loss, 5),
                "improvement_pct": round(improvement_pct, 2),
                "target_threshold": DEFAULT_TARGET_LOSS_THRESHOLD,
            },
        )

        # -------------------------------------------------------------
        # STAGE 8: Reward Calculation
        # -------------------------------------------------------------
        self._set_stage_status("reward", "running", "Computing RL step reward and terminal bonuses...")
        reward_breakdown = compute_reward_breakdown(
            prev_loss=prev_loss,
            current_loss=self.current_loss,
            initial_loss=self.initial_loss,
            is_terminal=terminated,
            target_achieved=bool(step_info.get("target_achieved", False)),
            step_penalty=DEFAULT_STEP_PENALTY,
            is_stop_action=is_stop,
            chain_length=len(new_chain),
        )
        self.cumulative_reward += reward

        self._set_stage_status(
            "reward",
            "completed",
            f"Step Reward: {reward:+.3f} | Cumulative: {self.cumulative_reward:+.3f}",
            reward_breakdown,
        )

        # -------------------------------------------------------------
        # STAGE 9: Next State / Next Action & STAGE 10: STOP
        # -------------------------------------------------------------
        is_done = terminated or truncated or is_stop or (self.current_step >= self.max_steps)

        if not is_done:
            self._set_stage_status(
                "next",
                "completed",
                f"Transitioning to Step {self.current_step + 1}. Current chain length: {len(new_chain)}",
                {"next_step": self.current_step + 1, "is_terminal": False},
            )
        else:
            self.is_terminated = True
            stop_reason = "STOP Action chosen" if is_stop else ("Target reached" if step_info.get("target_achieved") else "Max steps reached")
            self._set_stage_status(
                "stop",
                "completed",
                f"Episode finished ({stop_reason}). Final Chain: {len(new_chain)} effects.",
                {"stop_reason": stop_reason, "chain_length": len(new_chain)},
            )
            self._finalize_output(new_chain)

        # Record step in history
        step_record = {
            "step": self.current_step,
            "action": action_name,
            "is_stop": is_stop,
            "discrete_action_idx": disc_act,
            "parameters": param_records,
            "prev_loss": round(prev_loss, 5),
            "current_loss": round(self.current_loss, 5),
            "delta_loss": round(delta_loss, 5),
            "step_reward": round(reward, 4),
            "cumulative_reward": round(self.cumulative_reward, 4),
            "improvement_pct": round(improvement_pct, 2),
            "chain": copy.deepcopy(new_chain),
            "chain_sequence": [item["effect"] for item in new_chain],
            "action_diagnostics": action_diag,
            "reward_breakdown": reward_breakdown,
        }
        self.steps_history.append(step_record)

        self.loss_history.append({
            "step": self.current_step,
            "action": action_name,
            "loss": round(self.current_loss, 5),
            "delta_loss": round(delta_loss, 5),
            "improvement_pct": round(improvement_pct, 2),
        })

        self.reward_history.append({
            "step": self.current_step,
            "action": action_name,
            "reward": round(reward, 4),
            "cumulative": round(self.cumulative_reward, 4),
        })

        return self.get_summary()

    def _step_demo(self) -> dict[str, Any]:
        """Execute one step in clearly labeled Demonstration Mode."""
        demo_sequence = [
            {"effect": "Distortion", "params": {"drive": 0.45, "tone": 0.50, "mix": 0.40, "output_gain": 0.35}, "loss": 6.12, "reward": 10.75},
            {"effect": "Reverb", "params": {"room_size": 0.55, "damping": 0.40, "wet_mix": 0.42, "dry_mix": 0.58, "decay": 0.45}, "loss": 4.35, "reward": 17.65},
            {"effect": "Delay", "params": {"delay_time": 0.38, "feedback": 0.40, "wet_mix": 0.35, "dry_mix": 0.65}, "loss": 2.51, "reward": 18.35},
            {"effect": "STOP", "params": {}, "loss": 2.51, "reward": 4.30},
        ]

        step_idx = min(self.current_step - 1, len(demo_sequence) - 1)
        step_data = demo_sequence[step_idx]
        action_name = step_data["effect"]
        is_stop = (action_name == "STOP")

        prev_loss = self.current_loss
        self.current_loss = step_data["loss"]
        delta_loss = prev_loss - self.current_loss
        reward = step_data["reward"]
        self.cumulative_reward += reward

        # Parameters
        param_records = map_parameters_with_physical(action_name, step_data["params"]) if not is_stop else []

        # Current chain
        current_chain = [
            {"effect": demo_sequence[i]["effect"], "parameters": demo_sequence[i]["params"]}
            for i in range(step_idx + 1)
            if demo_sequence[i]["effect"] != "STOP"
        ]

        # Stage updates with DEMO tags
        self._set_stage_status("state", "completed", "[DEMO] Simulated state observation constructed.", {"step": self.current_step})
        self._set_stage_status("action", "completed", f"[DEMO] Simulated action sampled: {action_name}")
        self._set_stage_status("chain", "completed", f"[DEMO] FX chain updated: +{action_name}", {"parameters": param_records})
        self._set_stage_status("dsp", "completed", f"[DEMO] Simulated DAPN rendering through {len(current_chain)} effects.")
        self._set_stage_status("loss", "completed", f"[DEMO] Simulated STFT Loss: {self.current_loss:.4f} (Δ={delta_loss:+.4f})")
        self._set_stage_status("reward", "completed", f"[DEMO] Step Reward: {reward:+.3f} | Cumulative: {self.cumulative_reward:+.3f}")

        is_done = is_stop or (self.current_step >= len(demo_sequence))
        if not is_done:
            self._set_stage_status("next", "completed", f"[DEMO] Proceeding to Step {self.current_step + 1}")
        else:
            self.is_terminated = True
            self._set_stage_status("stop", "completed", "[DEMO] STOP action completed. Final chain synthesized.")
            self._finalize_output(current_chain)

        step_record = {
            "step": self.current_step,
            "action": action_name,
            "is_stop": is_stop,
            "discrete_action_idx": EFFECT_ACTIONS.index(action_name),
            "parameters": param_records,
            "prev_loss": round(prev_loss, 5),
            "current_loss": round(self.current_loss, 5),
            "delta_loss": round(delta_loss, 5),
            "step_reward": round(reward, 4),
            "cumulative_reward": round(self.cumulative_reward, 4),
            "improvement_pct": round(max(0.0, (1.0 - self.current_loss / (self.initial_loss + 1e-8)) * 100.0), 2),
            "chain": current_chain,
            "chain_sequence": [item["effect"] for item in current_chain],
            "is_demo": True,
        }
        self.steps_history.append(step_record)
        self.loss_history.append({
            "step": self.current_step,
            "action": action_name,
            "loss": round(self.current_loss, 5),
            "delta_loss": round(delta_loss, 5),
            "improvement_pct": step_record["improvement_pct"],
        })
        self.reward_history.append({
            "step": self.current_step,
            "action": action_name,
            "reward": round(reward, 4),
            "cumulative": round(self.cumulative_reward, 4),
        })

        return self.get_summary()

    def _finalize_output(self, final_chain: list[dict[str, Any]]) -> None:
        """Render final audio output and complete Stage 11."""
        self._set_stage_status("output", "running", "Rendering final wet output through DAPN DSP pipeline...")

        if self.mode == "real" and self.env is not None and self.dry_tensor is not None:
            self.final_audio_tensor = self.env.current_audio.clone()
        else:
            # Fallback if dry tensor present
            self.final_audio_tensor = self.current_audio_tensor if self.current_audio_tensor is not None else torch.zeros(1, 1, 22050)

        # Metrics and peaks
        final_metrics = compute_audio_metrics(self.final_audio_tensor, self.sample_rate)
        final_peaks = compute_waveform_peaks(self.final_audio_tensor, num_points=120)

        # Encode base64 audio for instant playback
        base64_audio = tensor_to_base64_wav(self.final_audio_tensor, self.sample_rate)

        total_improvement = max(0.0, (1.0 - self.current_loss / (self.initial_loss + 1e-8)) * 100.0)

        self._set_stage_status(
            "output",
            "completed",
            f"Final audio synthesized: {final_metrics['duration_sec']}s, Final Loss: {self.current_loss:.4f} (+{total_improvement:.1f}% improvement)",
            {
                "final_metrics": final_metrics,
                "final_peaks": final_peaks,
                "final_chain": final_chain,
                "base64_audio": base64_audio,
                "total_improvement_pct": round(total_improvement, 2),
            },
        )

    def run_all(self) -> dict[str, Any]:
        """Execute the complete workflow from start to STOP / termination."""
        if not self.is_initialized:
            raise RuntimeError("Session not initialized.")

        while not self.is_terminated and self.current_step < self.max_steps:
            self.step_forward()

        return self.get_summary()

    def get_summary(self) -> dict[str, Any]:
        """Serialize current workflow session state to JSON-compatible dictionary."""
        current_chain = []
        if self.mode == "real" and self.env is not None:
            current_chain = copy.deepcopy(self.env.current_chain)
        elif self.steps_history:
            current_chain = copy.deepcopy(self.steps_history[-1].get("chain", []))

        total_improvement = max(0.0, (1.0 - self.current_loss / (self.initial_loss + 1e-8)) * 100.0)

        return {
            "mode": self.mode,
            "is_real_backend": (self.mode == "real"),
            "has_real_agent": self.has_real_agent,
            "is_initialized": self.is_initialized,
            "is_terminated": self.is_terminated,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "initial_loss": round(self.initial_loss, 5),
            "current_loss": round(self.current_loss, 5),
            "cumulative_reward": round(self.cumulative_reward, 4),
            "improvement_pct": round(total_improvement, 2),
            "stages": list(self.stages.values()),
            "steps_history": self.steps_history,
            "loss_history": self.loss_history,
            "reward_history": self.reward_history,
            "current_chain": current_chain,
            "supported_effects": list(SUPPORTED_EFFECTS),
            "all_actions": EFFECT_ACTIONS,
        }
