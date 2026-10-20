"""
rl/visualization/workflow_engine.py
===================================
Coordinates the 11-stage RL audio processing workflow.

Execution modes (never mixed, never silently substituted):
  real : actual GRRLEnv + GRRLAgent checkpoint + corrected DAPN. If the checkpoint/predictor is
         missing or incompatible, ``initialize_session`` raises :class:`RealBackendUnavailable`.
  demo : scripted, SIMULATED numbers for UI inspection only. Every summary/stage/step carries
         ``simulated=True``; no audio is synthesized, so no output WAV is offered.

Baselines: ``dry_loss`` (no effects), ``predictor_loss`` (warm start only: the loss of the
predictor's exact chain) and ``start_loss`` (the RL episode's starting point = predictor loss when
warm-started, else dry loss). ``improvement_pct`` is SIGNED and measured against ``start_loss``;
``dry_improvement_pct`` is measured against the dry loss. Negative values mean degradation.
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
from common.audio_pair import prepare_audio_pair
from common.checkpoint import describe, load_checkpoint
from rl.grrl_agent import GRRLAgent
from rl.grrl_env import BACKTRACK_ACTION_IDX, EFFECT_ACTIONS, GRRLEnv, NUM_EFFECT_ACTIONS, STOP_ACTION_IDX, improvement_percent
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
    extract_action_diagnostics,
    extract_observation_diagnostics,
    map_parameters_with_physical,
)

logger = logging.getLogger(__name__)



class RealBackendUnavailable(RuntimeError):
    """Raised when real execution is requested but its checkpoint/dependency is missing or incompatible."""


class WorkflowStepError(RuntimeError):
    """A step failed (e.g. DSP error). The session state is unchanged by the failed step."""


def json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats with None so every export is valid JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def _r(x: float | None, nd: int = 6) -> float | None:
    return None if x is None else round(float(x), nd)


class WorkflowEngine:
    """Manages stage transitions, step execution, audio states and diagnostic recording.

    Parameters
    ----------
    predictor_ckpt, grrl_ckpt : schema-v2 checkpoints (see common.checkpoint). Legacy files are
        reported as incompatible; they are never loaded.
    """

    def __init__(self, predictor_ckpt=DEFAULT_PREDICTOR_CKPT, grrl_ckpt=DEFAULT_GRRL_CKPT, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.predictor_ckpt = Path(predictor_ckpt) if predictor_ckpt else None
        self.grrl_ckpt = Path(grrl_ckpt) if grrl_ckpt else None
        self.stages = {
            st["id"]: {"id": st["id"], "name": st["name"], "title": st["title"], "description": st["description"],
                       "status": "pending", "message": "Awaiting execution", "timestamp": None, "data": {}}
            for st in WORKFLOW_STAGES
        }
        self.requested_mode = "real"
        self.mode = "real"            # EXECUTION mode of the current session (== requested; never silently changed)
        self.is_initialized = self.is_terminated = False
        self.current_step = 0
        self.max_steps = DEFAULT_MAX_CHAIN_LENGTH
        self.dry_tensor = self.wet_tensor = self.current_audio_tensor = self.final_audio_tensor = None
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.env: GRRLEnv | None = None
        self.agent: GRRLAgent | None = None
        self.obs: torch.Tensor | None = None
        self.info: dict[str, Any] = {}
        self.steps_history, self.loss_history, self.reward_history, self.param_history = [], [], [], []
        self.cumulative_reward = 0.0
        self.dry_loss = self.start_loss = self.current_loss = float("nan")
        self.predictor_loss: float | None = None
        self.baseline_name = "dry"
        self.warm_started = False
        self.deterministic = True
        self.stop_reason: str | None = None
        self.verification: dict[str, Any] | None = None
        self.backend_error: str | None = None
        self.predictor_error: str | None = None
        self.predictor_available = False
        self.agent_config: dict[str, Any] = {}
        self.has_real_agent = self._init_real_backend()

    # ------------------------------------------------------------------
    @property
    def initial_loss(self) -> float:
        """Backward-compatible alias: the DRY loss (no effects)."""
        return self.dry_loss

    def checkpoint_status(self) -> dict[str, Any]:
        return {"predictor": describe(self.predictor_ckpt) if self.predictor_ckpt and self.predictor_ckpt.exists()
                else {"path": str(self.predictor_ckpt), "loadable": False, "error": "file not found"},
                "grrl": describe(self.grrl_ckpt) if self.grrl_ckpt and self.grrl_ckpt.exists()
                else {"path": str(self.grrl_ckpt), "loadable": False, "error": "file not found"}}

    def _init_real_backend(self) -> bool:
        """Build GRRLEnv + load the GRRL checkpoint. Records WHY the backend is unavailable."""
        self.backend_error = self.predictor_error = None
        predictor_path = None
        if self.predictor_ckpt and self.predictor_ckpt.exists():
            predictor_path = str(self.predictor_ckpt)
        else:
            self.predictor_error = f"predictor checkpoint not found: {self.predictor_ckpt}"
        cfg = {}
        if self.grrl_ckpt and self.grrl_ckpt.exists():
            try:   # the environment must be built the way the policy was trained (edit mode, penalties, chain limit)
                cfg = load_checkpoint(self.grrl_ckpt, kind="grrl", mode="eval")["config"]
            except Exception as exc:
                self.backend_error = f"{type(exc).__name__}: {exc}"
                cfg = {}
        env_kw = dict(allow_param_edits=bool(cfg.get("allow_param_edits", False)), edit_range=float(cfg.get("edit_range", 0.25)))
        try:
            try:
                self.env = GRRLEnv(predictor_checkpoint=predictor_path, device=str(self.device), **env_kw,
                                   max_chain_length=DEFAULT_MAX_CHAIN_LENGTH, step_penalty=DEFAULT_STEP_PENALTY,
                                   target_threshold_loss=DEFAULT_TARGET_LOSS_THRESHOLD, sample_rate=DEFAULT_SAMPLE_RATE,
                                   audio_duration_sec=DEFAULT_AUDIO_DURATION)
                self.predictor_available = predictor_path is not None
            except Exception as exc:  # incompatible/legacy predictor: cold-start still possible
                self.predictor_error = f"{type(exc).__name__}: {exc}"
                self.env = GRRLEnv(predictor_checkpoint=None, device=str(self.device), **env_kw, max_chain_length=DEFAULT_MAX_CHAIN_LENGTH,
                                   step_penalty=DEFAULT_STEP_PENALTY, target_threshold_loss=DEFAULT_TARGET_LOSS_THRESHOLD,
                                   sample_rate=DEFAULT_SAMPLE_RATE, audio_duration_sec=DEFAULT_AUDIO_DURATION)
                self.predictor_available = False
            self.agent = GRRLAgent(obs_dim=self.env.obs_dim, num_actions=self.env.num_actions, device=str(self.device))
            if self.backend_error:     # checkpoint was unreadable/incompatible (recorded above)
                return False
            if not (self.grrl_ckpt and self.grrl_ckpt.exists()):
                self.backend_error = f"GRRL checkpoint not found: {self.grrl_ckpt}"
                return False
            ckpt = self.agent.load(self.grrl_ckpt, mode="eval")
            self.agent_config = {k: ckpt["config"].get(k) for k in ("warm_start", "max_chain_length", "max_episode_steps", "synthetic_smoke", "seed", "allow_param_edits", "edit_range")}
            self.agent_config["selection"] = ckpt.get("selection")
            return True
        except Exception as exc:
            self.backend_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Real RL backend unavailable: %s", self.backend_error)
            self.agent = None
            return False

    def reset_session(self) -> None:
        self.is_initialized = self.is_terminated = False
        self.current_step = 0
        self.dry_tensor = self.wet_tensor = self.current_audio_tensor = self.final_audio_tensor = None
        for lst in (self.steps_history, self.loss_history, self.reward_history, self.param_history):
            lst.clear()
        self.cumulative_reward = 0.0
        self.dry_loss = self.start_loss = self.current_loss = float("nan")
        self.predictor_loss = None
        self.stop_reason = None
        self.verification = None
        for st in self.stages.values():
            st.update(status="pending", message="Awaiting execution", timestamp=None, data={})

    def _set_stage_status(self, stage_id: str, status: str, message: str = "", data: dict[str, Any] | None = None) -> None:
        if stage_id in self.stages:
            st = self.stages[stage_id]
            st["status"], st["message"], st["timestamp"] = status, message, round(time.time(), 3)
            if data:
                st["data"].update(data)

    # ------------------------------------------------------------------
    def initialize_session(self, dry_path, wet_path, mode: str = "real", use_warm_start: bool = False,
                           max_steps: int = DEFAULT_MAX_CHAIN_LENGTH, deterministic: bool = True) -> dict[str, Any]:
        """Stages 1-2. ``mode='real'`` NEVER falls back to demo: it raises RealBackendUnavailable."""
        if mode not in ("real", "demo"):
            raise ValueError("mode must be 'real' or 'demo'")
        self.reset_session()
        self.requested_mode = mode
        if mode == "real":
            if not self.has_real_agent:
                raise RealBackendUnavailable(f"Real RL execution requested but backend unavailable: {self.backend_error}")
            if use_warm_start and not self.predictor_available:
                raise RealBackendUnavailable(f"Warm start requested but predictor unavailable: {self.predictor_error}")
        self.mode = mode
        self.warm_started = bool(use_warm_start) and mode == "real"
        self.deterministic = deterministic
        self.max_steps = int(max_steps)

        self._set_stage_status("upload", "running", "Loading audio files...")
        try:
            dry_raw, sr_d = load_audio_from_path(dry_path, target_sample_rate=DEFAULT_SAMPLE_RATE)
            wet_raw, sr_w = load_audio_from_path(wet_path, target_sample_rate=DEFAULT_SAMPLE_RATE)
            self.sample_rate = sr_d
            dm, wm = compute_audio_metrics(dry_raw, sr_d), compute_audio_metrics(wet_raw, sr_w)
            self._set_stage_status("upload", "completed", f"Audio pair loaded: Dry ({dm['duration_sec']}s) & Wet ({wm['duration_sec']}s)",
                                   {"dry_file": Path(dry_path).name, "wet_file": Path(wet_path).name, "dry_metrics": dm, "wet_metrics": wm,
                                    "dry_peaks": compute_waveform_peaks(dry_raw, 120), "wet_peaks": compute_waveform_peaks(wet_raw, 120)})
        except Exception as exc:
            self._set_stage_status("upload", "failed", f"Failed to load audio: {exc}")
            raise

        self._set_stage_status("preprocess", "running", "Standardizing format and duration...")
        try:
            # One shared contract (common.audio_pair): mono, finite, aligned, cropped to the 2 s RL window.
            self.dry_tensor, self.wet_tensor = prepare_audio_pair(dry_raw, wet_raw, sample_rate=self.sample_rate,
                                                                  duration=DEFAULT_AUDIO_DURATION)
            if mode == "real":
                self.env.max_episode_steps = max(self.max_steps, self.env.min_chain_length, 1)
                self.obs, self.info = self.env.reset(self.dry_tensor, self.wet_tensor, use_warm_start=self.warm_started)
                self.dry_loss = float(self.info["dry_loss"])
                self.start_loss = float(self.info["start_loss"])
                self.predictor_loss = self.info["predictor_loss"]
                self.baseline_name = "predictor" if self.warm_started else "dry"
                self.current_loss = self.start_loss
                self.current_audio_tensor = self.env.current_audio.clone()   # the predictor's render when warm-started
                self.dry_tensor, self.wet_tensor = self.env.dry_audio, self.env.target_audio
            else:
                self.dry_loss = self.start_loss = self.current_loss = 7.20   # SIMULATED scripted baseline
                self.baseline_name = "dry"
                self.current_audio_tensor = self.dry_tensor.clone()
                self.obs = None
            first_action = ("Predictor warm-start" if self.warm_started else "Baseline (Dry)") if mode == "real" else "[SIMULATED] Baseline"
            self.loss_history.append({"step": 0, "action": first_action, "loss": _r(self.start_loss), "delta_loss": 0.0,
                                      "improvement_pct": 0.0, "simulated": mode == "demo"})
            self._set_stage_status("preprocess", "completed",
                                   f"Audio standardized: {self.dry_tensor.shape[-1]} samples at {self.sample_rate} Hz (mono). "
                                   f"Dry loss {self.dry_loss:.4f}" + (f"; predictor loss {self.predictor_loss:.4f}" if self.predictor_loss is not None else "")
                                   + (" [SIMULATED]" if mode == "demo" else ""),
                                   {"processed_samples": int(self.dry_tensor.shape[-1]), "sample_rate": self.sample_rate,
                                    "dry_loss": _r(self.dry_loss), "predictor_loss": _r(self.predictor_loss), "start_loss": _r(self.start_loss),
                                    "warm_started": self.warm_started, "simulated": mode == "demo",
                                    "start_chain": copy.deepcopy(self.env.current_chain) if mode == "real" else []})
            self.is_initialized = True
        except Exception as exc:
            self._set_stage_status("preprocess", "failed", f"Preprocessing error: {exc}")
            raise
        return self.get_summary()

    def step_forward(self) -> dict[str, Any]:
        if not self.is_initialized:
            raise RuntimeError("Workflow session is not initialized. Call initialize_session first.")
        if self.is_terminated:
            return self.get_summary()
        return self._step_real() if self.mode == "real" else self._step_demo()

    # ------------------------------------------------------------------
    def _step_real(self) -> dict[str, Any]:
        env, agent = self.env, self.agent
        assert env is not None and agent is not None
        step_no = self.current_step + 1
        prev_loss, prev_chain = self.current_loss, copy.deepcopy(env.current_chain)

        self._set_stage_status("state", "running", f"Extracting state observation vector for step {step_no}...")
        action_mask = env.get_action_mask()
        self._set_stage_status("state", "completed", "State computed: 188 dimensions (Acoustic 64, History 64, Metrics 4, Taste 16, Slot parameters 40)",
                               {"observation": extract_observation_diagnostics(self.obs), "step": step_no})

        self._set_stage_status("action", "running", "Selecting hybrid discrete effect & continuous parameters...")
        with torch.no_grad():
            features = agent.actor_critic.forward_features(self.obs.unsqueeze(0))
            logits = agent.actor_critic.discrete_head(features)[0]
            val_est = float(agent.actor_critic.value_head(features)[0].item())
        disc_act, cont_act, log_prob, _ = agent.select_action(self.obs, action_mask=action_mask, deterministic=self.deterministic)
        action_name = EFFECT_ACTIONS[disc_act]
        is_effect = disc_act < len(SUPPORTED_EFFECTS)
        is_stop = disc_act == STOP_ACTION_IDX
        is_backtrack = disc_act == BACKTRACK_ACTION_IDX
        action_diag = extract_action_diagnostics(logits=logits, action_mask=action_mask, chosen_action_idx=disc_act, value_estimate=val_est)
        self._set_stage_status("action", "completed",
                               f"Selected action: {action_name} (P={action_diag['distribution'][disc_act]['percentage']}%, V={val_est:+.3f}, "
                               f"{'deterministic' if self.deterministic else 'sampled'})", action_diag)

        self._set_stage_status("chain", "running", "Mapping effect parameters to physical DSP specifications...")
        param_records, removed, is_edit = [], None, False
        if is_effect:
            # Show the knob values that will actually be committed (for an in-place edit these are the
            # residual-updated values, not the raw policy action), with deltas vs the previous values.
            cand, is_edit = env.candidate_chain(disc_act, cont_act)
            p_dict = next(it["parameters"] for it in cand if it["effect"] == action_name)
            prev_p = next((it["parameters"] for it in prev_chain if it["effect"] == action_name), None)
            param_records = map_parameters_with_physical(action_name, p_dict, prev_p)
        elif is_backtrack and prev_chain:
            removed = prev_chain[-1]["effect"]
        msg = ((f"FX chain updated: edited {action_name} in place" if is_edit else f"FX chain updated: +{action_name}") + f" ({len(param_records)} parameters mapped)" if is_effect
               else f"BACKTRACK: removing last effect ({removed})" if is_backtrack else "STOP action: finalizing FX chain")
        self._set_stage_status("chain", "completed", msg, {"parameters": param_records, "action": action_name, "removed": removed})

        self._set_stage_status("dsp", "running", "Executing differentiable audio filters in DAPN...")
        self._set_stage_status("loss", "running", "Calculating Multi-Resolution STFT spectral distance...")
        t0 = time.time()
        try:
            next_obs, reward, terminated, truncated, step_info = env.step(disc_act, continuous_params=cont_act)
        except Exception as exc:  # env state is unchanged by a failed step; the session may be retried/inspected
            self._set_stage_status("dsp", "failed", f"DAPN step failed (nothing was committed): {exc}")
            raise WorkflowStepError(str(exc)) from exc
        dsp_ms = (time.time() - t0) * 1000.0
        self.current_step = step_no
        self.obs = next_obs
        self.current_loss = float(step_info["current_loss"])
        self.current_audio_tensor = env.current_audio.clone()
        new_chain = copy.deepcopy(env.current_chain)
        delta = prev_loss - self.current_loss
        impr_start = improvement_percent(self.start_loss, self.current_loss)
        impr_dry = improvement_percent(self.dry_loss, self.current_loss)
        self._set_stage_status("dsp", "completed", f"DAPN processed {len(new_chain)} effects in {dsp_ms:.1f}ms. Waveform finite: True",
                               {"runtime_ms": round(dsp_ms, 2), "waveform_peaks": compute_waveform_peaks(self.current_audio_tensor, 120), "chain_length": len(new_chain)})
        self._set_stage_status("loss", "completed",
                               f"MR-STFT Loss: {self.current_loss:.4f} (Δ={delta:+.4f}; {impr_start:+.1f}% vs {self.baseline_name}, {impr_dry:+.1f}% vs dry)",
                               {"current_loss": _r(self.current_loss), "prev_loss": _r(prev_loss), "delta_loss": _r(delta),
                                "improvement_pct": _r(impr_start, 4), "dry_improvement_pct": _r(impr_dry, 4),
                                "target_threshold": DEFAULT_TARGET_LOSS_THRESHOLD})

        # Reward components come from the environment itself, so they reconcile exactly with `reward`.
        breakdown = {**step_info["reward_breakdown"], "prev_loss": _r(prev_loss), "current_loss": _r(self.current_loss),
                     "delta_loss": _r(delta), "target_achieved": bool(step_info["target_achieved"])}
        self.cumulative_reward += reward
        self._set_stage_status("reward", "completed", f"Step Reward: {reward:+.4f} | Cumulative: {self.cumulative_reward:+.4f}", breakdown)

        done = bool(terminated or truncated)
        if done:
            self.is_terminated = True
            self.stop_reason = ("STOP action chosen" if is_stop else "target loss reached" if step_info["target_achieved"]
                                else f"time limit ({env.max_episode_steps} operations) - truncated, no STOP")
            self._set_stage_status("stop", "completed", f"Episode finished ({self.stop_reason}). Final chain: {len(new_chain)} effects.",
                                   {"stop_reason": self.stop_reason, "chain_length": len(new_chain), "terminated": bool(terminated), "truncated": bool(truncated)})
            self._finalize_output(new_chain)
        else:
            self._set_stage_status("next", "completed", f"Transitioning to step {step_no + 1}. Current chain length: {len(new_chain)}",
                                   {"next_step": step_no + 1, "is_terminal": False})

        self.steps_history.append({
            "step": step_no, "action": action_name, "is_stop": is_stop, "is_backtrack": is_backtrack, "is_edit": is_edit, "removed_effect": removed,
            "discrete_action_idx": disc_act, "parameters": param_records, "log_prob": _r(log_prob),
            "prev_loss": _r(prev_loss), "current_loss": _r(self.current_loss), "delta_loss": _r(delta),
            "step_reward": _r(reward), "cumulative_reward": _r(self.cumulative_reward),
            "improvement_pct": _r(impr_start, 4), "dry_improvement_pct": _r(impr_dry, 4),
            "chain": new_chain, "chain_sequence": [it["effect"] for it in new_chain],
            "terminated": bool(terminated), "truncated": bool(truncated), "action_diagnostics": action_diag,
            "reward_breakdown": breakdown, "simulated": False})
        self.loss_history.append({"step": step_no, "action": action_name, "loss": _r(self.current_loss), "delta_loss": _r(delta),
                                  "improvement_pct": _r(impr_start, 4), "simulated": False})
        self.reward_history.append({"step": step_no, "action": action_name, "reward": _r(reward), "cumulative": _r(self.cumulative_reward),
                                    "breakdown": {k: v for k, v in step_info["reward_breakdown"].items()}})
        return self.get_summary()

    def _step_demo(self) -> dict[str, Any]:
        """SIMULATED scripted progression (UI inspection only). Nothing here is measured."""
        script = [
            {"effect": "Distortion", "params": {"drive": 0.45, "tone": 0.50, "mix": 0.40, "output_gain": 0.35}, "loss": 6.12, "reward": 10.75},
            {"effect": "Reverb", "params": {"room_size": 0.55, "damping": 0.40, "wet_mix": 0.42, "dry_mix": 0.58, "decay": 0.45}, "loss": 4.35, "reward": 17.65},
            {"effect": "Delay", "params": {"delay_time": 0.38, "feedback": 0.40, "wet_mix": 0.35, "dry_mix": 0.65}, "loss": 2.51, "reward": 18.35},
            {"effect": "STOP", "params": {}, "loss": 2.51, "reward": 0.0},
        ]
        step_no = self.current_step + 1
        data = script[min(step_no - 1, len(script) - 1)]
        action_name, is_stop = data["effect"], data["effect"] == "STOP"
        prev_loss = self.current_loss
        self.current_step = step_no
        self.current_loss = data["loss"]
        delta = prev_loss - self.current_loss
        self.cumulative_reward += data["reward"]
        param_records = [] if is_stop else map_parameters_with_physical(action_name, data["params"])
        chain = [{"effect": script[i]["effect"], "parameters": script[i]["params"]} for i in range(min(step_no, len(script))) if script[i]["effect"] != "STOP"]
        tag = "[SIMULATED] "
        self._set_stage_status("state", "completed", tag + "Scripted state placeholder (no observation computed).", {"step": step_no, "simulated": True})
        self._set_stage_status("action", "completed", tag + f"Scripted action: {action_name}", {"simulated": True})
        self._set_stage_status("chain", "completed", tag + f"FX chain (scripted): +{action_name}", {"parameters": param_records, "simulated": True})
        self._set_stage_status("dsp", "completed", tag + "No DSP executed in demo mode.", {"simulated": True})
        self._set_stage_status("loss", "completed", tag + f"Scripted loss: {self.current_loss:.4f}", {"simulated": True})
        self._set_stage_status("reward", "completed", tag + f"Scripted reward: {data['reward']:+.3f}", {"simulated": True})
        done = is_stop or step_no >= len(script)
        if done:
            self.is_terminated = True
            self.stop_reason = "scripted demo finished"
            self._set_stage_status("stop", "completed", tag + "Scripted STOP. No audio was synthesized.", {"simulated": True})
            self._set_stage_status("output", "completed", tag + "Demo mode produces no synthesized audio (nothing to download).", {"simulated": True, "final_chain": chain})
        else:
            self._set_stage_status("next", "completed", tag + f"Proceeding to step {step_no + 1}", {"simulated": True})
        impr = improvement_percent(self.start_loss, self.current_loss)
        self.steps_history.append({"step": step_no, "action": action_name, "is_stop": is_stop, "is_backtrack": False, "discrete_action_idx": EFFECT_ACTIONS.index(action_name),
                                   "parameters": param_records, "prev_loss": _r(prev_loss), "current_loss": _r(self.current_loss), "delta_loss": _r(delta),
                                   "step_reward": data["reward"], "cumulative_reward": _r(self.cumulative_reward), "improvement_pct": _r(impr, 4),
                                   "dry_improvement_pct": _r(impr, 4), "chain": chain, "chain_sequence": [c["effect"] for c in chain], "simulated": True, "is_demo": True})
        self.loss_history.append({"step": step_no, "action": action_name, "loss": _r(self.current_loss), "delta_loss": _r(delta), "improvement_pct": _r(impr, 4), "simulated": True})
        self.reward_history.append({"step": step_no, "action": action_name, "reward": data["reward"], "cumulative": _r(self.cumulative_reward), "simulated": True})
        return self.get_summary()

    # ------------------------------------------------------------------
    def _finalize_output(self, final_chain: list[dict[str, Any]]) -> None:
        """Stage 11 (real mode): final audio IS env.current_audio; verify it against an independent re-render."""
        self._set_stage_status("output", "running", "Verifying and encoding final audio...")
        env = self.env
        self.final_audio_tensor = env.current_audio.clone()
        with torch.no_grad():
            ref = env.dapn.process(env.dry_audio, {"chain": final_chain}) if final_chain else env.dry_audio.clone()
            ref_loss = float(env.loss_fn(ref, env.target_audio))
        diff = float((ref - self.final_audio_tensor).abs().max())
        self.verification = {"rerender_max_abs_diff": diff, "rerender_loss": _r(ref_loss), "reported_loss": _r(self.current_loss),
                             "chain_matches_audio": bool(diff < 1e-5 and abs(ref_loss - self.current_loss) < 1e-5)}
        fm = compute_audio_metrics(self.final_audio_tensor, self.sample_rate)
        impr = improvement_percent(self.start_loss, self.current_loss)
        impr_dry = improvement_percent(self.dry_loss, self.current_loss)
        self._set_stage_status("output", "completed",
                               f"Final audio: {fm['duration_sec']}s, loss {self.current_loss:.4f} ({impr:+.1f}% vs {self.baseline_name}, {impr_dry:+.1f}% vs dry)",
                               {"final_metrics": fm, "final_peaks": compute_waveform_peaks(self.final_audio_tensor, 120), "final_chain": final_chain,
                                "base64_audio": tensor_to_base64_wav(self.final_audio_tensor, self.sample_rate),
                                "total_improvement_pct": _r(impr, 4), "dry_improvement_pct": _r(impr_dry, 4), "verification": self.verification})

    def run_all(self) -> dict[str, Any]:
        if not self.is_initialized:
            raise RuntimeError("Session not initialized.")
        guard = (self.env.max_episode_steps if self.mode == "real" else 8) + 2
        while not self.is_terminated and self.current_step < guard:
            self.step_forward()
        return self.get_summary()

    def get_summary(self) -> dict[str, Any]:
        """JSON-safe snapshot of the session (all non-finite numbers become null)."""
        chain = []
        if self.mode == "real" and self.env is not None and self.is_initialized:
            chain = copy.deepcopy(self.env.current_chain)
        elif self.steps_history:
            chain = copy.deepcopy(self.steps_history[-1].get("chain", []))
        have = self.is_initialized
        impr = improvement_percent(self.start_loss, self.current_loss) if have else 0.0
        impr_dry = improvement_percent(self.dry_loss, self.current_loss) if have else 0.0
        return json_safe({
            "mode": self.mode, "requested_mode": self.requested_mode, "simulated": self.mode == "demo",
            "is_real_backend": self.mode == "real", "has_real_agent": self.has_real_agent, "backend_error": self.backend_error,
            "predictor_available": self.predictor_available, "predictor_error": self.predictor_error, "agent_config": self.agent_config,
            "warm_started": self.warm_started, "deterministic": self.deterministic,
            "is_initialized": self.is_initialized, "is_terminated": self.is_terminated, "stop_reason": self.stop_reason,
            "current_step": self.current_step, "max_steps": self.max_steps,
            "initial_loss": _r(self.dry_loss, 5), "dry_loss": _r(self.dry_loss), "predictor_loss": _r(self.predictor_loss),
            "start_loss": _r(self.start_loss), "baseline_name": self.baseline_name, "baseline_loss": _r(self.start_loss),
            "current_loss": _r(self.current_loss), "final_loss": _r(self.current_loss) if self.is_terminated else None,
            "cumulative_reward": _r(self.cumulative_reward), "improvement_pct": _r(impr, 4), "dry_improvement_pct": _r(impr_dry, 4),
            "improvement_vs_start": _r(self.start_loss - self.current_loss) if have else None,
            "verification": self.verification, "has_output_audio": self.final_audio_tensor is not None,
            "stages": list(self.stages.values()), "steps_history": self.steps_history, "loss_history": self.loss_history,
            "reward_history": self.reward_history, "current_chain": chain, "supported_effects": list(SUPPORTED_EFFECTS), "all_actions": EFFECT_ACTIONS,
            "checkpoints": self.checkpoint_status(),
        })
