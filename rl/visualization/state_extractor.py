"""
rl/visualization/state_extractor.py
===================================
Diagnostic data extraction and parameter mapping for RL audio synthesis visualization.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from dataset.dapn_dataset import SUPPORTED_EFFECTS
from rl.grrl_env import EFFECT_ACTIONS, NUM_EFFECT_ACTIONS, STOP_ACTION_IDX
from rl.visualization.config import get_effects_spec_map


def map_parameters_with_physical(
    effect_name: str,
    param_dict: dict[str, float],
    prev_param_dict: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Map normalized parameter values [0, 1] to physical DSP values with units.

    Parameters
    ----------
    effect_name : str
        Name of the effect (e.g. 'EQ', 'Distortion', 'Reverb').
    param_dict : dict[str, float]
        Dictionary of normalized parameters {param_name: normalized_val}.
    prev_param_dict : dict[str, float], optional
        Previous step parameters if modified, for delta computation.

    Returns
    -------
    list of dict
        Detailed parameter records including normalized, physical, unit, and deltas.
    """
    if effect_name == "STOP":
        return []

    effects_spec = get_effects_spec_map()
    eff_spec = effects_spec.get(effect_name, {})
    param_specs = eff_spec.get("parameters", {})

    mapped_list = []
    for p_name, norm_val in param_dict.items():
        spec = param_specs.get(p_name, {})
        p_min, p_max = spec.get("physical_range", [0.0, 1.0])
        unit = spec.get("unit", "")

        norm_clamped = max(0.0, min(1.0, float(norm_val)))
        # Linear physical DSP mapping
        phys_val = p_min + norm_clamped * (p_max - p_min)

        # Previous values
        prev_norm = None
        prev_phys = None
        delta_norm = 0.0
        if prev_param_dict and p_name in prev_param_dict:
            prev_norm = max(0.0, min(1.0, float(prev_param_dict[p_name])))
            prev_phys = p_min + prev_norm * (p_max - p_min)
            delta_norm = norm_clamped - prev_norm

        # Format display string based on unit
        if unit == "Hz":
            display_str = f"{phys_val:,.1f} Hz"
        elif unit == "dB":
            display_str = f"{phys_val:+.1f} dB"
        elif unit == "ms":
            display_str = f"{phys_val:.1f} ms"
        elif unit == "s":
            display_str = f"{phys_val:.2f} s"
        elif unit == "dimensionless":
            if p_name == "ratio":
                display_str = f"{phys_val:.1f}:1"
            else:
                display_str = f"{phys_val:.2f}"
        else:
            display_str = f"{phys_val:.2f} {unit}".strip()

        mapped_list.append({
            "name": p_name,
            "normalized": round(norm_clamped, 4),
            "physical": round(phys_val, 4),
            "unit": unit,
            "display": display_str,
            "min_physical": p_min,
            "max_physical": p_max,
            "prev_normalized": round(prev_norm, 4) if prev_norm is not None else None,
            "prev_physical": round(prev_phys, 4) if prev_phys is not None else None,
            "delta_normalized": round(delta_norm, 4),
        })

    return mapped_list


def extract_observation_diagnostics(obs_tensor: torch.Tensor | None) -> dict[str, Any]:
    """Break down the 188-dimensional GRRL observation vector into readable sub-components.

    Observation Layout (188 dims, matches GRRLEnv._get_observation):
      - [0:64]    Acoustic distance embedding (spectral difference bins)
      - [64:128]  Ordered chain history (8 slots * 8 classes one-hot)
      - [128:132] Step progress, chain occupancy, normalized loss, improvement ratio vs start
      - [132:148] PTA Taste conditioning vector (16 dims)
      - [148:188] Explicit parameter values of the 8 history slots (8 x 5, normalized)
    """
    if obs_tensor is None:
        return {
            "total_dim": 0,
            "acoustic_profile": [],
            "chain_history_slots": [],
            "metrics": {},
            "taste_vector": [],
        }

    flat = obs_tensor.detach().cpu().float().flatten()
    total_dim = flat.shape[0]

    # 1. Acoustic distance profile (first 64 features)
    acoustic_slice = flat[:64].tolist() if total_dim >= 64 else flat.tolist()
    acoustic_profile = [round(v, 4) for v in acoustic_slice]

    # 2. Ordered chain history (8 slots * 8 classes = 64 features)
    history_slots = []
    classes = ["EMPTY"] + list(SUPPORTED_EFFECTS)
    if total_dim >= 128:
        hist_slice = flat[64:128].view(8, 8)
        for slot_idx in range(8):
            row = hist_slice[slot_idx]
            max_class_idx = int(torch.argmax(row).item())
            class_name = classes[max_class_idx] if max_class_idx < len(classes) else "UNKNOWN"
            confidence = float(row[max_class_idx].item())
            history_slots.append({
                "slot": slot_idx + 1,
                "effect": class_name,
                "confidence": round(confidence, 2),
                "is_empty": (class_name == "EMPTY"),
            })

    # 3. Scalar metrics (4 features: step_norm, chain_norm, loss_norm, improvement_ratio)
    metrics = {}
    if total_dim >= 132:
        m_slice = flat[128:132].tolist()
        metrics = {
            "step_progress_pct": round(m_slice[0] * 100.0, 1),
            "chain_occupancy_pct": round(m_slice[1] * 100.0, 1),
            "normalized_loss": round(m_slice[2], 4),
            "improvement_ratio": round(m_slice[3], 4),
        }

    # 4. PTA taste vector (16 dims)
    taste_vector = []
    if total_dim >= 148:
        taste_slice = flat[132:148].tolist()
        taste_vector = [round(v, 3) for v in taste_slice]

    slot_params = []
    if total_dim >= 188:
        pm = flat[148:188].view(8, 5)
        slot_params = [[round(float(v), 4) for v in row] for row in pm]

    return {
        "total_dim": total_dim,
        "acoustic_profile": acoustic_profile,
        "chain_history_slots": history_slots,
        "metrics": metrics,
        "taste_vector": taste_vector,
        "slot_parameters": slot_params,
    }


def extract_action_diagnostics(
    logits: torch.Tensor | None,
    action_mask: torch.Tensor | None,
    chosen_action_idx: int,
    value_estimate: float = 0.0,
) -> dict[str, Any]:
    """Format discrete action probabilities, action mask, and policy diagnostics."""
    action_items = []
    actions_list = EFFECT_ACTIONS  # 7 effects + STOP + BACKTRACK

    if logits is not None and logits.numel() >= NUM_EFFECT_ACTIONS:
        flat_logits = logits.detach().cpu().float().flatten()[:NUM_EFFECT_ACTIONS]
        flat_mask = action_mask.detach().cpu().bool().flatten() if action_mask is not None else torch.ones(NUM_EFFECT_ACTIONS, dtype=torch.bool)

        # Compute softmax over masked logits
        masked_l = flat_logits.clone()
        masked_l[~flat_mask] = -1e9
        probs = F.softmax(masked_l, dim=-1).tolist()
        raw_logits = flat_logits.tolist()
        mask_bools = flat_mask.tolist()
    else:
        probs = [1.0 / NUM_EFFECT_ACTIONS] * NUM_EFFECT_ACTIONS
        raw_logits = [0.0] * NUM_EFFECT_ACTIONS
        mask_bools = [True] * NUM_EFFECT_ACTIONS

    for idx, act_name in enumerate(actions_list):
        prob = probs[idx] if idx < len(probs) else 0.0
        logit = raw_logits[idx] if idx < len(raw_logits) else 0.0
        is_valid = mask_bools[idx] if idx < len(mask_bools) else True
        is_selected = (idx == chosen_action_idx)

        # Explanatory reason if masked
        mask_reason = ""
        if not is_valid:
            if act_name == "STOP":
                mask_reason = "Minimum chain length not yet met"
            elif act_name == "BACKTRACK":
                mask_reason = "Nothing to remove, or too few steps left to restore the minimum chain"
            else:
                mask_reason = "Effect already used or max length reached"

        action_items.append({
            "index": idx,
            "action": act_name,
            "probability": round(prob, 4),
            "percentage": round(prob * 100.0, 1),
            "logit": round(logit, 3),
            "is_valid": is_valid,
            "is_selected": is_selected,
            "mask_reason": mask_reason,
            "is_stop": (idx == STOP_ACTION_IDX),
            "is_backtrack": (act_name == "BACKTRACK"),
        })

    chosen_name = actions_list[chosen_action_idx] if 0 <= chosen_action_idx < len(actions_list) else "UNKNOWN"

    return {
        "chosen_action_idx": chosen_action_idx,
        "chosen_action": chosen_name,
        "is_stop": (chosen_action_idx == STOP_ACTION_IDX),
        "value_estimate": round(value_estimate, 4),
        "distribution": action_items,
    }


def compute_reward_breakdown(
    prev_loss: float,
    current_loss: float,
    initial_loss: float | None = None,
    is_terminal: bool = False,
    target_achieved: bool = False,
    step_penalty: float = 0.05,
    action_was_masked: bool = False,
    is_stop_action: bool = False,
    chain_length: int = 0,
    *,
    length_penalty: float = 0.0,
    chain_length_before: int | None = None,
    started_above_target: bool = True,
) -> dict[str, Any]:
    """Reward components with the SAME formula and signs as ``GRRLEnv.step`` (lower loss is better):

      loss_improvement_reward = +10 * (prev_loss - current_loss)
      step_penalty            = -step_penalty                      (0 for STOP)
      length_penalty          = -length_penalty * (len_after - len_before)
      terminal_bonus          = +3 if the target was reached and the episode started above target
      total_step_reward       = sum of the four components

    The live dashboard uses ``info['reward_breakdown']`` from the environment directly; this helper
    exists for offline reconstruction and is tested to agree with the environment.
    ``initial_loss``/``is_terminal``/``action_was_masked`` are accepted for backward compatibility
    and do not affect the result (masked actions are rejected by the environment, not penalised).
    """
    delta_loss = prev_loss - current_loss
    before = chain_length if chain_length_before is None else chain_length_before
    components = {
        "loss_improvement_reward": 10.0 * delta_loss,
        "step_penalty": 0.0 if is_stop_action else -step_penalty,
        "length_penalty": -length_penalty * (chain_length - before),
        "terminal_bonus": 3.0 if (target_achieved and started_above_target) else 0.0,
    }
    total = sum(components.values())
    return {
        "prev_loss": round(prev_loss, 6),
        "current_loss": round(current_loss, 6),
        "delta_loss": round(delta_loss, 6),
        **{k: round(v, 6) for k, v in components.items()},
        "total_step_reward": round(total, 6),
        "target_achieved": target_achieved,
    }
