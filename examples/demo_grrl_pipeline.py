"""
examples/demo_grrl_pipeline.py
==============================
Interactive End-to-End Demonstration of the Dynamic Graph-Generating
Reinforcement Learning Agent (GRRL-Agent) on Audio Effects Chain Synthesis.

Demonstration Workflow:
1. Loads or generates Dry Audio A and Target Audio B.
2. Measures initial acoustic gap via Multi-Resolution STFT loss.
3. Steps through the GRRL sequential decision loop in real time:
   - Evaluates state and action mask.
   - Predicts next effect and continuous parameters.
   - Renders accumulated chain through DAPN.
   - Shows loss reduction and terminates on STOP.
4. Validates the generated chain schema with FXChainValidator.
5. Saves the final synthesized audio WAV file.
6. Prints a complete synthesis report with physical knob settings.

Usage:
------
# Quick run using built-in synthetic/validation audio:
python3 examples/demo_grrl_pipeline.py

# Run with custom input audio files:
python3 examples/demo_grrl_pipeline.py --dry audio/dry.wav --target audio/wet.wav --output audio/synthesized.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from audio_io import load_audio, save_audio
from common.audio_types import EffectParameterSpec
from common.config_loader import load_effects_config
from common.fx_schema import FXChainValidator
from common.losses import MultiResolutionSTFTLoss
from common.parameter_mapper import ParameterMapper
from dataset.dapn_dataset import PARAM_SCHEMA, SUPPORTED_EFFECTS
from rl.grrl_agent import GRRLAgent
from rl.grrl_env import EFFECT_ACTIONS, GRRLEnv, STOP_ACTION_IDX


def get_physical_mappers() -> dict[str, dict[str, ParameterMapper]]:
    """Loads physical parameter mappers for all registered effects."""
    eff_cfg = load_effects_config()["effects"]
    mappers: dict[str, dict[str, ParameterMapper]] = {}
    for eff_name, eff_info in eff_cfg.items():
        if eff_name == "STOP":
            continue
        mappers[eff_name] = {}
        for param_name, p_info in eff_info.get("parameters", {}).items():
            spec = EffectParameterSpec(
                name=param_name,
                normalized_min=p_info["normalized_range"][0],
                normalized_max=p_info["normalized_range"][1],
                physical_min=p_info["physical_range"][0],
                physical_max=p_info["physical_range"][1],
                unit=p_info.get("unit", ""),
                mapping_type=p_info.get("mapping_type", "linear"),
                required=p_info.get("required", True),
            )
            mappers[eff_name][param_name] = ParameterMapper(spec)
    return mappers


def create_demo_synthetic_pair() -> tuple[torch.Tensor, torch.Tensor, int]:
    """Generates a rich dry harmonic sound and a target processed tone."""
    sr = 44100
    t = torch.linspace(0, 2.0, sr * 2)
    # Dry audio: Clean guitar-like fundamental tone
    dry = (
        0.5 * torch.sin(2 * 3.14159 * 220.0 * t)
        + 0.25 * torch.sin(2 * 3.14159 * 440.0 * t)
    ).unsqueeze(0)

    # Target audio: Harmonic distortion + ambience simulation
    target = (
        0.4 * torch.sin(2 * 3.14159 * 220.0 * t)
        + 0.3 * torch.tanh(1.5 * torch.sin(2 * 3.14159 * 440.0 * t))
        + 0.15 * torch.sin(2 * 3.14159 * 880.0 * t)
    ).unsqueeze(0)
    return dry, target, sr


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end Demonstration of Dynamic GRRL-Agent Synthesis")
    parser.add_argument("--dry", type=str, default=None, help="Path to input Dry Audio A (.wav)")
    parser.add_argument("--target", type=str, default=None, help="Path to Target Reference Audio B (.wav)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/grrl/best_grrl_model.pth", help="GRRL agent checkpoint")
    parser.add_argument("--output", type=str, default="audio/grrl_demo_output.wav", help="Path to save output synthesized WAV")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum sequential chain steps")
    parser.add_argument("--warm-start", action="store_true", help="Enable M2 predictor warm-start prior")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu', 'cuda', 'mps')")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("    DYNAMIC GRAPH-GENERATING RL AGENT (GRRL) — END-TO-END DEMONSTRATION")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. Load Audio Inputs
    # -------------------------------------------------------------
    print("\n[Step 1] Ingesting and Preparing Audio Files...")
    sample_rate = 44100

    if args.dry and args.target and Path(args.dry).exists() and Path(args.target).exists():
        dry_tensor, sample_rate = load_audio(args.dry)
        target_tensor, _ = load_audio(args.target)
        print(f"  • Dry Audio A    : {Path(args.dry).resolve()}")
        print(f"  • Target Audio B : {Path(args.target).resolve()}")
    elif Path("dataset/synthetic/val/audio/sample_000001_dry.wav").exists():
        dry_path = "dataset/synthetic/val/audio/sample_000001_dry.wav"
        target_path = "dataset/synthetic/val/audio/sample_000001_wet.wav"
        dry_tensor, sample_rate = load_audio(dry_path)
        target_tensor, _ = load_audio(target_path)
        print(f"  • Using Held-out Dataset Sample:")
        print(f"    - Dry Audio A   : {dry_path}")
        print(f"    - Target Audio B: {target_path}")
    else:
        dry_tensor, target_tensor, sample_rate = create_demo_synthetic_pair()
        print("  • Using Synthetic Harmonic Test Pair (44.1 kHz, 2.0s mono)")

    # Ensure shape (1, 1, T)
    max_len = sample_rate * 2
    dry_in = dry_tensor[..., :max_len].unsqueeze(0) if dry_tensor.ndim == 2 else dry_tensor[..., :max_len]
    target_in = target_tensor[..., :max_len].unsqueeze(0) if target_tensor.ndim == 2 else target_tensor[..., :max_len]

    # -------------------------------------------------------------
    # 2. Initial Acoustic Distance Gap
    # -------------------------------------------------------------
    loss_fn = MultiResolutionSTFTLoss()
    with torch.no_grad():
        initial_loss = float(loss_fn(dry_in, target_in).item())

    print(f"  • Audio Format   : 44.1 kHz Mono Float32 ({dry_in.shape[-1]} samples, 2.0s)")
    print(f"  • Baseline Gap   : Initial MR-STFT Loss = {initial_loss:.5f}")

    # -------------------------------------------------------------
    # 3. Initialize GRRL Environment & Agent
    # -------------------------------------------------------------
    print("\n[Step 2] Initializing GRRL Sequential Decision Environment & Agent...")
    env = GRRLEnv(
        predictor_checkpoint="checkpoints/best_model.pth" if Path("checkpoints/best_model.pth").exists() else None,
        device=args.device,
        max_chain_length=args.max_steps,
        sample_rate=sample_rate,
        audio_duration_sec=2.0,
    )

    agent = GRRLAgent(
        obs_dim=env.obs_dim,
        num_actions=env.num_actions,
        device=args.device,
    )

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        agent.load(ckpt_path)
        print(f"  • Policy Status  : Loaded trained GRRL checkpoint from {ckpt_path}")
    else:
        print(f"  • Policy Status  : Initialized policy (checkpoint {ckpt_path} not found)")

    # -------------------------------------------------------------
    # 4. Sequential Chain Generation Loop
    # -------------------------------------------------------------
    print("\n[Step 3] Executing Dynamic Sequential Generation Loop (Step-by-Step):")
    print("-" * 80)
    print(f"{'Step':<6} | {'Action Selected':<18} | {'Loss Before':<13} | {'Loss After':<13} | {'Delta':<10} | {'Status'}")
    print("-" * 80)

    obs, info = env.reset(dry_in, target_in, use_warm_start=args.warm_start)
    done = False
    step_num = 0
    t0 = time.time()

    mappers = get_physical_mappers()

    while not done and step_num < args.max_steps:
        step_num += 1
        loss_before = env.current_loss

        # Agent observes state and valid action mask
        mask = env.get_action_mask()
        disc_act, cont_act, log_prob, val = agent.select_action(obs, action_mask=mask, deterministic=True)

        action_name = EFFECT_ACTIONS[disc_act]

        # Execute in environment
        obs, reward, terminated, truncated, step_info = env.step(disc_act, continuous_params=cont_act)
        loss_after = step_info["current_loss"]
        delta = loss_before - loss_after
        status = "TERMINATE" if terminated else ("TRUNCATE" if truncated else "APPEND")

        delta_sign = "+" if delta > 0 else ""
        print(f"{step_num:<6} | {action_name:<18} | {loss_before:<13.5f} | {loss_after:<13.5f} | {delta_sign}{delta:<9.4f} | {status}")

        done = terminated or truncated

    total_time = time.time() - t0
    final_loss = env.current_loss
    improvement_pct = max(0.0, (1.0 - final_loss / (initial_loss + 1e-8)) * 100.0)

    print("-" * 80)
    print(f"Chain Generation Finished in {step_num} steps ({total_time:.3f} seconds).")

    # -------------------------------------------------------------
    # 5. Schema Validation & Audio Rendering
    # -------------------------------------------------------------
    print("\n[Step 4] Validating Synthesized Chain & Saving Audio...")
    summary = env.get_current_chain_summary()
    chain_spec_dict = {"chain": summary["chain"]}

    validator = FXChainValidator()
    validated_spec = validator.validate(chain_spec_dict)
    print(f"  ✓ FXChainValidator Schema Check: PASSED ({len(validated_spec.steps)} effects validated)")

    # Save output audio
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_audio(env.current_audio.squeeze(0), out_path, sample_rate=sample_rate)
    print(f"  ✓ Processed Audio Saved to      : {out_path.resolve()}")

    # -------------------------------------------------------------
    # 6. Final Detailed Synthesis Report
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                        SYNTHESIS SUMMARY REPORT")
    print("=" * 80)
    chain_str = " -> ".join(summary["sequence"]) if summary["sequence"] else "None (Dry Signal)"
    print(f"  • Discovered Chain Sequence : [ {chain_str} ]")
    print(f"  • Total Chain Length        : {summary['num_effects']} effects")
    print(f"  • Initial Baseline Loss     : {initial_loss:.5f}")
    print(f"  • Final Synthesized Loss    : {final_loss:.5f}")
    print(f"  • Acoustic Distance Reduced : {improvement_pct:.2f}%")
    print(f"  • Synthesis Latency         : {total_time:.4f} seconds")
    print("-" * 80)
    print("  PHYSICAL DSP PARAMETERS CONFIGURED BY AGENT:")

    if summary["chain"]:
        for order_idx, step in enumerate(summary["chain"]):
            eff = step["effect"]
            print(f"\n  [{order_idx + 1}] {eff.upper()}:")
            for p_name, norm_val in step["parameters"].items():
                phys_str = ""
                if eff in mappers and p_name in mappers[eff]:
                    mapper = mappers[eff][p_name]
                    phys_val = mapper.to_physical(torch.tensor(norm_val)).item()
                    unit = getattr(mapper._spec, "unit", "")
                    phys_str = f"--> {phys_val:.2f} {unit}".strip()
                print(f"      • {p_name:<16}: {norm_val:>7.4f} (normalized)  {phys_str}")
    else:
        print("  (No effects applied; output is dry audio)")

    print("=" * 80)
    print(f"✓ DEMONSTRATION COMPLETE! Play audio file at: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
