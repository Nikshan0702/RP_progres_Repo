"""
rl/process_pair.py
==================
Direct command-line runner to execute the complete RL Audio Processing workflow
on any Dry + Wet audio pair.

Usage
-----
# Basic execution with default output (audio/rl_synthesized_output.wav):
python3 rl/process_pair.py --dry audio/dry.wav --wet audio/wet.wav

# Custom output destination and max steps:
python3 rl/process_pair.py --dry audio/dry.wav --wet audio/wet.wav --output audio/my_result.wav --max-steps 6

# Enable M2 predictor warm-start:
python3 rl/process_pair.py --dry audio/dry.wav --wet audio/wet.wav --warm-start
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.visualization.audio_utils import save_audio_to_path
from rl.visualization.chart_exporter import export_svg_charts
from rl.visualization.workflow_engine import WorkflowEngine


def format_step_row(step_info: dict) -> str:
    """Format a single step diagnostic row for the terminal."""
    step = step_info["step"]
    action = step_info["action"]
    prev_l = step_info["prev_loss"]
    curr_l = step_info["current_loss"]
    delta_l = step_info["delta_loss"]
    reward = step_info["step_reward"]
    imprv = step_info["improvement_pct"]

    delta_sign = "-" if delta_l >= 0 else "+"
    rew_sign = "+" if reward >= 0 else ""

    return (
        f"  Step {step:<2} | {action:<11} | Loss: {prev_l:.4f} -> {curr_l:.4f} "
        f"(Δ {delta_sign}{abs(delta_l):.4f}, {imprv:>5.1f}%) | "
        f"Reward: {rew_sign}{reward:>6.3f}"
    )


def process_audio_pair(
    dry_path: str | Path,
    wet_path: str | Path,
    output_path: str | Path = "audio/rl_synthesized_output.wav",
    mode: str = "real",
    use_warm_start: bool = False,
    max_steps: int = 8,
) -> dict:
    """Run full RL audio processing on dry/wet pair."""
    dry_p = Path(dry_path)
    wet_p = Path(wet_path)
    out_p = Path(output_path)

    if not dry_p.exists():
        raise FileNotFoundError(f"Dry audio file not found: {dry_p}")
    if not wet_p.exists():
        raise FileNotFoundError(f"Wet audio file not found: {wet_p}")

    print("\n" + "=" * 78)
    print(" 🎛️  DAPN RL AUDIO PROCESSING EXECUTION")
    print("=" * 78)
    print(f" Dry Input:       {dry_p}")
    print(f" Wet Reference:   {wet_p}")
    print(f" Output Target:   {out_p}")
    print(f" Mode:            {mode.upper()}")
    print(f" Max Steps:       {max_steps}")
    print(f" M2 Warm-Start:   {use_warm_start}")
    print("-" * 78)

    t0 = time.time()
    engine = WorkflowEngine()

    # Stage 1 & 2: Upload & Preprocessing
    print("[1/3] Loading and preprocessing audio pair...")
    summary = engine.initialize_session(
        dry_path=dry_p,
        wet_path=wet_p,
        mode=mode,
        use_warm_start=use_warm_start,
        max_steps=max_steps,
    )

    init_loss = summary["initial_loss"]
    print(f"      Initial Baseline Loss (MR-STFT): {init_loss:.4f}\n")

    # Step-by-step RL Execution Loop
    print("[2/3] Executing RL Agent Action Loop:")
    step_num = 0
    while not engine.is_terminated and step_num < max_steps:
        step_summary = engine.step_forward()
        step_record = step_summary["steps_history"][-1]
        step_num += 1
        print(format_step_row(step_record))

        # Print detailed physical parameters for non-STOP actions
        if not step_record["is_stop"] and step_record.get("parameters"):
            param_strs = [f"{p['name']}={p['display']}" for p in step_record["parameters"]]
            print(f"         └─ DSP Knobs: {', '.join(param_strs)}")

    # Stage 11: Final audio rendering & saving
    print("\n[3/3] Finalizing audio output and reports...")
    final_summary = engine.get_summary()

    if engine.final_audio_tensor is not None:
        save_audio_to_path(engine.final_audio_tensor, out_p, sample_rate=engine.sample_rate)
        print(f"      ✓ Synthesized WAV saved to: {out_p}")
    else:
        print("      ⚠️ Final audio tensor was empty.")

    # Save JSON report
    json_path = out_p.with_suffix(".json")
    report_data = {
        "dry_input": str(dry_p),
        "wet_reference": str(wet_p),
        "output_audio": str(out_p),
        "initial_loss": final_summary["initial_loss"],
        "final_loss": final_summary["current_loss"],
        "improvement_pct": final_summary["improvement_pct"],
        "cumulative_reward": final_summary["cumulative_reward"],
        "total_steps": final_summary["current_step"],
        "chain": final_summary["current_chain"],
        "runtime_sec": round(time.time() - t0, 3),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"      ✓ Execution report saved to: {json_path}")

    # Save SVG Charts Graph
    svg_path = out_p.with_name(out_p.stem + "_charts.svg")
    try:
        export_svg_charts(final_summary, svg_path)
        print(f"      ✓ Vector graphs (SVG) saved to: {svg_path}")
    except Exception as e:
        print(f"      ⚠️ Failed to export SVG charts: {e}")

    print("-" * 78)
    print(f" Summary: Loss {init_loss:.4f} -> {final_summary['current_loss']:.4f} (+{final_summary['improvement_pct']:.1f}% improvement)")
    print(f" Sequence: {' -> '.join([item['effect'] for item in final_summary['current_chain']]) or 'Empty'}")
    print(f" Total Runtime: {report_data['runtime_sec']}s")
    print("=" * 78 + "\n")

    return report_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute DAPN RL Audio Processing on a Dry + Wet pair")
    parser.add_argument("--dry", type=str, required=True, help="Path to input Dry WAV file")
    parser.add_argument("--wet", "--target", type=str, required=True, dest="wet", help="Path to target Wet reference WAV file")
    parser.add_argument("--output", type=str, default="audio/rl_synthesized_output.wav", help="Destination path for synthesized WAV")
    parser.add_argument("--mode", type=str, choices=["real", "demo"], default="real", help="Execution mode ('real' or 'demo')")
    parser.add_argument("--warm-start", action="store_true", help="Use M2 InverseFXNet predictor warm-start candidate")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum sequential actions before termination")
    args = parser.parse_args()

    process_audio_pair(
        dry_path=args.dry,
        wet_path=args.wet,
        output_path=args.output,
        mode=args.mode,
        use_warm_start=args.warm_start,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
