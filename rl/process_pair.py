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

# Enable predictor warm-start (RL result is compared against the predictor-only chain):
python3 rl/process_pair.py --dry audio/dry.wav --wet audio/wet.wav --warm-start

Real mode fails with a non-zero exit if the GRRL/predictor checkpoints are missing or incompatible;
it never falls back to demo. ``--mode demo`` is scripted/simulated and writes no audio.
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
from rl.visualization.workflow_engine import RealBackendUnavailable, WorkflowEngine


def format_step_row(step_info: dict) -> str:
    """Format a single step diagnostic row for the terminal."""
    step = step_info["step"]
    action = step_info["action"]
    prev_l = step_info["prev_loss"]
    curr_l = step_info["current_loss"]
    delta_l = step_info["delta_loss"]
    reward = step_info["step_reward"]
    imprv = step_info["improvement_pct"]

    delta_sign = "-" if delta_l >= 0 else "+"  # '-' = loss went DOWN (better)
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
    predictor_ckpt: str | None = None,
    grrl_ckpt: str | None = None,
    step_images: bool = True,
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
    print(f" Mode:            {mode.upper()}{' (SIMULATED - no audio is synthesized)' if mode == 'demo' else ''}")
    print(f" Max Steps:       {max_steps}")
    print(f" M2 Warm-Start:   {use_warm_start}")
    print("-" * 78)

    t0 = time.time()
    kw = {k: v for k, v in (('predictor_ckpt', predictor_ckpt), ('grrl_ckpt', grrl_ckpt)) if v}
    engine = WorkflowEngine(**kw)

    # Stage 1 & 2: Upload & Preprocessing
    print("[1/3] Loading and preprocessing audio pair...")
    summary = engine.initialize_session(
        dry_path=dry_p,
        wet_path=wet_p,
        mode=mode,
        use_warm_start=use_warm_start,
        max_steps=max_steps,
    )

    init_loss = summary["start_loss"]
    print(f"      Dry loss (MR-STFT)       : {summary['dry_loss']:.4f}")
    if summary["predictor_loss"] is not None:
        print(f"      Predictor-only loss      : {summary['predictor_loss']:.4f}  <- RL baseline")
    print(f"      RL starting loss         : {init_loss:.4f} ({summary['baseline_name']})\n")

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
    fs = engine.get_summary()
    if mode == "real" and engine.final_audio_tensor is not None:
        save_audio_to_path(engine.final_audio_tensor, out_p, sample_rate=engine.sample_rate)
        print(f"      ✓ Synthesized WAV saved to: {out_p}")
    elif mode == "real":
        print("      ✗ Episode did not finish (no final audio).")
    else:
        print("      (demo mode: no audio synthesized)")

    json_path = out_p.with_suffix(".json")
    report_data = {
        "execution_mode": fs["mode"], "requested_mode": fs["requested_mode"], "simulated": fs["simulated"],
        "warm_started": fs["warm_started"], "dry_input": str(dry_p), "wet_reference": str(wet_p),
        "output_audio": str(out_p) if engine.final_audio_tensor is not None else None,
        "dry_loss": fs["dry_loss"], "predictor_loss": fs["predictor_loss"], "start_loss": fs["start_loss"],
        "baseline": fs["baseline_name"], "final_loss": fs["current_loss"],
        "improvement_vs_start_pct": fs["improvement_pct"], "improvement_vs_dry_pct": fs["dry_improvement_pct"],
        "cumulative_reward": fs["cumulative_reward"], "total_steps": fs["current_step"], "stop_reason": fs["stop_reason"],
        "chain": fs["current_chain"], "verification": fs["verification"], "full_length": fs["full_length"],
        "steps": [{k: st.get(k) for k in ("step", "action", "prev_loss", "current_loss", "step_reward", "reward_breakdown", "parameters", "terminated", "truncated")}
                  for st in fs["steps_history"]],
        "runtime_sec": round(time.time() - t0, 3),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, allow_nan=False)
    print(f"      ✓ Execution report saved to: {json_path}")

    if mode == "real" and step_images:
        from rl.visualization.step_images import export_step_pngs
        img_dir = out_p.with_name(out_p.stem + "_steps")
        pngs = export_step_pngs(engine, img_dir)
        print(f"      ✓ {len(pngs)} step PNGs saved to: {img_dir}/")

    svg_path = out_p.with_name(out_p.stem + "_charts.svg")
    try:
        export_svg_charts(fs, svg_path)
        print(f"      ✓ Vector graphs (SVG) saved to: {svg_path}")
    except Exception as e:
        print(f"      ⚠️ Failed to export SVG charts: {e}")

    print("-" * 78)
    print(f" Summary [{fs['mode'].upper()}]: {fs['baseline_name']} loss {fs['start_loss']:.4f} -> {fs['current_loss']:.4f} "
          f"({fs['improvement_pct']:+.1f}% vs {fs['baseline_name']}; {fs['dry_improvement_pct']:+.1f}% vs dry; negative = worse)")
    if fs["full_length"]:
        fl = fs["full_length"]
        print(f" Full-length output: {fl['output_seconds']}s rendered (chain chosen on first {fl['decision_seconds']}s); "
              f"full loss {fl['full_final_loss']:.4f} vs dry {fl['full_dry_loss']:.4f} ({fl['full_improvement_pct']:+.1f}%)")
    print(f" Sequence: {' -> '.join([item['effect'] for item in fs['current_chain']]) or 'Empty'}")
    print(f" Total Runtime: {report_data['runtime_sec']}s")
    print("=" * 78 + "\n")

    return report_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute DAPN RL Audio Processing on a Dry + Wet pair")
    parser.add_argument("--dry", type=str, required=True, help="Path to input Dry WAV file")
    parser.add_argument("--wet", "--target", type=str, required=True, dest="wet", help="Path to target Wet reference WAV file")
    parser.add_argument("--output", type=str, default="audio/rl_synthesized_output.wav", help="Destination path for synthesized WAV")
    parser.add_argument("--mode", type=str, choices=["real", "demo"], default="real", help="Execution mode ('real' or 'demo')")
    parser.add_argument("--warm-start", action="store_true", help="Start from the predictor's chain; reports RL change vs predictor-only")
    parser.add_argument("--no-step-images", action="store_true", help="skip the per-step PNG export")
    parser.add_argument("--predictor-ckpt", default=None); parser.add_argument("--grrl-ckpt", default=None)
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum sequential actions before termination")
    args = parser.parse_args()

    try:
        run(args)
    except RealBackendUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)


def run(args) -> dict:
    return process_audio_pair(
        dry_path=args.dry,
        wet_path=args.wet,
        output_path=args.output,
        mode=args.mode,
        use_warm_start=args.warm_start,
        max_steps=args.max_steps,
        predictor_ckpt=args.predictor_ckpt,
        grrl_ckpt=args.grrl_ckpt,
        step_images=not args.no_step_images,
    )


if __name__ == "__main__":
    main()
