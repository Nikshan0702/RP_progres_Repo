"""
rl/evaluate_protocol.py
=======================
Research evaluation protocol: the SAME held-out dry/wet pairs, the SAME loss (multi-resolution STFT,
lower = better), the SAME preprocessing (common.audio_pair, 2 s mono crop), for

  1. dry                 - no processing
  2. predictor_only      - ordered InverseFXNet chain (exactly the GRRL warm-start state)
  3. grrl_cold           - GRRL from an empty chain
  4. grrl_warm           - GRRL starting from the predictor chain
  5. ppo_fixed_slot      - fixed-slot PPO baseline (optional)

Reported per method: n, mean/std/median loss, per-example losses, % improved/degraded/unchanged,
paired bootstrap 95% CI of the mean improvement, runtime, failures, STOP rate / episode length /
truncation rate / reward components, and (vs ground truth) presence, ORDER and parameter metrics of the
final chain. Headline numbers:

    RL improvement (warm)  = predictor_only_loss - grrl_warm_final_loss     (+ = better, - = degradation)
    cold-start improvement = dry_loss           - grrl_cold_final_loss      (reported separately)

The test split must be track-disjoint from training (checked when --train-metadata is given); never use
the checkpoint-selection split (valsel) as test. Nothing is tuned here.

  python -m rl.evaluate_protocol --test-metadata dataset/synthetic_v2/test/metadata.jsonl \
      --train-metadata dataset/synthetic_v2/train/metadata.jsonl --predictor-checkpoint checkpoints_v2/best_model.pth \
      --grrl-warm-checkpoint checkpoints_v2/grrl_warm/best_grrl_model.pth --grrl-cold-checkpoint checkpoints_v2/grrl_cold/best_grrl_model.pth \
      --num-samples 100 --out-dir results/protocol_run1
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from common.checkpoint import dataset_provenance, environment_info, load_checkpoint
from dataset.dapn_dataset import DAPNSyntheticDataset
from models.predictor_metrics import parameter_mae, sequence_metrics
from rl.dapn_env import AudioFXEnv
from rl.evaluation import evaluate_grrl, evaluate_ppo, evaluate_references, load_pairs, paired_summary, summarize
from rl.grrl_agent import GRRLAgent
from rl.grrl_env import GRRLEnv
from rl.ppo_agent import PPOAgent

logging.basicConfig(level=logging.WARNING)


def chain_metrics(records, truths):
    """Presence / order / parameter metrics of each method's FINAL chain vs ground truth."""
    ok = [(r, t) for r, t in zip(records, truths) if not r.get("error")]
    if not ok:
        return {}
    preds = [r["final_sequence"] for r, _ in ok]
    seq = sequence_metrics([p if p else [] for p in preds], [t["chain_sequence"] for _, t in ok])
    par = parameter_mae([r["final_chain"] for r, _ in ok], [t["effects"] for _, t in ok])
    return {**seq, **par}


def method_block(name, records, truths, reference):
    summ = summarize(records)
    block = {"method": name, "summary": summ, "chain_metrics": chain_metrics(records, truths)}
    ok = [r for r in records if not r.get("error")]
    block["runtime_total_sec"] = sum(r["seconds"] for r in records)
    for ref_name, ref in reference.items():
        paired = [(r["final_loss"], ref[r["sample_id"]]) for r in ok if r["sample_id"] in ref]
        if paired:
            block[f"improvement_vs_{ref_name}"] = paired_summary([p[0] for p in paired], [p[1] for p in paired])
    return block


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-metadata", required=True)
    ap.add_argument("--train-metadata", default=None)
    ap.add_argument("--predictor-checkpoint", required=True)
    ap.add_argument("--grrl-warm-checkpoint", default=None)
    ap.add_argument("--grrl-cold-checkpoint", default=None)
    ap.add_argument("--ppo-checkpoint", default=None)
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-chain-length", type=int, default=8)
    ap.add_argument("--max-episode-steps", type=int, default=24)
    ap.add_argument("--stochastic-diagnostic", action="store_true", help="also run stochastic GRRL rollouts (diagnostic only)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    ds = DAPNSyntheticDataset(args.test_metadata)
    if args.train_metadata:
        tr = DAPNSyntheticDataset(args.train_metadata)
        for key in ("dry_path", "wet_path", "source_track"):
            if {r[key] for r in tr.records} & {r[key] for r in ds.records}:
                raise SystemExit(f"Test/train overlap on '{key}': refusing to evaluate (leakage)")
    idx = list(range(len(ds))); random.Random(args.seed).shuffle(idx)
    idx = idx[: args.num_samples]
    pairs = load_pairs(ds, idx)
    truths = {ds.records[i]["sample_id"]: ds.records[i] for i in idx}
    ordered_truths = [truths[sid] for sid, _, _ in pairs]
    print(f"Evaluating on {len(pairs)} held-out pairs from {args.test_metadata}")

    kw = dict(device=args.device, max_chain_length=args.max_chain_length, max_episode_steps=args.max_episode_steps,
              sample_rate=44100, audio_duration_sec=2.0)
    methods, per_method = {}, {}

    ref_env = GRRLEnv(predictor_checkpoint=args.predictor_checkpoint, **kw)
    dry_recs = evaluate_references(ref_env, pairs, predictor=False)
    pred_recs = evaluate_references(ref_env, pairs, predictor=True)
    per_method["dry"], per_method["predictor_only"] = dry_recs, pred_recs
    dry_loss = {r["sample_id"]: r["final_loss"] for r in dry_recs if not r["error"]}
    pred_loss = {r["sample_id"]: r["final_loss"] for r in pred_recs if not r["error"]}
    for r in dry_recs:
        r.setdefault("final_chain", []); r.setdefault("final_sequence", [])

    def run_grrl(name, ckpt, warm):
        cfg = load_checkpoint(ckpt, kind="grrl", mode="eval")["config"]       # env must match how the policy was trained
        env = GRRLEnv(predictor_checkpoint=args.predictor_checkpoint if warm else None, failure_mode="raise",
                      allow_param_edits=cfg.get("allow_param_edits", False), edit_range=cfg.get("edit_range", 0.25), **kw)
        agent = GRRLAgent(obs_dim=env.obs_dim, num_actions=env.num_actions, device=args.device)
        ck = agent.load(ckpt, mode="eval")
        recs = evaluate_grrl(env, agent, pairs, warm_start=warm, deterministic=True)
        per_method[name] = recs
        blk = method_block(name, recs, ordered_truths, {"dry": dry_loss, "predictor": pred_loss})
        blk["checkpoint"], blk["checkpoint_selection"] = str(ckpt), ck.get("selection")
        blk["checkpoint_training_progress"] = ck.get("progress", {}).get("timesteps")
        methods[name] = blk
        if args.stochastic_diagnostic:
            srecs = evaluate_grrl(env, agent, pairs, warm_start=warm, deterministic=False)
            methods[name]["stochastic_diagnostic"] = summarize(srecs)

    if args.grrl_cold_checkpoint:
        run_grrl("grrl_cold", args.grrl_cold_checkpoint, warm=False)
    if args.grrl_warm_checkpoint:
        run_grrl("grrl_warm", args.grrl_warm_checkpoint, warm=True)
    if args.ppo_checkpoint:
        penv = AudioFXEnv(predictor_checkpoint=args.predictor_checkpoint, device=args.device, max_steps_per_episode=10,
                          sample_rate=44100, audio_duration_sec=2.0)
        pagent = PPOAgent(obs_dim=penv.obs_dim, discrete_dim=penv.num_effects, continuous_dim=penv.num_params, device=args.device)
        ck = pagent.load(args.ppo_checkpoint, mode="eval")
        warm = bool(ck["config"].get("warm_start"))
        recs = evaluate_ppo(penv, pagent, pairs, warm_start=warm, deterministic=True)
        per_method["ppo_fixed_slot"] = recs
        blk = method_block("ppo_fixed_slot", recs, ordered_truths, {"dry": dry_loss, "predictor": pred_loss})
        blk["warm_start"], blk["checkpoint"], blk["note"] = warm, str(args.ppo_checkpoint), "fixed schema order; no order prediction"
        methods["ppo_fixed_slot"] = blk

    methods["dry"] = method_block("dry", dry_recs, ordered_truths, {})
    methods["predictor_only"] = method_block("predictor_only", pred_recs, ordered_truths, {"dry": dry_loss})

    headline = {}
    if "grrl_warm" in methods and "improvement_vs_predictor" in methods["grrl_warm"]:
        headline["rl_improvement_over_predictor_warm"] = methods["grrl_warm"]["improvement_vs_predictor"]
    if "grrl_cold" in methods:
        headline["cold_start_improvement_over_dry"] = methods["grrl_cold"].get("improvement_vs_dry")
    report = {"protocol": "same pairs / same MR-STFT loss / same preprocessing for every method; lower loss is better; improvement = reference - final",
              "test_metadata": args.test_metadata, "n_examples": len(pairs), "seed": args.seed,
              "provenance": {"datasets": dataset_provenance(args.test_metadata, args.train_metadata), "environment": environment_info(),
                             "predictor_checkpoint": args.predictor_checkpoint},
              "headline": headline, "methods": methods}
    (out / "protocol_report.json").write_text(json.dumps(report, indent=2, default=float))

    ids = [sid for sid, _, _ in pairs]
    with open(out / "per_example.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        names = list(per_method)
        w.writerow(["sample_id", "true_sequence"] + [f"{n}_loss" for n in names] + [f"{n}_sequence" for n in names] + [f"{n}_error" for n in names])
        by = {n: {r["sample_id"]: r for r in recs} for n, recs in per_method.items()}
        for sid in ids:
            w.writerow([sid, "->".join(truths[sid]["chain_sequence"])]
                       + [by[n].get(sid, {}).get("final_loss", "") for n in names]
                       + ["->".join(by[n].get(sid, {}).get("final_sequence", [])) for n in names]
                       + [by[n].get(sid, {}).get("error") or "" for n in names])

    lines = [f"# Held-out evaluation (n={len(pairs)}, {args.test_metadata})", "", "| method | mean loss | std | median | vs dry: mean Δ [95% CI] | improved/degraded % | vs predictor: mean Δ [95% CI] | improved/degraded % | stop% | steps | failed |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    def cell(b):
        return (f"{b['mean_improvement']:+.4f} [{b['ci95_low']:+.4f},{b['ci95_high']:+.4f}]", f"{b['pct_improved']:.0f}/{b['pct_degraded']:.0f}") if b else ("-", "-")
    for n in ["dry", "predictor_only", "grrl_cold", "grrl_warm", "ppo_fixed_slot"]:
        if n not in methods:
            continue
        m = methods[n]; s = m["summary"]
        d, p = cell(m.get("improvement_vs_dry")), cell(m.get("improvement_vs_predictor"))
        lines.append(f"| {n} | {s.get('final_loss_mean', float('nan')):.4f} | {s.get('final_loss_std', float('nan')):.4f} | {s.get('final_loss_median', float('nan')):.4f} | {d[0]} | {d[1]} | {p[0]} | {p[1]} | "
                     f"{100 * s.get('stop_rate', float('nan')):.0f} | {s.get('mean_steps', float('nan')):.1f} | {s['n_failed']} |")
    lines += ["", "Positive Δ = lower loss than the reference (better); negative = degradation.", "",
              "| method | presence F1 | exact set | exact sequence | pairwise order acc | order given correct set | param MAE |", "|---|---|---|---|---|---|---|"]
    for n, m in methods.items():
        c = m["chain_metrics"]
        if c:
            lines.append(f"| {n} | {c['presence_f1']:.3f} | {c['presence_set_exact']:.3f} | {c['order_exact_sequence']:.3f} | {c['order_pairwise_acc']:.3f} | {c['order_given_correct_set']:.3f} | {c.get('param_mae', float('nan')):.3f} |")
    (out / "protocol_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return report


if __name__ == "__main__":
    main()
