"""
rl/train_ppo.py
===============
Fixed-slot PPO baseline (AudioFXEnv: 7 presence toggles + 29 knob deltas, FIXED schema order).

Same engineering guarantees as rl/train_grrl.py: separate training/evaluation environments,
threshold-crossing evaluation schedule, true-terminal vs time-limit-truncation handling with
V(s_next) bootstrapping for truncations, disjoint train/validation splits, schema-v2 checkpoints
(weights + optimizer + RNG + counters + best score + provenance), fail-loud resume, no silent
synthetic fallback, and output under ``checkpoints_v2/ppo`` (legacy ``checkpoints/`` untouched).

Action distribution note: presence ~ Bernoulli(logits); the knob action is an UNBOUNDED Gaussian
*delta* that is stored and evaluated as sampled; the environment bounds the resulting knob value
(clamp of params, not of actions), so log-probabilities are exact for the stored actions.

Usage:
  python -m rl.train_ppo --total-timesteps 2000 --rollout-steps 64 --predictor-checkpoint checkpoints_v2/best_model.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from common.checkpoint import dataset_provenance
from dataset.dapn_dataset import DAPNSyntheticDataset
from rl.dapn_env import AudioFXEnv
from rl.evaluation import evaluate_ppo, load_pairs, summarize
from rl.ppo_agent import PPOAgent
from rl.rollout_buffer import RolloutBuffer
from rl.train_grrl import LEGACY_DIRS, PairSampler, assert_disjoint, set_seed, synthetic_pair

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
for _noisy in ("dapn", "audio_io", "common"):  # per-render INFO lines drown out training logs
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def build_env(args) -> AudioFXEnv:
    return AudioFXEnv(predictor_checkpoint=args.predictor_checkpoint, device=args.device,
                      max_steps_per_episode=args.max_episode_steps, sample_rate=44100, audio_duration_sec=2.0)


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    out_dir = Path(args.checkpoint_dir)
    if any(l == out_dir.resolve() or l in out_dir.resolve().parents for l in LEGACY_DIRS):
        raise SystemExit(f"Refusing to write into legacy checkpoint tree {out_dir}; use --checkpoint-dir checkpoints_v2/ppo")
    out_dir.mkdir(parents=True, exist_ok=True)
    warm = args.predictor_checkpoint is not None   # warm start is explicit: only with a predictor checkpoint

    train_ds = DAPNSyntheticDataset(args.train_metadata, max_samples=args.max_train_samples) if Path(args.train_metadata).exists() else None
    val_ds = DAPNSyntheticDataset(args.val_metadata, max_samples=args.max_val_samples) if Path(args.val_metadata).exists() else None
    if (train_ds is None or val_ds is None) and not args.allow_synthetic_smoke:
        raise SystemExit("Training/validation metadata missing; pass --allow-synthetic-smoke for a plumbing-only run")
    if train_ds is not None and val_ds is not None:
        assert_disjoint(train_ds, val_ds)
    synthetic = train_ds is None or val_ds is None

    train_env, eval_env = build_env(args), build_env(args)   # independent episode state
    agent = PPOAgent(obs_dim=train_env.obs_dim, discrete_dim=train_env.num_effects, continuous_dim=train_env.num_params, lr=args.lr,
                     clip_eps=args.clip_eps, value_coef=args.value_coef, entropy_coef=args.entropy_coef,
                     max_grad_norm=args.max_grad_norm, device=args.device)
    buffer = RolloutBuffer(args.rollout_steps, train_env.obs_dim, train_env.num_effects, train_env.num_params, args.device)
    sampler = PairSampler(train_ds, args.seed, synthetic)
    val_pairs = (load_pairs(val_ds, range(min(args.eval_episodes, len(val_ds)))) if val_ds is not None
                 else [("synthetic", *synthetic_pair(random.Random(1234 + i))) for i in range(args.eval_episodes)])

    step_count = update_count = episode_count = 0
    best, next_eval = float("inf"), args.eval_freq
    n_term = n_trunc = 0
    extra_cfg = {"warm_start": warm, "max_episode_steps": args.max_episode_steps, "predictor_checkpoint": args.predictor_checkpoint,
                 "synthetic_smoke": synthetic, "gamma": args.gamma, "gae_lambda": args.gae_lambda, "seed": args.seed}
    provenance = {"datasets": dataset_provenance(args.train_metadata, args.val_metadata), "args": vars(args)}

    def save(path, selection):
        agent.save(path, progress=dict(timesteps=step_count, updates=update_count, episodes=episode_count, next_eval=next_eval,
                                       sampler=sampler.state()), selection=selection, provenance=provenance, extra_config=extra_cfg)

    if args.resume:
        ck = agent.load(args.resume, mode="resume")
        pr = ck["progress"]
        step_count, update_count, episode_count, next_eval = pr["timesteps"], pr["updates"], pr["episodes"], pr["next_eval"]
        sampler.load(pr["sampler"]); best = ck["selection"].get("best_val_final_loss", float("inf"))
    elif args.init_from:
        agent.load(args.init_from, mode="eval")

    metrics_path = out_dir / "training_metrics.jsonl"

    def validate(step):
        recs = evaluate_ppo(eval_env, agent, val_pairs, warm_start=warm, deterministic=True)
        summ = summarize(recs)
        for r in recs:
            r.pop("final_audio", None)
        (out_dir / "eval").mkdir(exist_ok=True)
        (out_dir / "eval" / f"eval_step_{step:08d}.json").write_text(json.dumps({"step": step, "summary": summ, "records": recs}, default=float))
        return summ

    def log(entry):
        with open(metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=float) + "\n")

    if not args.resume:
        s0 = validate(0)
        log({"kind": "eval", "step": 0, "synthetic": synthetic, **{k: v for k, v in s0.items() if not isinstance(v, (dict, list))}})

    t_start = time.time()
    _, d, w = sampler.next()
    obs, info = train_env.reset(d, w, use_warm_start=warm)

    while step_count < args.total_timesteps:
        buffer.reset()
        while not buffer.is_full and step_count < args.total_timesteps:
            action_env, disc, cont, logp, val = agent.select_action(obs, deterministic=False)
            next_obs, reward, terminated, truncated, sinfo = train_env.step(action_env)
            done = terminated or truncated
            nv = agent.value(next_obs) if truncated else None
            buffer.add(obs, disc, cont, reward, val, logp, done, truncated=truncated, next_value=nv)
            step_count += 1
            if done:
                episode_count += 1; n_term += int(terminated); n_trunc += int(truncated)
                _, d, w = sampler.next()
                obs, info = train_env.reset(d, w, use_warm_start=warm)
            else:
                obs = next_obs
        last_done = bool(buffer.dones[buffer.ptr - 1].item())
        buffer.compute_returns_and_advantages(0.0 if last_done else agent.value(obs), last_done, args.gamma, args.gae_lambda)
        tm = agent.train_step(buffer, batch_size=args.batch_size, epochs=args.epochs)
        update_count += 1
        elapsed = time.time() - t_start
        entry = {"kind": "train", "step": step_count, "update": update_count, "episodes": episode_count, "terminated": n_term,
                 "truncated": n_trunc, "elapsed_sec": round(elapsed, 1), **tm}
        if step_count >= next_eval or step_count >= args.total_timesteps:
            summ = validate(step_count)
            while next_eval <= step_count:
                next_eval += args.eval_freq
            entry.update({f"eval_{k}": v for k, v in summ.items() if not isinstance(v, (dict, list))})
            score = summ.get("final_loss_mean", float("inf"))
            logger.info("[eval @%d] val final loss %.4f vs start %+.4f", step_count, score, summ.get("vs_start", {}).get("mean_improvement", float("nan")))
            if score < best:
                best = score
                save(out_dir / "best_ppo_model.pth", dict(best_val_final_loss=best, metric="mean val final MR-STFT loss (lower better)", step=step_count))
            save(out_dir / "last_ppo_model.pth", dict(best_val_final_loss=best, step=step_count))
        log(entry)
    save(out_dir / "final_ppo_model.pth", dict(best_val_final_loss=best, final=True))
    return {"steps": step_count, "updates": update_count, "episodes": episode_count, "best_val_final_loss": best}


def build_parser():
    p = argparse.ArgumentParser(description="Train fixed-slot PPO baseline (AudioFXEnv)")
    a = p.add_argument
    a("--total-timesteps", type=int, default=2000); a("--rollout-steps", type=int, default=64)
    a("--epochs", type=int, default=4); a("--batch-size", type=int, default=32); a("--lr", type=float, default=3e-4)
    a("--gamma", type=float, default=0.99); a("--gae-lambda", type=float, default=0.95); a("--clip-eps", type=float, default=0.20)
    a("--value-coef", type=float, default=0.50); a("--entropy-coef", type=float, default=0.01); a("--max-grad-norm", type=float, default=0.50)
    a("--max-episode-steps", type=int, default=10); a("--eval-freq", type=int, default=500); a("--eval-episodes", type=int, default=5)
    a("--seed", type=int, default=42); a("--checkpoint-dir", default="checkpoints_v2/ppo")
    a("--resume", default=None); a("--init-from", default=None)
    a("--predictor-checkpoint", default=None, help="schema-v2 predictor; enables warm start")
    a("--train-metadata", default="dataset/synthetic/train/metadata.jsonl"); a("--val-metadata", default="dataset/synthetic/val/metadata.jsonl")
    a("--max-train-samples", type=int, default=100); a("--max-val-samples", type=int, default=20)
    a("--allow-synthetic-smoke", action="store_true"); a("--device", default="cpu")
    return p


def main(argv=None) -> None:
    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
