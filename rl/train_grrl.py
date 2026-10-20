"""
rl/train_grrl.py
================
PPO + GAE training of the GRRL agent (sequential FX-chain editing MDP).

Design guarantees (each covered by tests/test_training_isolation.py):
  * Training and evaluation use TWO independent GRRLEnv instances. Evaluation never touches the
    training env, its observation, the rollout buffer, episode counters or the global RNG.
  * Evaluation is scheduled by threshold crossing (``step >= next_eval``), so intervals are honoured
    even when rollout boundaries never land on an exact multiple.
  * True terminal states (STOP / target reached) and time-limit truncation are tracked separately.
    Truncated transitions bootstrap with V(s_next) in GAE; terminals bootstrap with 0.
  * Validation pairs are never added to the rollout buffer. The training and validation splits must
    be disjoint on audio path AND source track.
  * Checkpoints (schema v2) store weights, optimizer, RNG, counters, sampler state, config, best
    validation score and dataset/dependency provenance. ``--resume`` fails if the file is missing;
    ``--init-from`` loads weights only (evaluation-style load).
  * Default output is ``checkpoints_v2/grrl``; the legacy ``checkpoints/`` tree is never overwritten.
  * No silent synthetic-data fallback: a missing dataset is an error unless
    ``--allow-synthetic-smoke`` is passed (then every metric is flagged ``synthetic``).

Usage:
  python -m rl.train_grrl --total-timesteps 2000 --rollout-steps 64 --use-warm-start \
      --predictor-checkpoint checkpoints_v2/best_model.pth
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

import numpy as np
import torch

from common.checkpoint import dataset_provenance
from dataset.dapn_dataset import DAPNSyntheticDataset
from rl.evaluation import evaluate_grrl, load_pairs, summarize
from rl.grrl_agent import GRRLAgent
from rl.grrl_buffer import GRRLRolloutBuffer
from rl.grrl_env import GRRLEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
for _noisy in ("dapn", "audio_io", "common"):  # per-render INFO lines drown out training logs
    logging.getLogger(_noisy).setLevel(logging.WARNING)
LEGACY_DIRS = [(ROOT / "checkpoints").resolve()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthetic_pair(rng: random.Random):
    """Smoke-test only (``--allow-synthetic-smoke``)."""
    sr = 44100
    t = torch.arange(int(sr * 1.5)) / sr
    f = rng.choice([220.0, 330.0, 440.0, 550.0])
    dry = torch.sin(2 * torch.pi * f * t)[None]
    return dry, torch.tanh(2.5 * dry)


class PairSampler:
    """Seeded epoch-shuffled sampler over a dataset; its state is checkpointed."""

    def __init__(self, dataset: DAPNSyntheticDataset | None, seed: int, synthetic: bool = False):
        self.dataset, self.synthetic = dataset, synthetic
        self.rng = random.Random(seed)
        self.order: list[int] = []
        self.pos = 0

    def next(self):
        if self.dataset is None:
            d, w = synthetic_pair(self.rng)
            return "synthetic", d, w
        if self.pos >= len(self.order):
            self.order = list(range(len(self.dataset)))
            self.rng.shuffle(self.order)
            self.pos = 0
        i = self.order[self.pos]
        self.pos += 1
        s = self.dataset[i]
        return s["sample_id"], s["dry_audio"], s["wet_audio"]

    def state(self):
        return {"rng": self.rng.getstate(), "order": self.order, "pos": self.pos}

    def load(self, st):
        self.rng.setstate(st["rng"]); self.order, self.pos = st["order"], st["pos"]


def assert_disjoint(train: DAPNSyntheticDataset, val: DAPNSyntheticDataset) -> None:
    for key in ("dry_path", "wet_path", "source_track"):
        a, b = {r[key] for r in train.records}, {r[key] for r in val.records}
        if a & b:
            raise ValueError(f"Train/validation overlap on '{key}' ({len(a & b)} shared): refusing to train")


def build_env(args, **kw) -> GRRLEnv:
    return GRRLEnv(predictor_checkpoint=args.predictor_checkpoint if (args.use_warm_start or args.predictor_checkpoint) else None,
                   device=args.device, max_chain_length=args.max_chain_length, step_penalty=args.step_penalty,
                   max_episode_steps=args.max_episode_steps, sample_rate=44100, audio_duration_sec=2.0,
                   allow_param_edits=args.allow_param_edits, edit_range=args.edit_range, **kw)


def run_validation(eval_env, agent, val_pairs, args, step: int, out_dir: Path) -> dict:
    recs = evaluate_grrl(eval_env, agent, val_pairs, warm_start=args.use_warm_start, deterministic=True)
    summ = summarize(recs)
    for r in recs:
        r.pop("final_audio", None)
    (out_dir / "eval").mkdir(parents=True, exist_ok=True)
    (out_dir / "eval" / f"eval_step_{step:08d}.json").write_text(json.dumps({"step": step, "summary": summ, "records": recs}, default=float))
    return summ


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    out_dir = Path(args.checkpoint_dir)
    if any(l == out_dir.resolve() or l in out_dir.resolve().parents for l in LEGACY_DIRS):
        raise SystemExit(f"Refusing to write into legacy checkpoint tree {out_dir}; use --checkpoint-dir checkpoints_v2/grrl")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.use_warm_start and not args.predictor_checkpoint:
        raise SystemExit("--use-warm-start requires --predictor-checkpoint (a schema-v2 ordered predictor)")

    train_ds = DAPNSyntheticDataset(args.train_metadata, max_samples=args.max_train_samples) if Path(args.train_metadata).exists() else None
    val_ds = DAPNSyntheticDataset(args.val_metadata, max_samples=args.max_val_samples) if Path(args.val_metadata).exists() else None
    if (train_ds is None or val_ds is None) and not args.allow_synthetic_smoke:
        raise SystemExit("Training/validation metadata missing; pass --allow-synthetic-smoke for a plumbing-only run")
    if train_ds is not None and val_ds is not None:
        assert_disjoint(train_ds, val_ds)
    synthetic = train_ds is None or val_ds is None

    train_env = build_env(args, failure_mode="penalty")   # training: failures become explicit penalties
    eval_env = build_env(args, failure_mode="raise")      # evaluation: failures are recorded, never penalised silently
    agent = GRRLAgent(obs_dim=train_env.obs_dim, num_actions=train_env.num_actions, lr=args.lr, clip_eps=args.clip_eps,
                      value_coef=args.value_coef, entropy_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                      device=args.device)
    buffer = GRRLRolloutBuffer(args.rollout_steps, train_env.obs_dim, train_env.num_actions, train_env.max_params_per_effect, args.device)
    sampler = PairSampler(train_ds, args.seed, synthetic)
    val_pairs = (load_pairs(val_ds, range(min(args.eval_episodes, len(val_ds)))) if val_ds is not None
                 else [("synthetic", *synthetic_pair(random.Random(1234 + i))) for i in range(args.eval_episodes)])

    step_count = episode_count = update_count = 0
    best = float("inf")
    next_eval = args.eval_freq
    n_term = n_trunc = n_stop = n_fail = 0
    extra_cfg = {"warm_start": args.use_warm_start, "max_chain_length": args.max_chain_length,
                 "max_episode_steps": args.max_episode_steps, "step_penalty": args.step_penalty,
                 "allow_param_edits": args.allow_param_edits, "edit_range": args.edit_range,
                 "predictor_checkpoint": args.predictor_checkpoint, "synthetic_smoke": synthetic,
                 "gamma": args.gamma, "gae_lambda": args.gae_lambda, "seed": args.seed}
    provenance = {"datasets": dataset_provenance(args.train_metadata, args.val_metadata), "args": vars(args)}

    def save(path, selection):
        agent.save(path, progress=dict(timesteps=step_count, updates=update_count, episodes=episode_count,
                                        next_eval=next_eval, sampler=sampler.state()),
                   selection=selection, provenance=provenance, extra_config=extra_cfg)

    if args.resume:
        ck = agent.load(args.resume, mode="resume")   # FileNotFoundError if missing; Incompatible* if wrong arch
        for k in ("allow_param_edits", "edit_range", "warm_start"):
            want = {"allow_param_edits": args.allow_param_edits, "edit_range": args.edit_range, "warm_start": args.use_warm_start}[k]
            if ck["config"].get(k, False if k != "edit_range" else 0.25) != want:
                raise SystemExit(f"Cannot resume: checkpoint config {k}={ck['config'].get(k)!r} differs from requested {want!r}")
        pr = ck["progress"]
        step_count, update_count, episode_count, next_eval = pr["timesteps"], pr["updates"], pr["episodes"], pr["next_eval"]
        sampler.load(pr["sampler"])
        best = ck["selection"].get("best_val_final_loss", float("inf"))
        logger.info("Resumed from %s at step %d (update %d, best val loss %.4f)", args.resume, step_count, update_count, best)
    elif args.init_from:
        agent.load(args.init_from, mode="eval")
        logger.info("Initialised weights (no optimizer/RNG) from %s", args.init_from)

    metrics_path = out_dir / "grrl_training_metrics.jsonl"

    def log(entry):
        with open(metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=float) + "\n")

    if not args.resume:
        s0 = run_validation(eval_env, agent, val_pairs, args, 0, out_dir)
        log({"kind": "eval", "step": 0, "synthetic": synthetic, **{k: v for k, v in s0.items() if k != "failures"}})
        logger.info("[eval @0 untrained] val final loss %.4f", s0.get("final_loss_mean", float("nan")))

    logger.info("GRRL training: total=%d rollout=%d warm_start=%s synthetic=%s device=%s",
                args.total_timesteps, args.rollout_steps, args.use_warm_start, synthetic, args.device)
    t_start = time.time()
    _, d, w = sampler.next()
    obs, info = train_env.reset(d, w, use_warm_start=args.use_warm_start)
    ep_reward, ep_start_loss = 0.0, info["start_loss"]

    while step_count < args.total_timesteps:
        buffer.reset()
        while not buffer.is_full and step_count < args.total_timesteps:
            mask = train_env.get_action_mask()
            a, cont, logp, val = agent.select_action(obs, action_mask=mask, deterministic=False)
            next_obs, reward, terminated, truncated, sinfo = train_env.step(a, continuous_params=cont)
            done = terminated or truncated
            nv = agent.value(next_obs) if truncated else None   # time-limit: bootstrap with V(s_next)
            buffer.add(obs, mask, a, cont, reward, val, logp, done, truncated=truncated, next_value=nv)
            step_count += 1; ep_reward += reward
            n_fail += int(sinfo.get("render_failed", False))
            if done:
                episode_count += 1; n_term += int(terminated); n_trunc += int(truncated); n_stop += int(sinfo["action"] == "STOP")
                logger.debug("episode %d reward=%.3f start=%.4f final=%.4f chain=%s", episode_count, ep_reward, ep_start_loss,
                             sinfo["current_loss"], [s["effect"] for s in train_env.current_chain])
                _, d, w = sampler.next()
                obs, info = train_env.reset(d, w, use_warm_start=args.use_warm_start)
                ep_reward, ep_start_loss = 0.0, info["start_loss"]
            else:
                obs = next_obs
        last_done = bool(buffer.dones[buffer.ptr - 1].item())
        buffer.compute_returns_and_advantages(last_value=0.0 if last_done else agent.value(obs), last_done=last_done,
                                              gamma=args.gamma, gae_lambda=args.gae_lambda)
        tm = agent.train_step(buffer, batch_size=args.batch_size, epochs=args.epochs)
        update_count += 1
        elapsed = time.time() - t_start
        entry = {"kind": "train", "step": step_count, "update": update_count, "episodes": episode_count,
                 "terminated": n_term, "truncated": n_trunc, "stop_actions": n_stop, "render_failures": n_fail,
                 "elapsed_sec": round(elapsed, 1), "fps": round(step_count / max(elapsed, 1e-6), 1), **tm}
        if step_count >= next_eval or step_count >= args.total_timesteps:
            summ = run_validation(eval_env, agent, val_pairs, args, step_count, out_dir)
            while next_eval <= step_count:
                next_eval += args.eval_freq
            entry.update({f"eval_{k}": v for k, v in summ.items() if not isinstance(v, (dict, list))})
            if "vs_predictor" in summ:
                entry["eval_rl_improvement_vs_predictor"] = summ["vs_predictor"]["mean_improvement"]
            score = summ.get("final_loss_mean", float("inf"))
            logger.info("[eval @%d] val final loss %.4f | vs start %+.4f | stop %.0f%% | trunc %.0f%% | steps %.1f",
                        step_count, score, summ.get("vs_start", {}).get("mean_improvement", float("nan")),
                        100 * summ.get("stop_rate", 0), 100 * summ.get("truncation_rate", 0), summ.get("mean_steps", 0))
            if score < best:
                best = score
                save(out_dir / "best_grrl_model.pth", dict(best_val_final_loss=best, metric="mean val final MR-STFT loss (lower better)",
                                                           step=step_count, n_val=len(val_pairs), deterministic_eval=True))
            save(out_dir / "last_grrl_model.pth", dict(best_val_final_loss=best, step=step_count))
        log(entry)

    save(out_dir / "final_grrl_model.pth", dict(best_val_final_loss=best, final=True))
    logger.info("Done: %d steps, %d updates, %d episodes. NOTE: this is a training run, not evidence of improvement; "
                "see rl/evaluate_protocol.py for the held-out comparison.", step_count, update_count, episode_count)
    return {"steps": step_count, "updates": update_count, "episodes": episode_count, "best_val_final_loss": best}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train GRRL agent with PPO on corrected DAPN")
    a = p.add_argument
    a("--total-timesteps", type=int, default=1000); a("--rollout-steps", type=int, default=64)
    a("--epochs", type=int, default=4); a("--batch-size", type=int, default=32)
    a("--lr", type=float, default=3e-4); a("--clip-eps", type=float, default=0.2)
    a("--value-coef", type=float, default=0.5); a("--entropy-coef", type=float, default=0.01)
    a("--max-grad-norm", type=float, default=0.5); a("--gamma", type=float, default=0.99)
    a("--gae-lambda", type=float, default=0.95); a("--max-chain-length", type=int, default=8)
    a("--max-episode-steps", type=int, default=24); a("--step-penalty", type=float, default=0.05)
    a("--use-warm-start", action="store_true"); a("--predictor-checkpoint", default=None)
    a("--allow-param-edits", action="store_true", help="choosing an effect already in the chain edits its knobs in place (residual, see GRRLEnv)")
    a("--edit-range", type=float, default=0.25)
    a("--eval-freq", type=int, default=200); a("--eval-episodes", type=int, default=5)
    a("--train-metadata", default="dataset/synthetic/train/metadata.jsonl")
    a("--val-metadata", default="dataset/synthetic/val/metadata.jsonl")
    a("--max-train-samples", type=int, default=200); a("--max-val-samples", type=int, default=50)
    a("--checkpoint-dir", default="checkpoints_v2/grrl")
    a("--resume", default=None, help="Resume training (must exist; restores optimizer, RNG, counters)")
    a("--init-from", default=None, help="Load weights only (no optimizer/RNG/counters)")
    a("--allow-synthetic-smoke", action="store_true"); a("--device", default="cpu"); a("--seed", type=int, default=42)
    return p


def main(argv=None) -> None:
    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
