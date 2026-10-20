"""Shared evaluation for GRRL / fixed-slot PPO / predictor-only / dry baselines.

Isolation guarantees (tested):
  * runs on a SEPARATE environment instance - never the training env;
  * does not consume or alter the global RNG (``torch.random.fork_rng``);
  * restores the policy's train/eval mode afterwards;
  * deterministic by default; ``deterministic=False`` gives a stochastic diagnostic.
All methods are scored with the SAME loss (``env.loss_fn`` = multi-resolution STFT distance,
lower is better) on the SAME cropped/aligned pair (``common.audio_pair.prepare_audio_pair``).
Improvements are signed: positive = lower loss than the reference, negative = degradation.
"""
from __future__ import annotations

import math
import random
import statistics
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterable

import torch

from rl.grrl_env import EFFECT_ACTIONS, STOP_ACTION_IDX

EPS_SAME = 1e-6


@contextmanager
def eval_mode(module: torch.nn.Module):
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


@contextmanager
def isolated_rng():
    py = random.getstate()
    with torch.random.fork_rng():
        try:
            yield
        finally:
            random.setstate(py)


def grrl_episode(env, agent, dry, wet, *, warm_start: bool, deterministic: bool = True, sample_id: str | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    rec: dict[str, Any] = {"sample_id": sample_id, "warm_start": warm_start, "deterministic": deterministic, "error": None}
    try:
        obs, info = env.reset(dry, wet, use_warm_start=warm_start)
        rec.update(dry_loss=info["dry_loss"], predictor_loss=info["predictor_loss"], start_loss=info["start_loss"],
                   start_chain=[s["effect"] for s in env.current_chain],
                   start_chain_full=[dict(s) for s in env.current_chain])
        steps, total_reward, done, last = [], 0.0, False, None
        comp = {}
        while not done:
            mask = env.get_action_mask()
            a, cont, logp, val = agent.select_action(obs, action_mask=mask, deterministic=deterministic)
            obs, r, term, trunc, last = env.step(a, continuous_params=cont)
            total_reward += r
            for k, v in last["reward_breakdown"].items():
                comp[k] = comp.get(k, 0.0) + v
            steps.append({"step": last["step"], "action": last["action"], "loss": last["current_loss"], "reward": r,
                          "breakdown": last["reward_breakdown"], "chain_length": last["chain_length"],
                          "log_prob": logp, "value": val})
            done = term or trunc
        rec.update(final_loss=env.current_loss, final_chain=[dict(s) for s in env.current_chain],
                   final_sequence=[s["effect"] for s in env.current_chain], steps=steps, n_steps=len(steps),
                   stop_selected=steps[-1]["action"] == "STOP", terminated=bool(last["terminated"]),
                   truncated=bool(last["truncated"]), total_reward=total_reward, reward_components=comp)
        rec["final_audio"] = env.current_audio.detach().cpu()
    except Exception as exc:  # recorded, never hidden: counted in summaries as failures
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["seconds"] = time.perf_counter() - t0
    return rec


def evaluate_grrl(env, agent, pairs: Iterable[tuple[str, torch.Tensor, torch.Tensor]], *, warm_start: bool,
                  deterministic: bool = True) -> list[dict[str, Any]]:
    """Per-example records; ``env`` must not be the training env."""
    with eval_mode(agent.actor_critic), isolated_rng():
        return [grrl_episode(env, agent, d, w, warm_start=warm_start, deterministic=deterministic, sample_id=sid)
                for sid, d, w in pairs]


def evaluate_references(env, pairs, *, predictor: bool) -> list[dict[str, Any]]:
    """Dry baseline (always) and predictor-only chain (if ``predictor``), scored by env.loss_fn."""
    out = []
    for sid, d, w in pairs:
        t0 = time.perf_counter()
        rec: dict[str, Any] = {"sample_id": sid, "error": None}
        try:
            with torch.no_grad():
                env.reset(d, w, use_warm_start=predictor)
            rec.update(dry_loss=env.initial_loss, final_loss=env.current_loss if predictor else env.initial_loss,
                       final_chain=[dict(s) for s in env.current_chain] if predictor else [],
                       final_sequence=[s["effect"] for s in env.current_chain] if predictor else [])
            env._done = True  # reference scoring only; no episode is run
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["seconds"] = time.perf_counter() - t0
        out.append(rec)
    return out


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    g = random.Random(seed)
    means = sorted(statistics.fmean(g.choices(values, k=len(values))) for _ in range(n_boot))
    return means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1]


def paired_summary(final: list[float], reference: list[float]) -> dict[str, float]:
    """improvement = reference - final (positive = better). Reports degradation honestly."""
    diffs = [r - f for f, r in zip(final, reference)]
    n = len(diffs)
    if n == 0:
        return {"n": 0}
    lo, hi = bootstrap_ci(diffs)
    return {"n": n, "mean_improvement": statistics.fmean(diffs), "median_improvement": statistics.median(diffs),
            "std_improvement": statistics.stdev(diffs) if n > 1 else float("nan"),
            "ci95_low": lo, "ci95_high": hi,
            "pct_improved": 100 * sum(d > EPS_SAME for d in diffs) / n,
            "pct_degraded": 100 * sum(d < -EPS_SAME for d in diffs) / n,
            "pct_unchanged": 100 * sum(abs(d) <= EPS_SAME for d in diffs) / n,
            "mean_relative_improvement_pct": statistics.fmean(100 * d / max(abs(r), 1e-8) for d, r in zip(diffs, reference))}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if not r.get("error")]
    out: dict[str, Any] = {"n_total": len(records), "n_failed": len(records) - len(ok),
                           "failures": [{"sample_id": r.get("sample_id"), "error": r["error"]} for r in records if r.get("error")]}
    if not ok:
        return out
    fl = [r["final_loss"] for r in ok]
    out.update(final_loss_mean=statistics.fmean(fl), final_loss_std=statistics.stdev(fl) if len(fl) > 1 else float("nan"),
               final_loss_median=statistics.median(fl),
               mean_chain_length=statistics.fmean(len(r["final_chain"]) for r in ok),
               seconds_mean=statistics.fmean(r["seconds"] for r in ok))
    if "dry_loss" in ok[0]:
        out["vs_dry"] = paired_summary(fl, [r["dry_loss"] for r in ok])
    if all(r.get("predictor_loss") is not None for r in ok) and ok[0].get("predictor_loss") is not None:
        out["vs_predictor"] = paired_summary(fl, [r["predictor_loss"] for r in ok])
    if "start_loss" in ok[0]:
        out["vs_start"] = paired_summary(fl, [r["start_loss"] for r in ok])
    if "stop_selected" in ok[0]:
        out.update(stop_rate=sum(r["stop_selected"] for r in ok) / len(ok),
                   truncation_rate=sum(r["truncated"] for r in ok) / len(ok),
                   mean_steps=statistics.fmean(r["n_steps"] for r in ok),
                   mean_total_reward=statistics.fmean(r["total_reward"] for r in ok))
        comps: dict[str, list[float]] = {}
        for r in ok:
            for k, v in r["reward_components"].items():
                comps.setdefault(k, []).append(v)
        out["reward_components_mean"] = {k: statistics.fmean(v) for k, v in comps.items()}
    return out


def load_pairs(dataset, indices: Iterable[int]) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    pairs = []
    for i in indices:
        s = dataset[i]
        pairs.append((s["sample_id"], s["dry_audio"], s["wet_audio"]))
    return pairs


# ---------------------------------------------------------------------------
# Fixed-slot PPO baseline (AudioFXEnv): same record schema, so summarize() is shared.
# ---------------------------------------------------------------------------
def ppo_episode(env, agent, dry, wet, *, warm_start: bool, deterministic: bool = True, sample_id: str | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    rec: dict[str, Any] = {"sample_id": sample_id, "warm_start": warm_start, "deterministic": deterministic, "error": None}
    try:
        obs, info = env.reset(dry, wet, use_warm_start=warm_start)
        rec.update(dry_loss=info["initial_loss"], predictor_loss=info["predictor_loss"], start_loss=info["start_loss"])
        done, steps, total, comp, last = False, [], 0.0, {}, None
        while not done:
            action_env, _, _, logp, val = agent.select_action(obs, deterministic=deterministic)
            obs, r, term, trunc, last = env.step(action_env)
            total += r
            for k, v in last["reward_breakdown"].items():
                comp[k] = comp.get(k, 0.0) + v
            steps.append({"step": last["step"], "loss": last["current_loss"], "reward": r, "breakdown": last["reward_breakdown"]})
            done = term or trunc
        chain = env.get_current_chain_summary()["chain"]
        rec.update(final_loss=env.current_loss, final_chain=chain, final_sequence=[s["effect"] for s in chain], steps=steps,
                   n_steps=len(steps), stop_selected=False, terminated=bool(term), truncated=bool(trunc), total_reward=total,
                   reward_components=comp, final_audio=env.current_audio.detach().cpu())
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["seconds"] = time.perf_counter() - t0
    return rec


def evaluate_ppo(env, agent, pairs, *, warm_start: bool, deterministic: bool = True) -> list[dict[str, Any]]:
    with eval_mode(agent.actor_critic), isolated_rng():
        return [ppo_episode(env, agent, d, w, warm_start=warm_start, deterministic=deterministic, sample_id=sid) for sid, d, w in pairs]
