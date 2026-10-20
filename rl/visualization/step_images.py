"""PNG export of every RL step (real-mode sessions only).

For each step (0 = starting state: dry or predictor render) one image shows
  row 1: waveform of the rendered audio vs the target (decision crop),
  row 2: log-magnitude spectrograms - rendered | target | |difference| (what the agent "sees"),
  title: action, knob values (physical), loss, delta, step reward, chain so far.
Plus ``summary.png``: loss and reward per step with the target line, and the spectral-difference strip.
Everything is computed from the session's recorded audio; nothing is simulated (demo sessions are refused).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def _spec(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(1, -1).float()
    win = torch.hann_window(1024)
    return 20 * torch.log10(torch.stft(x, 1024, 256, window=win, return_complex=True).abs()[0] + 1e-5)


def _chain_label(chain: list[dict[str, Any]]) -> str:
    return " → ".join(c["effect"] for c in chain) or "(empty)"


def export_step_pngs(engine, out_dir: str | Path) -> list[Path]:
    if engine.mode != "real" or not engine.step_audio:
        raise ValueError("Step images require a real-mode session with recorded step audio (demo sessions are simulated)")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    sr = engine.sample_rate
    target = engine.env.target_audio.detach().cpu()
    ts = _spec(target)
    summary = engine.get_summary()
    losses = [h["loss"] for h in summary["loss_history"]]
    paths: list[Path] = []
    diffs = []
    start_chain = engine.stages["preprocess"]["data"].get("start_chain", [])
    for i, audio in enumerate(engine.step_audio):
        a = audio.reshape(1, -1)
        n = min(a.shape[-1], target.shape[-1])
        sp = _spec(a[..., :n]); st = ts[:, : sp.shape[1]]
        diff = (sp - st).abs(); diffs.append(diff)
        if i == 0:
            title = f"Step 0 — start ({summary['baseline_name']}): chain {_chain_label(start_chain)}"
            sub = f"loss {losses[0]:.4f}"
        else:
            h = summary["steps_history"][i - 1]
            knobs = ", ".join(f"{p['name']}={p['display']}" for p in h["parameters"]) or "-"
            title = f"Step {i} — {h['action']}{' (edit)' if h.get('is_edit') else ''}: chain {_chain_label(h['chain'])}"
            sub = (f"loss {h['prev_loss']:.4f} → {h['current_loss']:.4f} (change {-h['delta_loss']:+.4f}; negative = closer to target) | "
                   f"reward {h['step_reward']:+.3f} | knobs: {knobs}")
        fig, ax = plt.subplots(2, 3, figsize=(15, 7), gridspec_kw={"height_ratios": [1, 1.4]})
        t = torch.arange(n) / sr
        gs = ax[0, 0].get_gridspec()
        for a_ in ax[0, :]:
            a_.remove()
        wave = fig.add_subplot(gs[0, :])
        wave.plot(t, target.reshape(-1)[:n], color="#94a3b8", lw=.6, label="target (wet)")
        wave.plot(t, a[0, :n], color="#6366f1", lw=.6, alpha=.85, label="rendered")
        wave.set_xlabel("time (s)"); wave.set_ylabel("amplitude"); wave.legend(loc="upper right"); wave.set_xlim(0, t[-1])
        vmin, vmax = float(min(sp.min(), st.min())), float(max(sp.max(), st.max()))
        ext = [0, t[-1], 0, sr / 2 / 1000]
        for k, (img, name, cmap, kw) in enumerate([(sp, "rendered (dB)", "magma", dict(vmin=vmin, vmax=vmax)),
                                                   (st, "target (dB)", "magma", dict(vmin=vmin, vmax=vmax)),
                                                   (diff, "|difference| (dB)", "viridis", {})]):
            axk = ax[1, k]
            im = axk.imshow(img.numpy(), origin="lower", aspect="auto", extent=ext, cmap=cmap, **kw)
            axk.set_title(name); axk.set_xlabel("time (s)"); axk.set_ylabel("kHz"); fig.colorbar(im, ax=axk, fraction=.046)
        fig.suptitle(title + "\n" + sub, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, .92])
        p = out / f"step_{i:02d}.png"; fig.savefig(p, dpi=110); plt.close(fig); paths.append(p)
    # summary
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    steps = list(range(len(losses)))
    axs[0, 0].plot(steps, losses, "-o", color="#6366f1"); axs[0, 0].axhline(0.40, color="#ef4444", ls="--", label="target 0.40")
    axs[0, 0].axhline(summary["dry_loss"], color="#94a3b8", ls=":", label="dry")
    if summary["predictor_loss"] is not None:
        axs[0, 0].axhline(summary["predictor_loss"], color="#f59e0b", ls=":", label="predictor-only")
    axs[0, 0].set_title("MR-STFT loss per step (lower is better)"); axs[0, 0].set_xlabel("step"); axs[0, 0].legend()
    rh = summary["reward_history"]
    if rh:
        axs[0, 1].bar([r["step"] for r in rh], [r["reward"] for r in rh], color=["#10b981" if r["reward"] >= 0 else "#ef4444" for r in rh])
        axs[0, 1].plot([r["step"] for r in rh], [r["cumulative"] for r in rh], "-o", color="#06b6d4")
    axs[0, 1].axhline(0, color="#64748b", lw=.8); axs[0, 1].set_title("step reward (bars) and cumulative (line)"); axs[0, 1].set_xlabel("step")
    ax_s = axs[1, 0]; ax_s.axis("off")
    lines = [f"mode: {summary['mode'].upper()}  warm-start: {summary['warm_started']}",
             f"dry loss {summary['dry_loss']:.4f}" + (f" | predictor {summary['predictor_loss']:.4f}" if summary["predictor_loss"] is not None else ""),
             f"final loss {summary['current_loss']:.4f}  ({summary['improvement_pct']:+.1f}% vs {summary['baseline_name']}, {summary['dry_improvement_pct']:+.1f}% vs dry)",
             f"stop: {summary['stop_reason']}", f"final chain: {_chain_label(summary['current_chain'])}"]
    ax_s.text(0, .95, "\n".join(lines), va="top", family="monospace", fontsize=10)
    mean_diff = [float(d.mean()) for d in diffs]
    axs[1, 1].plot(steps, mean_diff, "-o", color="#f59e0b"); axs[1, 1].set_title("mean |spectral difference| (dB) per step"); axs[1, 1].set_xlabel("step")
    fig.tight_layout(); p = out / "summary.png"; fig.savefig(p, dpi=110); plt.close(fig); paths.append(p)
    return paths
