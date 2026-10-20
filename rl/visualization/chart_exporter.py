"""
rl/visualization/chart_exporter.py
==================================
Exports standalone SVG charts and self-contained HTML reports for offline viewing
without requiring any server or external dependencies.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def export_svg_charts(summary: dict[str, Any], output_path: str | Path) -> Path:
    """Generate a clean, standalone vector SVG image containing:
    1. Action Chain Sequence
    2. MR-STFT Loss Trajectory Chart
    3. Step & Cumulative Reward Chart
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    loss_hist = summary.get("loss_history", [])
    reward_hist = summary.get("reward_history", [])
    chain = summary.get("current_chain", [])
    init_loss = summary.get("initial_loss", 7.0)
    final_loss = summary.get("current_loss", 2.5)
    imprv = summary.get("improvement_pct", 0.0)
    cum_rew = summary.get("cumulative_reward", 0.0)

    w = 900
    h = 600

    # Loss chart coordinates (Left: 50, Top: 180, Width: 380, Height: 240)
    # Reward chart coordinates (Right: 480, Top: 180, Width: 380, Height: 240)
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="background-color: #0b111e; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">',
        '<defs>',
        '  <linearGradient id="gradCard" x1="0" y1="0" x2="0" y2="1">',
        '    <stop offset="0%" stop-color="#151f35"/>',
        '    <stop offset="100%" stop-color="#0f172a"/>',
        '  </linearGradient>',
        '</defs>',
        # Header Banner
        '<rect x="20" y="20" width="860" height="70" rx="10" fill="url(#gradCard)" stroke="#1e293b" stroke-width="1"/>',
        '<text x="40" y="52" fill="#38bdf8" font-size="18" font-weight="bold">DAPN RL Audio Synthesis — Execution Graphs</text>',
        f'<text x="40" y="74" fill="#94a3b8" font-size="12">Loss: {init_loss:.4f} &rarr; {final_loss:.4f} (+{imprv:.1f}% improvement) | Cumulative Reward: {cum_rew:+.3f} | Steps: {len(loss_hist)-1}</text>',
    ]

    # Action Sequence Ribbon (Y: 105 to 155)
    svg_parts.append('<rect x="20" y="105" width="860" height="52" rx="8" fill="#111c34" stroke="#1e293b" stroke-width="1"/>')
    svg_parts.append('<text x="35" y="136" fill="#cbd5e1" font-size="12" font-weight="bold">FX Chain:</text>')

    chain_seq = ["Dry Input"] + [item["effect"] for item in chain] + ["Final Output"]
    x_pos = 110
    for idx, eff in enumerate(chain_seq):
        color = "#3b82f6" if idx == 0 else ("#10b981" if idx == len(chain_seq) - 1 else "#6366f1")
        svg_parts.append(f'<rect x="{x_pos}" y="117" width="95" height="28" rx="4" fill="{color}" opacity="0.2" stroke="{color}" stroke-width="1"/>')
        svg_parts.append(f'<text x="{x_pos + 47}" y="135" fill="#f8fafc" font-size="11" font-weight="bold" text-anchor="middle">{eff}</text>')
        x_pos += 105
        if idx < len(chain_seq) - 1:
            svg_parts.append(f'<text x="{x_pos - 6}" y="135" fill="#64748b" font-size="12" text-anchor="middle">&rarr;</text>')

    # -------------------------------------------------------------
    # 1. Loss Chart (Left)
    # -------------------------------------------------------------
    lx, ly, lw, lh = 20, 175, 420, 390
    svg_parts.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" rx="10" fill="url(#gradCard)" stroke="#1e293b" stroke-width="1"/>')
    svg_parts.append(f'<text x="{lx + 20}" y="{ly + 32}" fill="#f8fafc" font-size="14" font-weight="bold">Multi-Resolution STFT Loss</text>')
    svg_parts.append(f'<text x="{lx + 20}" y="{ly + 50}" fill="#94a3b8" font-size="11">Spectral distance convergence across steps</text>')

    # Plot area
    px, py, pw, ph = lx + 55, ly + 75, lw - 80, lh - 110
    losses = [pt["loss"] for pt in loss_hist] if loss_hist else [init_loss, final_loss]
    max_l = max(max(losses), init_loss, 1.0) * 1.15
    min_l = 0.0
    n_steps = max(len(losses) - 1, 1)

    def to_loss_coord(step_i: int, l_val: float) -> tuple[float, float]:
        cx = px + (step_i / n_steps) * pw
        cy = py + ph - ((l_val - min_l) / (max_l - min_l)) * ph
        return cx, cy

    # Grid & Y-Axis
    for i in range(5):
        val = min_l + (i / 4.0) * (max_l - min_l)
        _, y_coord = to_loss_coord(0, val)
        svg_parts.append(f'<line x1="{px}" y1="{y_coord}" x2="{px + pw}" y2="{y_coord}" stroke="#334155" stroke-dasharray="2,2"/>')
        svg_parts.append(f'<text x="{px - 8}" y="{y_coord + 4}" fill="#64748b" font-size="10" text-anchor="end">{val:.1f}</text>')

    # X-Axis Ticks
    for s_i in range(n_steps + 1):
        x_coord, _ = to_loss_coord(s_i, 0)
        label = "Dry" if s_i == 0 else f"S{s_i}"
        svg_parts.append(f'<text x="{x_coord}" y="{py + ph + 20}" fill="#64748b" font-size="10" text-anchor="middle">{label}</text>')

    # Target threshold (0.40) line
    _, target_y = to_loss_coord(0, 0.40)
    svg_parts.append(f'<line x1="{px}" y1="{target_y}" x2="{px + pw}" y2="{target_y}" stroke="#ef4444" stroke-dasharray="4,4" stroke-width="1.5"/>')
    svg_parts.append(f'<text x="{px + pw - 4}" y="{target_y - 5}" fill="#ef4444" font-size="9" text-anchor="end">Target 0.40</text>')

    # Loss Curve
    pts = [to_loss_coord(pt["step"], pt["loss"]) for pt in loss_hist]
    if pts:
        path_str = f"M {pts[0][0]:.1f} {pts[0][1]:.1f}"
        for pt in pts[1:]:
            path_str += f" L {pt[0]:.1f} {pt[1]:.1f}"
        svg_parts.append(f'<path d="{path_str}" fill="none" stroke="#6366f1" stroke-width="3" stroke-linecap="round"/>')
        for idx, pt in enumerate(pts):
            l_val = loss_hist[idx]["loss"]
            svg_parts.append(f'<circle cx="{pt[0]:.1f}" cy="{pt[1]:.1f}" r="4.5" fill="#6366f1" stroke="#ffffff" stroke-width="1.5"/>')
            svg_parts.append(f'<text x="{pt[0]:.1f}" y="{pt[1] - 8:.1f}" fill="#cbd5e1" font-size="10" font-weight="bold" text-anchor="middle">{l_val:.2f}</text>')

    # -------------------------------------------------------------
    # 2. Reward Chart (Right)
    # -------------------------------------------------------------
    rx, ry, rw, rh = 460, 175, 420, 390
    svg_parts.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" rx="10" fill="url(#gradCard)" stroke="#1e293b" stroke-width="1"/>')
    svg_parts.append(f'<text x="{rx + 20}" y="{ry + 32}" fill="#f8fafc" font-size="14" font-weight="bold">Step &amp; Cumulative Reward</text>')
    svg_parts.append(f'<text x="{rx + 20}" y="{ry + 50}" fill="#94a3b8" font-size="11">Green = improvement reward; Red = step penalty</text>')

    r_px, r_py, r_pw, r_ph = rx + 55, ry + 75, rw - 80, rh - 110
    step_rewards = [pt["reward"] for pt in reward_hist] if reward_hist else [0.0]
    cum_rewards = [pt["cumulative"] for pt in reward_hist] if reward_hist else [0.0]

    max_r = max(max(step_rewards), max(cum_rewards), 5.0) * 1.2
    min_r = min(min(step_rewards), min(cum_rewards), -5.0) * 1.2
    r_n_steps = max(len(reward_hist), 1)

    def to_rew_coord(step_i: int, r_val: float) -> tuple[float, float]:
        cx = r_px + ((step_i - 0.5) / r_n_steps) * r_pw
        cy = r_py + r_ph - ((r_val - min_r) / (max_r - min_r)) * r_ph
        return cx, cy

    # Zero Line
    zero_y = r_py + r_ph - ((0.0 - min_r) / (max_r - min_r)) * r_ph
    svg_parts.append(f'<line x1="{r_px}" y1="{zero_y}" x2="{r_px + r_pw}" y2="{zero_y}" stroke="#64748b" stroke-width="1"/>')
    svg_parts.append(f'<text x="{r_px - 8}" y="{zero_y + 4}" fill="#64748b" font-size="10" text-anchor="end">0.0</text>')

    # Step Reward Bars
    bar_w = min(30.0, (r_pw / r_n_steps) * 0.45)
    for pt in reward_hist:
        bx, by = to_rew_coord(pt["step"], pt["reward"])
        h_bar = abs(by - zero_y)
        bar_y = min(by, zero_y)
        color = "#10b981" if pt["reward"] >= 0 else "#ef4444"
        svg_parts.append(f'<rect x="{bx - bar_w/2:.1f}" y="{bar_y:.1f}" width="{bar_w:.1f}" height="{max(2.0, h_bar):.1f}" rx="2" fill="{color}" opacity="0.85"/>')

    # Cumulative Return Polyline
    cum_pts = [to_rew_coord(pt["step"], pt["cumulative"]) for pt in reward_hist]
    if cum_pts:
        c_path = f"M {cum_pts[0][0]:.1f} {cum_pts[0][1]:.1f}"
        for cpt in cum_pts[1:]:
            c_path += f" L {cpt[0]:.1f} {cpt[1]:.1f}"
        svg_parts.append(f'<path d="{c_path}" fill="none" stroke="#06b6d4" stroke-width="2.5" stroke-linecap="round"/>')
        for cpt in cum_pts:
            svg_parts.append(f'<circle cx="{cpt[0]:.1f}" cy="{cpt[1]:.1f}" r="4" fill="#06b6d4" stroke="#ffffff" stroke-width="1"/>')

    svg_parts.append('</svg>')

    full_svg = "\n".join(svg_parts)
    out_path.write_text(full_svg, encoding="utf-8")
    return out_path
