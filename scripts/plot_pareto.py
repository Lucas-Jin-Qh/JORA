#!/usr/bin/env python3
"""
Plot Pareto curve: trainable params vs avg accuracy across MMLU/ARC-C/GSM8K.
Data from formal_runs/three_gpu_bf16_phase3/summary.json
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────
# anchor results (3-seed mean)
# appendix_cores results (1-seed)
# All avg = mean(MMLU, ARC-C, GSM8K)

methods = [
    # name,                           params,      avg,     family,  seeds, note
    # Exact values from summary.json (three_gpu_bf16_phase3)
    # anchor = 3-seed mean; appendix = 3-seed mean (updated 2026-03-28)
    ("JORA-selective_diag\n(s96/k16)", 14_336,    0.44870, "jora", 3, "anchor"),
    ("JORA-diag",                      266_240,   0.48857, "jora", 3, "anchor"),
    ("JORA-block (bs=4)",             1_052_672,  0.47714, "jora", 1, "appendix"),
    ("JORA-lowrank (r=1)",             528_384,   0.44975, "jora", 1, "appendix"),
    ("LoRA (r=1)",                     524_288,   0.48491, "lora", 3, "anchor"),
    ("LoRA (r=2)",                    1_048_576,  0.47385, "lora", 3, "anchor"),
]

# ── Pareto frontier computation ───────────────────────────────────────────────
def pareto_frontier(points):
    """Return indices of Pareto-optimal points (minimize params, maximize acc)."""
    pts = sorted(enumerate(points), key=lambda x: x[1][0])  # sort by params
    frontier = []
    best_acc = -1
    for idx, (params, acc) in pts:
        if acc > best_acc:
            best_acc = acc
            frontier.append(idx)
    return frontier

points = [(m[1], m[2]) for m in methods]
pareto_idx = set(pareto_frontier(points))

# Note: diag is 1-seed (appendix). Mark it visually.
appendix_idx = {i for i, m in enumerate(methods) if m[5] == "appendix" and m[4] == 1}

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

colors = {"jora": "#2196F3", "lora": "#FF5722"}
markers = {"jora": "o", "lora": "s"}
sizes = {"jora": 100, "lora": 100}

for i, (name, params, acc, family, seeds, note) in enumerate(methods):
    is_pareto = i in pareto_idx
    is_appendix = i in appendix_idx
    ec = "black" if is_pareto else colors[family]
    lw = 2.0 if is_pareto else 0.8
    # Appendix (1-seed) shown with open marker to flag lower confidence
    marker = markers[family]
    fill = colors[family] if not is_appendix else "white"
    ax.scatter(params, acc,
               color=fill,
               marker=marker,
               s=sizes[family] * (1.6 if is_pareto else 1.0),
               edgecolors=colors[family],
               linewidths=2.0 if is_appendix else lw,
               zorder=5)
    # Label offset to avoid overlap
    xoff, yoff = 0, 0.005
    ha = "left"
    if "selective" in name:
        xoff = params * 0.4
        yoff = -0.012
    elif "lowrank" in name:
        xoff = -params * 0.05
        yoff = -0.014
        ha = "right"
    elif "LoRA (r=2)" in name:
        yoff = 0.006
        ha = "right"
        xoff = -params * 0.02
    ax.annotate(name, (params, acc),
                xytext=(params + xoff, acc + yoff),
                fontsize=8, ha=ha, va="bottom",
                color="black")

# Draw Pareto frontier line
pareto_pts = sorted([(methods[i][1], methods[i][2]) for i in pareto_idx])
px, py = zip(*pareto_pts)
ax.step(px, py, where="post", color="gray", linestyle="--", linewidth=1.2,
        alpha=0.7, zorder=3, label="Pareto frontier")

ax.set_xscale("log")
ax.set_xlabel("Trainable Parameters", fontsize=11)
ax.set_ylabel("Avg Accuracy (MMLU / ARC-C / GSM8K)", fontsize=11)
ax.set_title("Parameter Efficiency: JORA vs LoRA\n(Mistral-7B, 1 epoch fine-tuning)", fontsize=12)

ax.set_xlim(5_000, 3_000_000)
ax.set_ylim(0.430, 0.510)
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))

# Legend
jora_patch = mpatches.Patch(color=colors["jora"], label="JORA (ours, 3-seed)")
jora_open = matplotlib.lines.Line2D([], [], marker="o", color="w",
                                     markeredgecolor=colors["jora"],
                                     markeredgewidth=2, markersize=8,
                                     label="JORA (ours, 1-seed)")
lora_patch = mpatches.Patch(color=colors["lora"], label="LoRA baseline (3-seed)")
pareto_line = matplotlib.lines.Line2D([], [], color="gray", linestyle="--",
                                       linewidth=1.2, label="Pareto frontier")
handles = [jora_patch, lora_patch, pareto_line]
if appendix_idx:
    handles.insert(1, jora_open)
ax.legend(handles=handles, fontsize=9, loc="lower right")

ax.grid(True, which="both", alpha=0.3)

out = Path(__file__).parent.parent / "figures/pareto_curve.pdf"
fig.tight_layout()
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")

# Also save PNG for quick preview
out_png = out.with_suffix(".png")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved: {out_png}")
