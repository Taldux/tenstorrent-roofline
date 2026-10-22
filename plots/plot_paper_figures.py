#!/usr/bin/env python3
"""
Bar charts for the paper

Usage:
    python3 plots/plot_paper_figures.py <block_results_dir> <dram_add_dir> [out_dir]
    e.g. results_2026-10-30 results_2026-10-31_batch3 (out_dir default: paper/figures/)

matmul_block.pdf  matmul_tiles vs best matmul_block shape, every format x fidelity
                  (from peak_flops_block_matrix.csv)
dram_noc.pdf      copy/add with our vs tt-metal's reader/writer NOCs, next to TTNN
                  (from dram_add_*.csv and ttnn_ops.csv)
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BLOCK = Path(sys.argv[1])
DRAM = Path(sys.argv[2])
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COLUMN_WIDTH_IN = 3.5
THEO_BW_GBS = 288.0

BEFORE = "#2a78d6"
AFTER = "#eb6834"
TTNN = "#1baf7a"
INK = "#333333"
MUTED = "#777777"

plt.rcParams.update(
    {
        "font.size": 7,
        "axes.labelsize": 7,
        "legend.fontsize": 6,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "xtick.color": INK,
        "ytick.color": INK,
        "pdf.fonttype": 42,
    }
)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    print(f"Saved {OUT / (name + '.pdf')}")


##### matmul_tiles vs best matmul_block

rows = read_csv(BLOCK / "peak_flops_block_matrix.csv")
formats = ["BF4", "BF8", "BF16"]
fidelities = ["LoFi", "HiFi2", "HiFi4"]

tiles, best, peaks = [], [], []
for f in formats:
    for fid in fidelities:
        cfg = [r for r in rows if r["data_format"] == f and r["math_fidelity"] == fid]
        tiles.append(next(float(r["tflops"]) for r in cfg if r["kernel"] == "tiles"))
        best.append(
            max(float(r["tflops"]) for r in cfg if r["kernel"].startswith("block"))
        )
        peaks.append(float(cfg[0]["theoretical_peak_tflops"]))

fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.4))
# a gap between format groups
x = np.array([i + (i // 3) * 0.6 for i in range(9)])
w = 0.4

ax.bar(x - w / 2 - 0.01, tiles, w, color=BEFORE, label="matmul_tiles")
ax.bar(x + w / 2 + 0.01, best, w, color=AFTER, label="matmul_block (best)")
for xi, p in zip(x, peaks):
    ax.plot([xi - w - 0.05, xi + w + 0.05], [p, p], color=INK, lw=0.8, ls="--")
ax.plot([], [], color=INK, lw=0.8, ls="--", label="theoretical peak")

# speedup above each pair where it matters
for xi, t, b, p in zip(x, tiles, best, peaks):
    if b / t > 1.1:
        ax.text(
            xi,
            max(b, p) + 4,
            f"{b / t:.1f}×",
            ha="center",
            va="bottom",
            fontsize=6,
            color=INK,
        )

ax.set_xticks(x)
ax.set_xticklabels(fidelities * 3)
for i, f in enumerate(formats):
    ax.text(
        x[3 * i + 1],
        -0.2,
        f,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7,
        fontweight="bold",
        color=INK,
    )
ax.set_ylabel("Throughput (TFLOPS)")
ax.set_ylim(0, 250)
style_axes(ax)
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=3,
    frameon=False,
    handlelength=1.5,
    columnspacing=1.0,
)
save(fig, "matmul_block")


##### DRAM bandwidth: our vs tt-metal's NOC assignment, next to TTNN

runs = {}
for path in DRAM.glob("dram_add_*.csv"):
    for r in read_csv(path):
        runs.setdefault(r["variant"], []).append(float(r["bw_gbs"]))
mean = {k: sum(v) / len(v) for k, v in runs.items()}

ttnn = {
    r["op"].split("_")[0]: float(r["gb_s"]) for r in read_csv(DRAM / "ttnn_ops.csv")
}

groups = ["copy\n(1 read, 1 write)", "add\n(2 reads, 1 write)"]
ours = [mean["copy_128MiB"], mean["add_128MiB"]]
tt_noc = [mean["copy_128MiB_tt_noc"], mean["add_128MiB_tt_noc"]]
ttnn_vals = [ttnn["exp"], ttnn["add"]]  # exp has the same 1:1 pattern as copy

fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.4))
x = np.arange(len(groups))
w = 0.26

bars = [
    (x - w - 0.01, ours, BEFORE, "handwritten, reader NOC 1 / writer NOC 0"),
    (x, tt_noc, AFTER, "handwritten, reader NOC 0 / writer NOC 1"),
    (x + w + 0.01, ttnn_vals, TTNN, "TTNN (exp / add)"),
]
for xs, vals, color, label in bars:
    ax.bar(xs, vals, w, color=color, label=label)
    for xi, v in zip(xs, vals):
        ax.text(xi, v + 3, f"{v:.0f}", ha="center", va="bottom", fontsize=6, color=INK)

ax.axhline(THEO_BW_GBS, color=INK, lw=0.8, ls="--")
ax.text(
    x[-1] + 1.5 * w,
    THEO_BW_GBS - 4,
    f"theoretical {THEO_BW_GBS:.0f} GB/s",
    ha="right",
    va="top",
    fontsize=6,
    color=MUTED,
)

ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel("DRAM bandwidth (GB/s)")
ax.set_ylim(0, 300)
ax.set_xlim(-0.6, 1.6)
style_axes(ax)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)
save(fig, "dram_noc")
