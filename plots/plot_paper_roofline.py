#!/usr/bin/env python3
"""
Roofline figure for the paper

Usage:
    python3 plots/plot_paper_roofline.py [results_dir] [out_dir] [dram_add_dir]
    defaults: results/  paper/figures/  (none)

Reads peak_bandwidth.csv, peak_flops_sweep.csv and (optional) ttnn_ops.csv.
If peak_flops_block_matrix.csv exists, the compute roofs use the best matmul_block shape.
If dram_add_dir is given, the DRAM slope uses the best copy variant from its dram_add_*.csv
(mean over runs), if that is higher than peak_bandwidth.csv.
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "paper" / "figures"
DRAM_ADD = Path(sys.argv[3]) if len(sys.argv) > 3 else None
OUT.mkdir(parents=True, exist_ok=True)

THEO_BW_GBS = 288.0
COLUMN_WIDTH_IN = 3.5

STYLE = {
    "BF4": "#2ca02c",
    "BF8": "#1f77b4",
    "BF16": "#9467bd",
}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# my DRAM kernel: best point of the core count sweep
bw_rows = read_csv(RESULTS / "peak_bandwidth.csv")
dram_gbs = max(float(r["bw_gbs"]) for r in bw_rows)

# best copy variant of dram_add (1 read + 1 write, same workload as peak_bandwidth)
if DRAM_ADD:
    runs = {}
    for path in DRAM_ADD.glob("dram_add_*.csv"):
        for r in read_csv(path):
            if r["variant"].startswith("copy"):
                runs.setdefault(r["variant"], []).append(float(r["bw_gbs"]))
    if runs:
        dram_gbs = max(dram_gbs, max(sum(v) / len(v) for v in runs.values()))

# my matmul at each format's recommended fidelity
compute = [
    {
        "format": r["data_format"],
        "fidelity": r["math_fidelity"],
        "tflops": float(r["tflops"]),
        "peak": float(r["theoretical_peak_tflops"]),
    }
    for r in read_csv(RESULTS / "peak_flops_sweep.csv")
]

# if the matmul_block results exist, use the best block shape at the same fidelity
block_path = RESULTS / "peak_flops_block_matrix.csv"
if block_path.exists():
    block = [r for r in read_csv(block_path) if r["kernel"].startswith("block")]
    for c in compute:
        rows = [
            r
            for r in block
            if r["data_format"] == c["format"] and r["math_fidelity"] == c["fidelity"]
        ]
        if rows:
            c["tflops"] = max(float(r["tflops"]) for r in rows)

ttnn_path = RESULTS / "ttnn_ops.csv"
ttnn = read_csv(ttnn_path) if ttnn_path.exists() else []

plt.rcParams.update(
    {
        "font.size": 7,
        "axes.labelsize": 7,
        "legend.fontsize": 5.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "pdf.fonttype": 42,
    }
)

fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.6))
ai = np.logspace(-2, 5, 1000)  # FLOP/byte

# bandwidth slopes (TFLOPS = GB/s * FLOP/byte / 1000)
theo_slope = THEO_BW_GBS * ai / 1e3
meas_slope = dram_gbs * ai / 1e3

for c in compute:
    color = STYLE.get(c["format"], "#8c564b")
    # theoretical roof
    ax.plot(
        ai, np.minimum(theo_slope, c["peak"]), color=color, lw=0.8, ls="--", alpha=0.6
    )
    # measured roof
    ax.plot(
        ai,
        np.minimum(meas_slope, c["tflops"]),
        color=color,
        lw=1.4,
        label=f"{c['format']} {c['fidelity']} ({c['tflops']:.0f} / {c['peak']:.0f})",
    )

# label the two DRAM slopes once
ax.plot(
    [],
    [],
    color="black",
    lw=0.8,
    ls="--",
    alpha=0.6,
    label=f"theoretical ({THEO_BW_GBS:.0f} GB/s)",
)
ax.plot(
    [],
    [],
    color="black",
    lw=1.4,
    label=f"measured ({dram_gbs:.1f} GB/s)",
)

# TTNN operators
if ttnn:
    families = {
        "matmul": ("TTNN matmul (LoFi)", "o", "#17becf"),
        "add": ("TTNN add", "^", "#d62728"),
        "exp": ("TTNN exp", "s", "#ff7f0e"),
    }
    seen = set()
    for r in ttnn:
        fam = r["op"].split("_")[0]
        if fam not in families:
            continue
        name, marker, color = families[fam]
        ax.scatter(
            float(r["arithmetic_intensity"]),
            float(r["tflops"]),
            marker=marker,
            s=12,
            color=color,
            edgecolors="black",
            linewidths=0.3,
            zorder=5,
            label=None if fam in seen else name,
        )
        seen.add(fam)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(ai[0], ai[-1])
ax.set_ylim(1e-2, 500)
ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
ax.set_ylabel("Throughput (TFLOPS)")

# plain numbers instead of 10^x, and no "0.0" tick
plain = ticker.FuncFormatter(lambda v, _: f"{v:g}")
ax.yaxis.set_major_locator(ticker.LogLocator(base=10))
ax.yaxis.set_major_formatter(plain)
ax.yaxis.set_minor_formatter(ticker.NullFormatter())
ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
ax.xaxis.set_minor_formatter(ticker.NullFormatter())

ax.grid(True, which="major", alpha=0.25, lw=0.5)
ax.legend(loc="lower right", framealpha=0.9, handlelength=1.8)

fig.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"roofline.{ext}", dpi=300)
print(
    f"Saved {OUT / 'roofline.pdf'}  (DRAM {dram_gbs:.1f} GB/s, results from {RESULTS})"
)
