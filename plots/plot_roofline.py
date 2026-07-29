#!/usr/bin/env python3
"""
Empirical Roofline Model

Usage:
    python3 plots/plot_roofline.py

Requires:
    results/peak_bandwidth.txt
    results/peak_flops_sweep.csv
Optional:
    results/l1_bandwidth.txt
    results/ttnn_ops.csv
    results/peak_flops_tileswp.csv
    results/peak_flops_tileswp_hifi4.csv
    results/peak_flops_matrix.csv
    results/peak_bandwidth.csv
"""

import re, sys, csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS = Path(__file__).parent.parent / "results"
PLOTS   = Path(__file__).parent
PLOTS.mkdir(exist_ok=True)


def parse_bandwidth(path: Path) -> float:
    text = path.read_text()
    m = re.search(r"Measured bandwidth:\s+([\d.]+)\s+GB/s", text)
    if not m:
        sys.exit(f"Could not parse bandwidth from {path}")
    return float(m.group(1))

bw_GBs = parse_bandwidth(RESULTS / "peak_bandwidth.txt")
bw_TBs = bw_GBs / 1000.0

def parse_core_sweep(path: Path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "rows_used":  int(row["rows_used"]),
                "cores_used": int(row["cores_used"]),
                "bw_gbs":     float(row["bw_gbs"]),
                "eff":        float(row["efficiency_pct"]),
            })
    return rows

core_sweep_path = RESULTS / "peak_bandwidth.csv"
core_sweep = parse_core_sweep(core_sweep_path) if core_sweep_path.exists() else []

l1_bw_path = RESULTS / "l1_bandwidth.txt"
l1_bw_GBs  = parse_bandwidth(l1_bw_path) if l1_bw_path.exists() else None
if l1_bw_GBs:
    print(f"L1/NoC BW:  {l1_bw_GBs:.0f} GB/s")
else:
    print("Note: l1_bandwidth.txt not found — omitting L1 slope")

# Parse flops CSV 

def parse_flops_csv(path: Path):
    """Returns list of dicts with keys: config, data_format, math_fidelity,
    tflops, theoretical_peak_tflops (None if unverified),
    efficiency_pct (None if peak unverified)."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            peak_raw = row.get("theoretical_peak_tflops", "NA")
            eff_raw  = row.get("efficiency_pct", "NA")
            rows.append({
                "config":         row["config"],
                "data_format":    row["data_format"],
                "math_fidelity":  row["math_fidelity"],
                "Mt": int(row["Mt"]), "Kt": int(row["Kt"]), "Nt": int(row["Nt"]),
                "tflops":         float(row["tflops"]),
                "peak_tflops":    None if peak_raw in ("NA", "") else float(peak_raw),
                "efficiency_pct": None if eff_raw in ("NA", "") else float(eff_raw),
                "time_ms":        float(row["time_per_iter_ms"]),
            })
    return rows

csv_path = RESULTS / "peak_flops_sweep.csv"
if not csv_path.exists():
    sys.exit(f"Could not find {csv_path}")
sweep = parse_flops_csv(csv_path)

print(f"Bandwidth:  {bw_GBs:.1f} GB/s")
for s in sweep:
    eff_str = f"{s['efficiency_pct']:.1f}%" if s["efficiency_pct"] is not None else "N/A (peak unverified)"
    print(f"  {s['config']:14s} {s['tflops']:.1f} TFLOPS  ({eff_str})")

# Parse TTNN ops

def parse_ttnn_ops(path: Path):
    """Returns list of dicts: op, arithmetic_intensity, tflops, gb_s"""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "op":    row["op"],
                "ai":    float(row["arithmetic_intensity"]),
                "tflops": float(row["tflops"]),
                "gb_s":   float(row["gb_s"]),
            })
    return rows

ttnn_ops_path = RESULTS / "ttnn_ops.csv"
ttnn_ops = parse_ttnn_ops(ttnn_ops_path) if ttnn_ops_path.exists() else []
if ttnn_ops:
    print(f"\nTTNN ops ({len(ttnn_ops)} entries):")
    for t in ttnn_ops:
        print(f"  {t['op']:26s} AI={t['ai']:.2f}  {t['tflops']:.4f} TFLOPS  {t['gb_s']:.1f} GB/s")
else:
    print("\nNote: results/ttnn_ops.csv not found — skipping TTNN scatter points")

# Parse tile sweep csv

def parse_tile_sweep(path: Path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "config":   row["config"],
                "tile_dim": int(row["tile_dim"]),
                "tflops":   float(row["tflops"]),
                "peak":     float(row["theoretical_peak_tflops"]),  # all BF16 LoFi, always known
                "eff":      float(row["efficiency_pct"]),
            })
    return rows

tswp_path = RESULTS / "peak_flops_tileswp.csv"
tile_sweep = parse_tile_sweep(tswp_path) if tswp_path.exists() else []

tswp_hifi4_path = RESULTS / "peak_flops_tileswp_hifi4.csv"
tile_sweep_hifi4 = parse_tile_sweep(tswp_hifi4_path) if tswp_hifi4_path.exists() else []

# Parse fidelity x matrix csv
def parse_fidelity_matrix(path: Path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "config":        row["config"],
                "data_format":   row["data_format"],
                "math_fidelity": row["math_fidelity"],
                "tflops":        float(row["tflops"]),
                "peak":          float(row["theoretical_peak_tflops"]),
                "eff":           float(row["efficiency_pct"]),
            })
    return rows

matrix_path_csv = RESULTS / "peak_flops_matrix.csv"
fidelity_matrix = parse_fidelity_matrix(matrix_path_csv) if matrix_path_csv.exists() else []


THEO_BW_GBs   = 288.0   # single chip: 12 GB GDDR6 @ 12 GT/s over 192-bit bus (n300 full card = 576 GB/s)

# Arithmetic intensity axis (FLOP/byte), log scale
ai = np.logspace(-2, 5, 2000)

##### Figure 1: Roofline Model
fig, ax = plt.subplots(figsize=(11, 7))

mem_slope_gflops = bw_TBs * 1e3 * ai   # GFLOPS = (TB/s) * (FLOP/byte)

known_peaks = [s["peak_tflops"] for s in sweep if s["peak_tflops"]]
max_peak = max(known_peaks) if known_peaks else max(s["tflops"] for s in sweep)

theo_bw_TBs = THEO_BW_GBs / 1000.0
theo_bw_slope_gflops = theo_bw_TBs * 1e3 * ai
ax.plot(ai, np.minimum(theo_bw_slope_gflops, max_peak * 1e3) / 1e3,
        color="#e07b39", linewidth=1.2, linestyle="--", alpha=0.7,
        label=f"Theoretical DRAM peak  ({THEO_BW_GBs:.0f} GB/s, "
              f"{bw_GBs / THEO_BW_GBs * 100:.0f}% measured)")
theo_bw_label_ai = 0.01
ax.annotate(f"Theoretical DRAM\n{THEO_BW_GBs:.0f} GB/s",
            xy=(theo_bw_label_ai, theo_bw_TBs * 1e3 * theo_bw_label_ai / 1e3),
            fontsize=7.5, color="#e07b39", ha="left", va="bottom",
            xytext=(theo_bw_label_ai * 1.3, theo_bw_TBs * 1e3 * theo_bw_label_ai * 1.6 / 1e3))

if l1_bw_GBs:
    l1_TBs = l1_bw_GBs / 1000.0
    l1_slope_gflops = l1_TBs * 1e3 * ai
    ax.plot(ai, np.minimum(l1_slope_gflops, max_peak * 1e3) / 1e3,
            color="#888888", linewidth=1.2, linestyle="-.", alpha=0.7,
            label=f"L1/NoC slope  ({l1_bw_GBs:.0f} GB/s measured)")
    ax.annotate(f"L1/NoC\n{l1_bw_GBs/1000:.1f} TB/s",
                xy=(0.01, l1_TBs * 1e3 * 0.01 / 1e3),
                fontsize=7.5, color="#888888", ha="left", va="bottom",
                xytext=(0.012, l1_TBs * 1e3 * 0.01 * 0.4 / 1e3))

config_styles = {
    "BF4":  ("#2ca02c", "o", "solid"),
    "BF8":  ("#1f77b4", "s", "solid"),
    "BF16": ("#9467bd", "D", "solid"),
}
default_style = ("#8c564b", "x", "solid")

for s in sweep:
    key   = s["config"]
    color, marker, ls = config_styles.get(key, default_style)
    tflops_val = s["tflops"]
    ceiling = np.full_like(ai, tflops_val * 1e3)
    line = np.minimum(mem_slope_gflops, ceiling) / 1e3
    ridge = tflops_val * 1e3 / bw_GBs   # FLOP/byte at which slope meets ceiling (TFLOPS -> GFLOPS to match GB/s)
    if s["efficiency_pct"] is not None:
        label = f"{key}  ({tflops_val:.1f} TFLOPS, {s['efficiency_pct']:.0f}% of {s['peak_tflops']:.0f} TFLOPS peak)"
    else:
        label = f"{key}  ({tflops_val:.1f} TFLOPS, peak unverified)"
    ax.plot(ai, line, color=color, linewidth=2.0, linestyle=ls, label=label)
    # Mark the ridge point
    ax.plot(ridge, tflops_val, marker=marker, color=color, markersize=7, zorder=5)
    ax.axvline(ridge, color=color, linewidth=0.6, linestyle=":", alpha=0.5)

    # Dashed theoretical ceiling for this config, if its peak is known.
    if s["peak_tflops"]:
        theo_line = np.minimum(mem_slope_gflops, np.full_like(ai, s["peak_tflops"] * 1e3)) / 1e3
        ax.plot(ai, theo_line, color=color, linewidth=1.2, linestyle="--", alpha=0.5,
                label=f"{key} theoretical peak  ({s['peak_tflops']:.0f} TFLOPS)")

bw_label_ai = 0.03
ax.annotate(f"Memory slope\n{bw_GBs:.0f} GB/s",
            xy=(bw_label_ai, bw_TBs * 1e3 * bw_label_ai / 1e3),
            fontsize=8, color="#555555",
            ha="left", va="bottom",
            xytext=(bw_label_ai * 2.5, bw_TBs * 1e3 * bw_label_ai * 0.6 / 1e3))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Arithmetic Intensity  (FLOP / byte)", fontsize=12)
ax.set_ylabel("Throughput  (TFLOPS)", fontsize=12)
ax.set_title(
    "Empirical Roofline Model — Wormhole n300, single chip\n"
    "(BF4/BF8/BF16 format sweep, each at its standard fidelity; dashed = theoretical peak)",
    fontsize=12)
ax.set_xlim(ai[0], ai[-1])
ax.set_ylim(5e-3, max_peak * 2.5)
ax.grid(True, which="both", alpha=0.25)

ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(ticker.NullFormatter())

if ttnn_ops:
    _op_style = {
        "matmul": ("#17becf", "o", 70),   # teal circles
        "add":    ("#2ca02c", "^", 70),   # green triangles
        "exp":    ("#ff7f0e", "s", 70),   # orange squares
    }
    _default_op_style = ("#9467bd", "D", 60)

    plotted_families = set()
    for t in ttnn_ops:
        family = "matmul" if t["op"].startswith("matmul") else \
                 "add"    if t["op"].startswith("add")    else \
                 "exp"    if t["op"].startswith("exp")    else "other"
        color, marker, sz = _op_style.get(family, _default_op_style)
        label = f"TTNN {family}" if family not in plotted_families else None
        plotted_families.add(family)
        ax.scatter(t["ai"], t["tflops"], color=color, marker=marker, s=sz,
                   zorder=10, label=label, edgecolors="black", linewidths=0.6)
        short = t["op"].replace("_8192x8192", "").replace("_4096x4096", "")
        if short.startswith("matmul_"):
            short = short.replace("matmul_", "MM ")
        ax.annotate(
            short, xy=(t["ai"], t["tflops"]),
            xytext=(5, 4), textcoords="offset points",
            fontsize=7.5, color=color,
            ha="left", va="bottom",
        )

ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)

plt.tight_layout()
out1 = PLOTS / "roofline.png"
fig.savefig(out1, dpi=150)
fig.savefig(out1.with_suffix(".pdf"))
print(f"Saved {out1}")

# Figure 2: Tile size sweep
if tile_sweep or tile_sweep_hifi4:
    fig3, (ax4, ax5) = plt.subplots(1, 2, figsize=(12, 4))

    all_dims = sorted({t["tile_dim"] for t in (tile_sweep + tile_sweep_hifi4)})
    x_labels = [str(d) for d in all_dims]
    x_ticks  = list(range(len(all_dims)))
    dim_to_x = {d: i for i, d in enumerate(all_dims)}

    for label, series, color, marker in [
        ("LoFi",  tile_sweep,       "#9467bd", "D"),
        ("HiFi4", tile_sweep_hifi4, "#1f77b4", "s"),
    ]:
        if not series:
            continue
        x      = [dim_to_x[t["tile_dim"]] for t in series]
        tflops = [t["tflops"] for t in series]
        eff    = [t["eff"]    for t in series]
        peak   = series[0]["peak"]
        ax4.plot(x, tflops, color=color, marker=marker, linewidth=2, label=label)
        ax4.axhline(peak, color=color, linewidth=1, linestyle="--", alpha=0.4,
                    label=f"{label} peak ({peak:.0f} TFLOPS)")
        ax5.plot(x, eff, color=color, marker=marker, linewidth=2, label=label)
        for i, v in zip(x, eff):
            ax5.text(i, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=7.5, color=color)

    ax4.set_xticks(x_ticks)
    ax4.set_xticklabels(x_labels, fontsize=10)
    ax4.set_xlabel("Mt=Kt dimension (elements, Nt=8 fixed)", fontsize=11)
    ax4.set_ylabel("Throughput (TFLOPS)", fontsize=11)
    ax4.set_title("BF16: TFLOPS vs matrix size (INNER_LOOP=10000)", fontsize=11)
    ax4.legend(fontsize=8.5)
    ax4.grid(axis="y", alpha=0.3)

    ax5.set_xticks(x_ticks)
    ax5.set_xticklabels(x_labels, fontsize=10)
    ax5.set_xlabel("Mt=Kt dimension (elements, Nt=8 fixed)", fontsize=11)
    ax5.set_ylabel("Efficiency (% of that fidelity's peak)", fontsize=11)
    ax5.set_title("BF16: efficiency vs matrix size", fontsize=11)
    ax5.set_ylim(0, 110)
    ax5.legend(fontsize=8.5)
    ax5.grid(axis="y", alpha=0.3)

    caption3 = (
        "Same Mt=Kt=T, Nt=8 matmul shape, only fidelity differs. HiFi4 does 4x more real FPU work per "
        "instruction than LoFi, so it dilutes fixed perinstruction overhead as size grows and climbs "
        "toward its peak. LoFi's plateau stays flat because its bottleneck (likely)"
        "is FPU instruction issue-rate itself, not overhead that more size can dilute."
    )
    fig3.text(0.5, -0.08, caption3, ha="center", fontsize=9, color="#444444", wrap=True, style="italic")

    plt.tight_layout()
    out3 = PLOTS / "tile_sweep.png"
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    fig3.savefig(out3.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {out3}")

# Figure 3: fidelity x format sweep
if fidelity_matrix:
    fig5, (ax8, ax9) = plt.subplots(1, 2, figsize=(12, 5))

    formats = ["BF4", "BF8", "BF16"]
    fidelities = ["LoFi", "HiFi2", "HiFi4"]
    fid_colors = {"LoFi": "#55a868", "HiFi2": "#4c72b0", "HiFi4": "#c44e52"}
    by_key = {(r["data_format"], r["math_fidelity"]): r for r in fidelity_matrix}

    bar_w = 0.25
    group_x = np.arange(len(formats))

    for j, fid in enumerate(fidelities):
        offs = (j - 1) * bar_w
        tflops_vals = [by_key[(f, fid)]["tflops"] for f in formats]
        eff_vals    = [by_key[(f, fid)]["eff"]    for f in formats]
        ax8.bar(group_x + offs, tflops_vals, width=bar_w, color=fid_colors[fid],
                edgecolor="white", label=fid)
        ax9.bar(group_x + offs, eff_vals, width=bar_w, color=fid_colors[fid],
                edgecolor="white", label=fid)
        for x, v in zip(group_x + offs, tflops_vals):
            ax8.text(x, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
        for x, v in zip(group_x + offs, eff_vals):
            ax9.text(x, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=7.5)

    ax8.set_xticks(group_x)
    ax8.set_xticklabels(formats, fontsize=11)
    ax8.set_ylabel("Throughput (TFLOPS)", fontsize=11)
    ax8.set_title("TFLOPS: every format x every fidelity", fontsize=11)
    ax8.legend(fontsize=9)
    ax8.grid(axis="y", alpha=0.3)

    ax9.set_xticks(group_x)
    ax9.set_xticklabels(formats, fontsize=11)
    ax9.set_ylabel("Efficiency (% of that fidelity's own peak)", fontsize=11)
    ax9.set_title("Efficiency: every format x every fidelity", fontsize=11)
    ax9.set_ylim(0, 110)
    ax9.legend(fontsize=9)
    ax9.grid(axis="y", alpha=0.3)

    caption5 = (
        "Mt=Kt=16, Nt=8 for all nine combinations"
        "Tests whether LoFi's inability to reach its own theoretical peak "
        "is specific to BF16 or happens for every format at LoFi, and whether HiFi4 "
        "reaches near-peak regardless of format."
    )
    fig5.text(0.5, -0.06, caption5, ha="center", fontsize=9, color="#444444", wrap=True, style="italic")

    plt.tight_layout()
    out5 = PLOTS / "fidelity_matrix.png"
    fig5.savefig(out5, dpi=150, bbox_inches="tight")
    fig5.savefig(out5.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {out5}")

# Figure 4: DRAM bandwidth vs core count sweep
if core_sweep:
    fig6, (ax10, ax11) = plt.subplots(1, 2, figsize=(11, 4))

    cores      = [c["cores_used"] for c in core_sweep]
    bw6        = [c["bw_gbs"]     for c in core_sweep]
    eff6       = [c["eff"]        for c in core_sweep]
    x_ticks6   = list(range(len(cores)))
    x_labels6  = [str(c) for c in cores]

    ax10.bar(x_ticks6, bw6, color="#4c72b0", edgecolor="white")
    ax10.axhline(THEO_BW_GBs, color="#e07b39", linewidth=1.2, linestyle="--",
                 alpha=0.7, label=f"Theoretical DRAM peak ({THEO_BW_GBs:.0f} GB/s)")
    ax10.set_xticks(x_ticks6)
    ax10.set_xticklabels(x_labels6, fontsize=10)
    ax10.set_xlabel("Cores used", fontsize=11)
    ax10.set_ylabel("Bandwidth (GB/s)", fontsize=11)
    ax10.set_title("DRAM bandwidth vs core count", fontsize=11)
    ax10.legend(fontsize=8.5)
    ax10.grid(axis="y", alpha=0.3)
    for i, v in enumerate(bw6):
        ax10.text(i, v + 2, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)

    ax11.bar(x_ticks6, eff6, color="#dd8452", edgecolor="white")
    ax11.set_xticks(x_ticks6)
    ax11.set_xticklabels(x_labels6, fontsize=10)
    ax11.set_xlabel("Cores used", fontsize=11)
    ax11.set_ylabel("Efficiency (% of theoretical DRAM peak)", fontsize=11)
    ax11.set_title("DRAM efficiency vs core count", fontsize=11)
    ax11.set_ylim(0, 110)
    ax11.grid(axis="y", alpha=0.3)
    for i, v in enumerate(eff6):
        ax11.text(i, v + 1.0, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5)

    caption6 = (
        "Same buffer size and chunk/CB params throughout, only how many core rows participate "
        "changes. Tested whether concentrating traffic on fewer cores/banks"
        "would raise achieved bandwidth. (P.S: It doesn't)"
    )
    fig6.text(0.5, -0.08, caption6, ha="center", fontsize=9, color="#444444", wrap=True, style="italic")

    plt.tight_layout()
    out6 = PLOTS / "bandwidth_sweep.png"
    fig6.savefig(out6, dpi=150, bbox_inches="tight")
    fig6.savefig(out6.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {out6}")

