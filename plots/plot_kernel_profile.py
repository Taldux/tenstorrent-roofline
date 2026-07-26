#!/usr/bin/env python3
"""
Device profiler breakdown


Usage:
    python3 plots/plot_kernel_profile.py
Requires (produced by `ENABLE_DEVICE_PROFILER=1 bash run_benchmarks.sh`):
    results/profiler/peak_bandwidth_raw/.logs/profile_log_device.csv

Raw log format: line 1 is an "ARCH: ..., CHIP_FREQ[MHz]: <f>, ..."
header, line 2 is the column header, then one row per timer event: PCIe slot, core_x,
core_y, RISC processor type, timer_id, time[cycles since reset], data, run
host ID, trace id, trace id counter, zone name, type (ZONE_START/
ZONE_END), source line, source file, meta data
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).parent.parent / "results"
PROFILER = RESULTS / "profiler"
PLOTS = Path(__file__).parent
PLOTS.mkdir(exist_ok=True)

BENCH = "peak_bandwidth"
FALLBACK_CLOCK_GHZ = 0.9855

FREQ_RE = re.compile(r"CHIP_FREQ\[MHz\]:\s*([\d.]+)")


def load_kernel_durations(path: Path):
    """Parse a raw profile_log_device.csv.

    Series are keyed "<RISC type> · <zone name>" (e.g. "TRISC_1 ·
    TRISC-KERNEL"), aggregating every core and every dispatch's duration
    into one mean +/- stddev
    """
    with open(path, newline="") as f:
        lines = f.readlines()
    if len(lines) < 3:
        return {}

    freq_mhz = FALLBACK_CLOCK_GHZ * 1e3
    m = FREQ_RE.search(lines[0])
    if m:
        freq_mhz = float(m.group(1))

    groups = defaultdict(list)
    for row in csv.reader(lines[2:]):
        if len(row) < 12:
            continue
        core_x, core_y, risc = row[1].strip(), row[2].strip(), row[3].strip()
        zone, kind = row[10].strip(), row[11].strip()
        if kind not in ("ZONE_START", "ZONE_END"):
            continue
        try:
            cycles = int(row[5].strip())
        except ValueError:
            continue
        groups[(core_x, core_y, risc, zone)].append((cycles, kind))

    series = defaultdict(list)
    for (core_x, core_y, risc, zone), events in groups.items():
        events.sort(key=lambda e: e[0])
        pending_start = None
        for cycles, kind in events:
            if kind == "ZONE_START":
                pending_start = cycles
            elif kind == "ZONE_END" and pending_start is not None:
                duration_cycles = cycles - pending_start
                if duration_cycles >= 0:
                    series[f"{risc} · {zone}"].append(duration_cycles / freq_mhz)
                pending_start = None

    return dict(series)


# Load
log_path = PROFILER / f"{BENCH}_raw" / ".logs" / "profile_log_device.csv"
if not log_path.exists():
    sys.exit(
        f"{log_path.relative_to(RESULTS.parent)} not found.\n"
        "Run: ENABLE_DEVICE_PROFILER=1 bash run_benchmarks.sh\n"
        "then re-run this script."
    )

cols = load_kernel_durations(log_path)
if not cols:
    sys.exit(f"{log_path.name} produced no ZONE_START/END pairs — check the raw log, format may have changed")

# Summary Csv
summary_path = PROFILER / "kernel_profile_summary.csv"
with open(summary_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["series", "n_samples", "mean_us", "std_us", "min_us", "max_us"])
    for series, vals in cols.items():
        w.writerow([series, len(vals),
                    f"{mean(vals):.4f}", f"{pstdev(vals):.4f}",
                    f"{min(vals):.4f}", f"{max(vals):.4f}"])
print(f"Saved {summary_path}")

print(f"\n{BENCH}: per-kernel device duration (mean ± std over n samples):")
for series, vals in sorted(cols.items(), key=lambda kv: -mean(kv[1])):
    print(f"  {series:40s} {mean(vals):8.2f} ± {pstdev(vals):6.2f} µs  (n={len(vals)})")

# Plot
names  = list(cols.keys())
means  = [mean(cols[c]) for c in names]
stds   = [pstdev(cols[c]) for c in names]
ns     = [len(cols[c]) for c in names]
order  = np.argsort(means)[::-1]
names  = [names[i] for i in order]
means  = [means[i] for i in order]
stds   = [stds[i] for i in order]
ns     = [ns[i] for i in order]

fig, ax = plt.subplots(figsize=(9, 5))
palette = plt.get_cmap("tab10")

y = np.arange(len(names))
colors = [palette(i % 10) for i in range(len(names))]
ax.barh(y, means, xerr=stds, color=colors, edgecolor="white",
        height=0.6, capsize=3, error_kw={"linewidth": 1})
ax.set_yticks(y)
ax.set_yticklabels([n.strip() for n in names], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Device duration (µs, mean ± std)", fontsize=10)
ax.set_title(f"{BENCH}\n(n={ns[0] if ns else 0} samples/column)", fontsize=11)
ax.grid(axis="x", alpha=0.3)
for yi, (m, n) in zip(y, zip(means, ns)):
    ax.text(m, yi, f"  {m:.2f} µs", va="center", fontsize=7.5)

fig.suptitle("Per-kernel device profiler breakdown — Wormhole n300", fontsize=13)
plt.tight_layout()
out = PLOTS / "kernel_profile.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
