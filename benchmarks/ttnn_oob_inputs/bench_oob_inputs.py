#!/usr/bin/env python3
"""
Out-of-box matmul sweep with Tenstorrent's inputs

Same sweep as bench_ttnn_ops.py, run three ways:
  zeros_10        zeros, 10 timed iterations (thesis method)
  zeros_100       zeros, 100 timed iterations
  ones_rand_100   ones x random, 100 timed iterations (Tenstorrent's method)

Usage (inside Apptainer):
    python3 benchmarks/ttnn_oob_inputs/bench_oob_inputs.py [out.csv]
"""

import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ttnn_ops"))
import ttnn
from bench_ttnn_ops import (
    OOB_MATMUL_SHAPES,
    OOB_DTYPES,
    WARMUP_ITERS,
    DEVICE_ID,
    make_dram_tensor,
    bench_op,
)

VARIANTS = [
    # name, inputs, timed iters
    ("zeros_10", "zeros", 10),
    ("zeros_100", "zeros", 100),
    ("ones_rand_100", "ones_rand", 100),
]


def make_ones(rows, cols, device, dtype):
    return ttnn.ones(
        (rows, cols),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def make_random(rows, cols, device, dtype):
    """ttnn.rand if available, otherwise torch.randn. Returns (tensor, source)"""
    try:
        t = ttnn.rand(
            (rows, cols),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return t, "ttnn.rand"
    except (AttributeError, TypeError, RuntimeError):
        import torch

        t = ttnn.from_torch(
            torch.randn((rows, cols), dtype=torch.float32),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return t, "torch.randn"


def run_variant(device, grid, variant, inputs, iters):
    num_cores = grid.x * grid.y
    rows = []

    print("\n" + "=" * 60)
    print(f"Out of Box sweep: {variant}")
    print("=" * 60)

    for label, dtype, peak_per_core in OOB_DTYPES:
        peak_tflops = peak_per_core * num_cores
        for base_m, base_k, base_n in OOB_MATMUL_SHAPES:
            m = base_m * grid.y
            k = base_k * grid.x
            n = base_n * grid.x

            if inputs == "zeros":
                in0_t = make_dram_tensor(m, k, device, dtype=dtype)
                in1_t = make_dram_tensor(k, n, device, dtype=dtype)
                in1_source = "ttnn.zeros"
            else:
                in0_t = make_ones(m, k, device, dtype)
                in1_t, in1_source = make_random(k, n, device, dtype)

            def run_oob_matmul(a=in0_t, b=in1_t):
                return a @ b

            elapsed_s = bench_op(
                run_oob_matmul, device, warmup=WARMUP_ITERS, iters=iters
            )

            flops = 2 * m * k * n
            tflops = flops / elapsed_s / 1e12
            util_pct = tflops / peak_tflops * 100.0

            print(
                f"  {label:5s} m={m:5d} k={k:5d} n={n:5d}  "
                f"{elapsed_s * 1e3:8.3f} ms  {tflops:7.2f} TFLOPS  {util_pct:5.1f}%"
            )
            rows.append(
                dict(
                    variant=variant,
                    inputs=inputs,
                    in1_source=in1_source,
                    timed_iters=iters,
                    dtype=label,
                    base_m=base_m,
                    base_k=base_k,
                    base_n=base_n,
                    m=m,
                    k=k,
                    n=n,
                    grid_x=grid.x,
                    grid_y=grid.y,
                    elapsed_ms=round(elapsed_s * 1e3, 4),
                    tflops=round(tflops, 4),
                    theoretical_peak_tflops=round(peak_tflops, 2),
                    utilization_pct=round(util_pct, 2),
                )
            )

            in0_t.deallocate()
            in1_t.deallocate()

    return rows


def main():
    out_csv = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parents[2] / "results" / "ttnn_oob_inputs.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Opening device {DEVICE_ID} …")
    device = ttnn.open_device(device_id=DEVICE_ID)
    grid = device.compute_with_storage_grid_size()
    print(f"Compute grid: {grid.x} × {grid.y}  ({grid.x * grid.y} cores)")

    rows = []
    for variant, inputs, iters in VARIANTS:
        rows += run_variant(device, grid, variant, inputs, iters)

    fieldnames = [
        "variant",
        "inputs",
        "in1_source",
        "timed_iters",
        "dtype",
        "base_m",
        "base_k",
        "base_n",
        "m",
        "k",
        "n",
        "grid_x",
        "grid_y",
        "elapsed_ms",
        "tflops",
        "theoretical_peak_tflops",
        "utilization_pct",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {out_csv}")
    ttnn.close_device(device)
    print("Done.")


if __name__ == "__main__":
    main()
