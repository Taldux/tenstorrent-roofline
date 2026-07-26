#!/usr/bin/env python3
"""
TTNN operator benchmarks

Measures wall-clock TFLOPS and effective GB/s for:
  - ttnn.matmul  (compute-bound at large sizes)
  - ttnn.add     (memory-bound, AI ≈ 0.17 FLOP/byte)
  - ttnn.exp     (memory-bound, AI ≈ 0.25 FLOP/byte)

Arithmetic intensity is computed from the DRAM I/O footprint (inputs + output,
BF16 = 2 bytes/element)

Output CSV columns:
  op, M, K, N, flops, dram_bytes, arithmetic_intensity,
  tflops, gb_s, elapsed_ms

Usage (inside Apptainer container):
    python3 benchmarks/ttnn_ops/bench_ttnn_ops.py [output.csv]
"""

import sys
import time
import csv
from pathlib import Path

try:
    import ttnn
except ImportError:
    sys.exit(
        "ERROR: could not import ttnn.  "
        "Run via run_benchmarks.sh which uses /opt/venv/bin/python3 inside the container."
    )

WARMUP_ITERS = 5   # JIT-compile + warm cahce
BENCH_ITERS  = 10  # timed iterations
DEVICE_ID    = 0
DTYPE_BYTES  = 2   # BF16

MATMUL_CONFIGS = [
    ("matmul_64",          64,    64,    64),
    ("matmul_128",         128,   128,   128),
    ("matmul_256",         256,   256,   256),
    ("matmul_512",         512,   512,   512),
    ("matmul_1024",        1024,  1024,  1024),
    ("matmul_2048",        2048,  2048,  2048),
    ("matmul_4096",        4096,  4096,  4096),
    ("matmul_4096x1024",   4096,  1024,  4096),
]

ELTWISE_CONFIGS = [
    ("add_8192x8192",  8192, 8192),
    ("exp_8192x8192",  8192, 8192),
]

def make_dram_tensor(rows, cols, device):
    """Zero-filled BF16 tile-layout DRAM tensor"""
    return ttnn.zeros(
        (rows, cols),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def bench_op(fn, device, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """
    Run fn() `warmup` times (discarded), then `iters` times timed.
    A single ttnn.synchronize_device call after all iters gives the total
    device-side wall time; dividing by iters yields the per-op cost.
    Returns: average seconds per iteration.
    """
    for _ in range(warmup):
        _ = fn()
    ttnn.synchronize_device(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()

    return (t1 - t0) / iters


def main():
    out_csv = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parents[2] / "results" / "ttnn_ops.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Opening device {DEVICE_ID} …")
    device = ttnn.open_device(device_id=DEVICE_ID)
    grid = device.compute_with_storage_grid_size()
    print(f"Compute grid: {grid.x} × {grid.y}  ({grid.x * grid.y} cores)")

    try:
        CORE_GRID = ttnn.CoreGrid(y=grid.y, x=grid.x)
    except AttributeError:
        CORE_GRID = None

    rows = []

    # Full-grid BF16 matmul

    print("\n" + "=" * 60)
    print("MATMUL (BF16, full-grid)")
    print("=" * 60)

    for label, M, K, N in MATMUL_CONFIGS:
        a = make_dram_tensor(M, K, device)
        b = make_dram_tensor(K, N, device)

        def run_matmul(a=a, b=b):
            return ttnn.matmul(
                a, b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                core_grid=CORE_GRID,
            )

        elapsed_s = bench_op(run_matmul, device)

        flops      = 2 * M * K * N
        dram_bytes = (M * K + K * N + M * N) * DTYPE_BYTES   # A + B + C
        ai         = flops / dram_bytes
        tflops     = flops / elapsed_s / 1e12
        gb_s       = dram_bytes / elapsed_s / 1e9

        print(
            f"  {label:22s}  {elapsed_s*1e3:7.2f} ms"
            f"  {tflops:6.2f} TFLOPS  {gb_s:7.1f} GB/s  AI={ai:.0f} F/B"
        )
        rows.append(dict(
            op=label, M=M, K=K, N=N,
            flops=flops, dram_bytes=dram_bytes,
            arithmetic_intensity=round(ai, 4),
            tflops=round(tflops, 4), gb_s=round(gb_s, 3),
            elapsed_ms=round(elapsed_s * 1e3, 3),
        ))

        a.deallocate()
        b.deallocate()


    # Elementwise add

    print("\n" + "=" * 60)
    print("ELTWISE ADD (BF16)")
    print("=" * 60)

    for label, R, C in ELTWISE_CONFIGS:
        if "add" not in label:
            continue
        a = make_dram_tensor(R, C, device)
        b = make_dram_tensor(R, C, device)

        def run_add(a=a, b=b):
            return ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        elapsed_s = bench_op(run_add, device)

        flops      = R * C                       # 1 FLOP per element
        dram_bytes = 3 * R * C * DTYPE_BYTES     # read A, read B, write C
        ai         = flops / dram_bytes
        tflops     = flops / elapsed_s / 1e12
        gb_s       = dram_bytes / elapsed_s / 1e9

        print(
            f"  {label:22s}  {elapsed_s*1e3:7.2f} ms"
            f"  {tflops*1e3:6.2f} GFLOPS  {gb_s:7.1f} GB/s  AI={ai:.4f} F/B"
        )
        rows.append(dict(
            op=label, M=R, K=0, N=C,
            flops=flops, dram_bytes=dram_bytes,
            arithmetic_intensity=round(ai, 6),
            tflops=round(tflops, 6), gb_s=round(gb_s, 3),
            elapsed_ms=round(elapsed_s * 1e3, 3),
        ))

        a.deallocate()
        b.deallocate()

    # Elementwise exp

    print("\n" + "=" * 60)
    print("ELTWISE EXP (BF16)")
    print("=" * 60)

    for label, R, C in ELTWISE_CONFIGS:
        if "exp" not in label:
            continue
        a = make_dram_tensor(R, C, device)

        def run_exp(a=a):
            return ttnn.exp(a, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        elapsed_s = bench_op(run_exp, device)

        flops      = R * C                    # 1 nominal FLOP per element
        dram_bytes = 2 * R * C * DTYPE_BYTES  # read A, write C
        ai         = flops / dram_bytes
        tflops     = flops / elapsed_s / 1e12
        gb_s       = dram_bytes / elapsed_s / 1e9

        print(
            f"  {label:22s}  {elapsed_s*1e3:7.2f} ms"
            f"  {tflops*1e3:6.2f} GFLOPS  {gb_s:7.1f} GB/s  AI={ai:.4f} F/B"
        )
        rows.append(dict(
            op=label, M=R, K=0, N=C,
            flops=flops, dram_bytes=dram_bytes,
            arithmetic_intensity=round(ai, 6),
            tflops=round(tflops, 6), gb_s=round(gb_s, 3),
            elapsed_ms=round(elapsed_s * 1e3, 3),
        ))

        a.deallocate()

    # Write CSV
    fieldnames = [
        "op", "M", "K", "N", "flops", "dram_bytes",
        "arithmetic_intensity", "tflops", "gb_s", "elapsed_ms",
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
