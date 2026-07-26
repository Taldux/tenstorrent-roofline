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
  ttnn_ops.csv:       op, M, K, N, flops, dram_bytes, arithmetic_intensity,
                       tflops, gb_s, elapsed_ms
  ttnn_oob_matmul.csv: dtype, base_m, base_k, base_n, m, k, n, grid_x, grid_y,
                       elapsed_ms, tflops, theoretical_peak_tflops, utilization_pct

Usage (inside Apptainer container):
    python3 benchmarks/ttnn_ops/bench_ttnn_ops.py [ops.csv] [oob_matmul.csv]
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


OOB_MATMUL_SHAPES = [
    (64, 64, 64), (64, 128, 128), (64, 128, 256),
    (128, 128, 128), (128, 128, 256), (128, 256, 256),
    (256, 256, 256), (256, 256, 384), (256, 384, 384),
    (384, 384, 384), (384, 384, 512), (384, 512, 512),
    (512, 512, 512),
]

OOB_DTYPES = [
    ("BF16", ttnn.bfloat16, 2.0),
    ("BF8",  ttnn.bfloat8_b, 4.0),
    ("BF4",  ttnn.bfloat4_b, 4.0),
]


def make_dram_tensor(rows, cols, device, dtype=ttnn.bfloat16):
    """Zero-filled tile-layout DRAM tensor"""
    return ttnn.zeros(
        (rows, cols),
        dtype=dtype,
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


def bench_oob_matmul_sweep(device, grid):
    """
    Zero-config ttnn matmul at the same base (m,k,n) shapes as Tenstorrent's
    Out of Box benchmark, scaled by this device's grid size.
    """
    num_cores = grid.x * grid.y
    rows = []

    print("\n" + "=" * 60)
    print("MATMUL — Out of Box sweep (Tenstorrent GEMM_FLOPS-style)")
    print("=" * 60)

    for label, dtype, peak_per_core in OOB_DTYPES:
        peak_tflops = peak_per_core * num_cores
        for base_m, base_k, base_n in OOB_MATMUL_SHAPES:
            m = base_m * grid.y
            k = base_k * grid.x
            n = base_n * grid.x

            in0_t = make_dram_tensor(m, k, device, dtype=dtype)
            in1_t = make_dram_tensor(k, n, device, dtype=dtype)

            def run_oob_matmul(a=in0_t, b=in1_t):
                return a @ b

            elapsed_s = bench_op(run_oob_matmul, device)

            flops    = 2 * m * k * n
            tflops   = flops / elapsed_s / 1e12
            util_pct = tflops / peak_tflops * 100.0

            print(
                f"  {label:5s} m={m:5d} k={k:5d} n={n:5d}  "
                f"{elapsed_s*1e3:8.2f} ms  {tflops:7.2f} TFLOPS  {util_pct:5.1f}%"
            )
            rows.append(dict(
                dtype=label, base_m=base_m, base_k=base_k, base_n=base_n,
                m=m, k=k, n=n, grid_x=grid.x, grid_y=grid.y,
                elapsed_ms=round(elapsed_s * 1e3, 3),
                tflops=round(tflops, 4),
                theoretical_peak_tflops=round(peak_tflops, 2),
                utilization_pct=round(util_pct, 2),
            ))

            in0_t.deallocate()
            in1_t.deallocate()

    return rows


def main():
    out_csv = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parents[2] / "results" / "ttnn_ops.csv"
    )
    oob_csv = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path(__file__).resolve().parents[2] / "results" / "ttnn_oob_matmul.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    oob_csv.parent.mkdir(parents=True, exist_ok=True)

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

    # Out of Box matmul sweep
    oob_rows = bench_oob_matmul_sweep(device, grid)

    # Write CSVs
    fieldnames = [
        "op", "M", "K", "N", "flops", "dram_bytes",
        "arithmetic_intensity", "tflops", "gb_s", "elapsed_ms",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    oob_fieldnames = [
        "dtype", "base_m", "base_k", "base_n", "m", "k", "n",
        "grid_x", "grid_y", "elapsed_ms", "tflops",
        "theoretical_peak_tflops", "utilization_pct",
    ]
    with open(oob_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=oob_fieldnames)
        writer.writeheader()
        writer.writerows(oob_rows)

    print(f"\nResults written to {out_csv}")
    print(f"Out of Box matmul sweep written to {oob_csv}")
    ttnn.close_device(device)
    print("Done.")


if __name__ == "__main__":
    main()
