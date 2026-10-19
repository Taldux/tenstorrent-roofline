#!/bin/bash
# Cluster batch 1, everything with the device profiler off:
#   1. rerun of the existing benchmarks
#   2. peak_flops_block (matmul_block vs matmul_tiles)
#   3. out-of-box matmul sweep with different inputs
#   4. repeats of the headline benchmarks (run_repeats.sh)
# Results go to results_<date>/, never results/. A failed step doesn't stop the others.
#
# Usage on headnode:
#   srun -w csg-torrent -p torrent --gres=tt:n300s:4 --pty bash -lc \
#     'cd /csghome/sc323/tt-analysis && bash run_batch1.sh'
set -uo pipefail

THESIS=/csghome/sc323/tt-analysis
TT_METAL=/csghome/sc323/tt-metal
CONTAINER=/csghome/sc323/tt-metalium-ubuntu-22.04-release-amd64_latest-rc.sif
BUILD=$THESIS/build
OUT=${OUT:-$THESIS/results_$(date +%Y-%m-%d)}
REPEATS=${REPEATS:-10}

if [ "$OUT" = "$THESIS/results" ] || [ "$OUT" = "$THESIS/results/" ]; then
    echo "Refusing to write into $THESIS/results"
    exit 1
fi

# step <name> <cmd...>: runs cmd, saves output to $OUT/<name>.txt, records exit status
step() {
    local name="$1"
    shift
    echo
    echo "=== $name ==="
    "$@" 2>&1 | tee "$OUT/${name}.txt"
    local status=${PIPESTATUS[0]}
    echo "$name,$status" >> "$OUT/status.csv"
    [ "$status" = "0" ] || echo "WARNING: $name exited with $status"
}

run_full() {
    cd "$THESIS"
    export TT_METAL_HOME="$TT_METAL"
    export TT_METAL_RUNTIME_ROOT="$TT_METAL"
    export LD_LIBRARY_PATH="$TT_METAL/build_Release/tt_metal:$TT_METAL/build_Release/tt_stl:$TT_METAL/build_Release/tt_metal/third_party/umd/lib:${LD_LIBRARY_PATH:-}"
    unset TT_METAL_DEVICE_PROFILER

    mkdir -p "$OUT" "$BUILD"
    echo "step,exit_status" > "$OUT/status.csv"
    {
        echo "date: $(date -Is)"
        echo "host: $(hostname)"
        echo "tt-metal commit: $(git -C "$TT_METAL" rev-parse HEAD 2>/dev/null || echo NA)"
        echo "tt-metal describe: $(git -C "$TT_METAL" describe --tags 2>/dev/null || echo NA)"
    } > "$OUT/run_info.txt"
    echo "Output: $OUT"

    echo 'Configuring with CMake...'
    cmake -S "$THESIS" -B "$BUILD" -G Ninja \
        -DCMAKE_CXX_COMPILER=clang++-20 -DCMAKE_BUILD_TYPE=Release \
        > "$OUT/cmake_configure.txt" 2>&1 || { echo "CMake configure failed, see $OUT/cmake_configure.txt"; exit 1; }

    # build one target at a time so a broken new one doesn't block the rest
    for target in peak_bandwidth peak_flops l1_bandwidth peak_flops_block; do
        step "build_${target}" cmake --build "$BUILD" --target "$target"
    done

    # 1. existing benchmarks
    step peak_bandwidth "$BUILD/peak_bandwidth" "$OUT/peak_bandwidth.csv"
    step peak_flops "$BUILD/peak_flops" \
        "$OUT/peak_flops_sweep.csv" "$OUT/peak_flops_tileswp.csv" \
        "$OUT/peak_flops_phasesweep.csv" "$OUT/peak_flops_tileswp_hifi4.csv" \
        "$OUT/peak_flops_matrix.csv"
    step l1_bandwidth "$BUILD/l1_bandwidth"
    step ttnn_ops env PYTHONPATH="" /opt/venv/bin/python3 \
        "$THESIS/benchmarks/ttnn_ops/bench_ttnn_ops.py" \
        "$OUT/ttnn_ops.csv" "$OUT/ttnn_oob_matmul.csv"

    # 2. matmul_block vs matmul_tiles
    step peak_flops_block "$BUILD/peak_flops_block" \
        "$OUT/peak_flops_block_matrix.csv" "$OUT/peak_flops_block_tileswp.csv" \
        "$OUT/peak_flops_block_correctness.csv"

    # 3. out-of-box sweep with different inputs
    step ttnn_oob_inputs env PYTHONPATH="" /opt/venv/bin/python3 \
        "$THESIS/benchmarks/ttnn_oob_inputs/bench_oob_inputs.py" \
        "$OUT/ttnn_oob_inputs.csv"

    # 4. repeats
    step repeats env RUN_REPEATS_INNER=1 REPEATS="$REPEATS" OUT="$OUT/repeats" \
        bash "$THESIS/run_repeats.sh"

    echo
    echo "Batch 1 done. Step status:"
    cat "$OUT/status.csv"
}

if [ "${RUN_BATCH_INNER:-0}" = "1" ]; then
    run_full
    exit 0
fi

APPTAINER_BIN="$(command -v apptainer || command -v singularity || true)"
if [ -z "$APPTAINER_BIN" ]; then
    echo "Apptainer not found; run inside the container with RUN_BATCH_INNER=1."
    exit 127
fi

"$APPTAINER_BIN" exec \
    --bind /csghome/sc323:/csghome/sc323 \
    "$CONTAINER" \
    bash -lc "RUN_BATCH_INNER=1 OUT=\"$OUT\" REPEATS=$REPEATS \"$THESIS/run_batch1.sh\""
