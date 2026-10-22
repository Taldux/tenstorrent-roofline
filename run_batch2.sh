#!/bin/bash
# Cluster batch 2, device profiler off:
#   dram_add (handwritten add vs copy at TTNN's size), REPEATS times
#   bench_ttnn_ops.py in the same session, for the TTNN add/exp numbers
#   peak_flops (headline configs only) MATMUL_REPEATS times, to check for slow outlier runs
# Results go to results_<date>_batch2/, never results/.
#
# Usage on headnode:
#   srun -w csg-torrent -p torrent --gres=tt:n300s:4 --time=01:00:00 --pty bash -lc \
#     'cd /csghome/sc323/tt-analysis && bash run_batch2.sh'
set -uo pipefail

THESIS=/csghome/sc323/tt-analysis
TT_METAL=/csghome/sc323/tt-metal
CONTAINER=/csghome/sc323/tt-metalium-ubuntu-22.04-release-amd64_latest-rc.sif
BUILD=$THESIS/build
OUT=${OUT:-$THESIS/results_$(date +%Y-%m-%d)_batch2}
REPEATS=${REPEATS:-5}
MATMUL_REPEATS=${MATMUL_REPEATS:-10}

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
    echo "Output: $OUT"

    cmake -S "$THESIS" -B "$BUILD" -G Ninja \
        -DCMAKE_CXX_COMPILER=clang++-20 -DCMAKE_BUILD_TYPE=Release \
        > "$OUT/cmake_configure.txt" 2>&1 || { echo "CMake configure failed, see $OUT/cmake_configure.txt"; exit 1; }
    step build_dram_add cmake --build "$BUILD" --target dram_add
    step build_peak_flops cmake --build "$BUILD" --target peak_flops

    for rep in $(seq 1 "$REPEATS"); do
        step "dram_add_${rep}" "$BUILD/dram_add" "$OUT/dram_add_${rep}.csv"
    done

    step ttnn_ops env PYTHONPATH="" /opt/venv/bin/python3 \
        "$THESIS/benchmarks/ttnn_ops/bench_ttnn_ops.py" \
        "$OUT/ttnn_ops.csv" "$OUT/ttnn_oob_matmul.csv"

    # matmul repeats, each a separate process like in run_repeats.sh
    echo "rep,frequency_ghz" > "$OUT/peak_flops_frequency.csv"
    for rep in $(seq 1 "$MATMUL_REPEATS"); do
        step "peak_flops_${rep}" "$BUILD/peak_flops" "$OUT/peak_flops_sweep_${rep}.csv"
        freq="$(grep -o 'frequency=[0-9.]*' "$OUT/peak_flops_${rep}.txt" | head -n1 | cut -d= -f2)"
        echo "$rep,${freq:-NA}" >> "$OUT/peak_flops_frequency.csv"
    done

    echo
    echo "Batch 2 done. Step status:"
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
    bash -lc "RUN_BATCH_INNER=1 OUT=\"$OUT\" REPEATS=$REPEATS MATMUL_REPEATS=$MATMUL_REPEATS \"$THESIS/run_batch2.sh\""
