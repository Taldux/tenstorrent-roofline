#!/bin/bash
# Runs each headline benchmark REPEATS times with the device profiler off
# and collects the clock from each log's "frequency=" line into frequency.csv
#
# Usage on csg-torrent (needs an existing build in $BUILD):
#   OUT=/csghome/sc323/tt-analysis/results_<date>/repeats bash run_repeats.sh
set -uo pipefail

THESIS=/csghome/sc323/tt-analysis
TT_METAL=/csghome/sc323/tt-metal
CONTAINER=/csghome/sc323/tt-metalium-ubuntu-22.04-release-amd64_latest-rc.sif
BUILD=$THESIS/build
REPEATS=${REPEATS:-10}
OUT=${OUT:-$THESIS/results_$(date +%Y-%m-%d)/repeats}

if [ "$OUT" = "$THESIS/results" ] || [ "$OUT" = "$THESIS/results/" ]; then
    echo "Refusing to write into $THESIS/results"
    exit 1
fi

# run_repeat <name> <rep> <cmd...>
run_repeat() {
    local name="$1" rep="$2"
    shift 2
    local log="$OUT/${name}_${rep}.txt"

    "$@" > "$log" 2>&1
    local status=$?

    local freq
    freq="$(grep -o 'frequency=[0-9.]*' "$log" | head -n1 | cut -d= -f2)"
    echo "$name,$rep,$status,${freq:-NA}" >> "$OUT/frequency.csv"
    echo "  $name rep $rep: exit $status, frequency=${freq:-NA}"
}

run_all() {
    cd "$THESIS"
    export TT_METAL_HOME="$TT_METAL"
    export TT_METAL_RUNTIME_ROOT="$TT_METAL"
    export LD_LIBRARY_PATH="$TT_METAL/build_Release/tt_metal:$TT_METAL/build_Release/tt_stl:$TT_METAL/build_Release/tt_metal/third_party/umd/lib:${LD_LIBRARY_PATH:-}"
    unset TT_METAL_DEVICE_PROFILER

    mkdir -p "$OUT"
    echo "bench,rep,exit_status,frequency_ghz" > "$OUT/frequency.csv"
    echo "Repeats: $REPEATS, output: $OUT"

    for rep in $(seq 1 "$REPEATS"); do
        echo "=== Repeat $rep/$REPEATS ==="
        run_repeat peak_bandwidth "$rep" \
            "$BUILD/peak_bandwidth" "$OUT/peak_bandwidth_${rep}.csv"
        run_repeat peak_flops "$rep" \
            "$BUILD/peak_flops" "$OUT/peak_flops_sweep_${rep}.csv"
        run_repeat l1_bandwidth "$rep" \
            "$BUILD/l1_bandwidth"
        run_repeat ttnn_ops "$rep" \
            env PYTHONPATH="" /opt/venv/bin/python3 "$THESIS/benchmarks/ttnn_ops/bench_ttnn_ops.py" \
                "$OUT/ttnn_ops_${rep}.csv" "$OUT/ttnn_oob_matmul_${rep}.csv"
    done
}

if [ "${RUN_REPEATS_INNER:-0}" = "1" ]; then
    run_all
    exit 0
fi

APPTAINER_BIN="$(command -v apptainer || command -v singularity || true)"
if [ -z "$APPTAINER_BIN" ]; then
    echo "Apptainer not found; run inside the container with RUN_REPEATS_INNER=1."
    exit 127
fi

"$APPTAINER_BIN" exec \
    --bind /csghome/sc323:/csghome/sc323 \
    "$CONTAINER" \
    bash -lc "RUN_REPEATS_INNER=1 REPEATS=$REPEATS OUT=\"$OUT\" \"$THESIS/run_repeats.sh\""
