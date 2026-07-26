#!/bin/bash
set -euo pipefail

THESIS=/csghome/sc323/tt-analysis
TT_METAL=/csghome/sc323/tt-metal
CONTAINER=/csghome/sc323/tt-metalium-ubuntu-22.04-release-amd64_latest-rc.sif
RESULTS=$THESIS/results
BUILD=$THESIS/build

# Off by default, enabling this defines PROFILE_KERNEL for every kernel
# JIT build, which makes the SDK's kernel_profiler.hpp emit a #pragma
# message compiler note per kernel
# Do ENABLE_DEVICE_PROFILER=1 when you actually want profiler output
ENABLE_DEVICE_PROFILER=${ENABLE_DEVICE_PROFILER:-0}

PROFILER_RESULTS_DIR="$RESULTS/profiler"
RAW_PROFILER_DIR="$THESIS/generated/profiler"
PROCESS_OPS_SCRIPT="$TT_METAL/tools/tracy/process_ops_logs.py"

mkdir -p "$RESULTS" "$BUILD"

export TT_METAL_HOME="$TT_METAL"
export TT_METAL_RUNTIME_ROOT="$TT_METAL"
export LD_LIBRARY_PATH="$TT_METAL/build_Release/tt_metal:$TT_METAL/build_Release/tt_stl:$TT_METAL/build_Release/tt_metal/third_party/umd/lib:${LD_LIBRARY_PATH:-}"
export TT_METAL_PROFILER_DIR="$RAW_PROFILER_DIR"

if [ "$ENABLE_DEVICE_PROFILER" = "1" ]; then
    export TT_METAL_DEVICE_PROFILER=1
fi

collect_device_profiler() {
    local bench="$1"
    [ "$ENABLE_DEVICE_PROFILER" = "1" ] || return 0

    mkdir -p "$PROFILER_RESULTS_DIR"

    # Benchmarks call tt_metal::detail::ReadDeviceProfilerResults(device)
    # after timed run, which writes timer records to 
    # $RAW_PROFILER_DIR/.logs/profile_log_device.csv.
    # process_ops_logs.py turns that into a csv of stats
    if [ "$bench" = "ttnn_ops" ] && [ -f "$PROCESS_OPS_SCRIPT" ]; then
        local ops_log_python="python3"
        [ -x /opt/venv/bin/python3 ] && ops_log_python=/opt/venv/bin/python3
        (
            cd "$THESIS"
            PYTHONPATH="$TT_METAL/tools:${PYTHONPATH:-}" \
                "$ops_log_python" "$PROCESS_OPS_SCRIPT" -o "$RAW_PROFILER_DIR"
        ) > "$PROFILER_RESULTS_DIR/${bench}_process_ops_logs.txt" 2>&1 || true
    fi

    if [ -d "$RAW_PROFILER_DIR" ]; then
        local dst="$PROFILER_RESULTS_DIR/${bench}_raw"
        rm -rf "$dst"
        cp -a "$RAW_PROFILER_DIR" "$dst"

        if [ "$bench" = "ttnn_ops" ]; then
            local newest
            newest="$(find "$RAW_PROFILER_DIR" -name 'ops_perf_results*.csv' -printf '%T@ %p\n' 2>/dev/null \
                | sort -rn | head -n1 | cut -d' ' -f2-)"
            if [ -n "$newest" ] && [ -f "$newest" ]; then
                cp -a "$newest" "$PROFILER_RESULTS_DIR/${bench}_ops_perf_results.csv"
            else
                echo "[profiling] No ops_perf_results*.csv found under $RAW_PROFILER_DIR for $bench" \
                    >> "$PROFILER_RESULTS_DIR/${bench}_process_ops_logs.txt"
            fi
        fi
    fi
}

run_with_profiling() {
    local bench="$1"
    local out_file="$2"
    shift 2

    mkdir -p "$PROFILER_RESULTS_DIR"
    [ "$ENABLE_DEVICE_PROFILER" = "1" ] && rm -rf "$RAW_PROFILER_DIR"

    set +e
    "$@" 2>&1 | tee "$out_file"
    local cmd_status=${PIPESTATUS[0]}
    set -e

    collect_device_profiler "$bench"

    return "$cmd_status"
}

run_full() {
    cd "$THESIS"

    if [ "$ENABLE_DEVICE_PROFILER" = "1" ]; then
        echo "Device profiler enabled"
    fi

    echo 'Configuring with CMake...'
    cmake -S "$THESIS" -B "$BUILD" \
        -G Ninja \
        -DCMAKE_CXX_COMPILER=clang++-20 \
        -DCMAKE_BUILD_TYPE=Release

    echo 'Building...'
    cmake --build "$BUILD"

    echo 'Running peak_bandwidth...'
    run_with_profiling "peak_bandwidth" "$RESULTS/peak_bandwidth.txt" \
        "$BUILD/peak_bandwidth" "$RESULTS/peak_bandwidth.csv"

    echo 'Running peak_flops...'
    run_with_profiling "peak_flops" "$RESULTS/peak_flops.txt" \
        "$BUILD/peak_flops" "$RESULTS/peak_flops_sweep.csv" "$RESULTS/peak_flops_tileswp.csv" \
        "$RESULTS/peak_flops_phasesweep.csv" "$RESULTS/peak_flops_tileswp_hifi4.csv" \
        "$RESULTS/peak_flops_matrix.csv"

    echo 'Running l1_bandwidth...'
    run_with_profiling "l1_bandwidth" "$RESULTS/l1_bandwidth.txt" "$BUILD/l1_bandwidth"

    echo 'Running TTNN Python op benchmarks...'
    run_with_profiling "ttnn_ops" "$RESULTS/ttnn_ops.txt" \
        env -u TT_METAL_DEVICE_PROFILER PYTHONPATH="" /opt/venv/bin/python3 "$THESIS/benchmarks/ttnn_ops/bench_ttnn_ops.py" \
                "$RESULTS/ttnn_ops.csv" "$RESULTS/ttnn_oob_matmul.csv" \
        || echo "Warning: TTNN Python benchmark failed — skipping (check ttnn installation)"
}

if [ "${RUN_BENCHMARKS_INNER:-0}" = "1" ]; then
    run_full
    echo "All benchmarks complete!"
    echo "Results saved to $RESULTS/"
    exit 0
fi

APPTAINER_BIN="$(command -v apptainer || command -v singularity || true)"
if [ -z "$APPTAINER_BIN" ]; then
    echo "Apptainer command not found on PATH in this shell."
    echo "Load Apptainer in the host environment, or run inside container with RUN_BENCHMARKS_INNER=1."
    exit 127
fi

if [ ! -f "$CONTAINER" ]; then
    echo "Container image not found at $CONTAINER"
    echo "Pull it first on headnode:"
    echo "  apptainer pull $CONTAINER docker://ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc"
    exit 1
fi

"$APPTAINER_BIN" exec \
    --bind /csghome/sc323:/csghome/sc323 \
    "$CONTAINER" \
    bash -lc "RUN_BENCHMARKS_INNER=1 \"$THESIS/run_benchmarks.sh\""
