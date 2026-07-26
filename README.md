# TT Wormhole n300: Empirical Roofline Model

Bachelor thesis project building an empirical roofline model for the Tenstorrent
Wormhole n300 accelerator. (Somewhat) handwritten TT-Metal kernels measure peak DRAM
bandwidth, L1/NoC bandwidth, and matmul compute throughput (TFLOPS) across
numerical formats and fidelities, and a Python benchmark that measures the same
things through TTNN's own operators for comparison.

## Repository layout

```
tt-analysis/
├── CMakeLists.txt       - builds 3 C++ benchmarks against a local tt-metal build
├── run_benchmarks.sh    - builds and runs everything, writes results/ (+ optional device profiler data)
├── benchmarks/          - benchmarks source
├── plots/                - plot results
└── results/              - benchmark output
```

## Requirements

- TT-Metal SDK v0.72.0, built locally (`CMakeLists.txt`'s `TT_METAL_HOME` cache
  variable points at the build, maybe edit it for your own machine?)
- `clang++-20`, CMake ≥3.20, Ninja
- Apptainer to run inside the container image referenced by
  `run_benchmarks.sh` (adjust the `CONTAINER` path at the top of that script
  for your own image, or run the built binaries directly on a host that
  already has the SDK available)
- Python 3 + `ttnn` + `torch` (only needed for `benchmarks/ttnn_ops/bench_ttnn_ops.py`)

## Running

On the cluster, everything is launched with a single

```
srun -w csg-torrent -p torrent --gres=tt:n300s:4 --pty bash -lc 'cd /csghome/sc323/tt-analysis && bash run_benchmarks.sh'
```

With kernel profiler enabled:

```
srun -w csg-torrent -p torrent --gres=tt:n300s:4 --pty bash -lc 'cd /csghome/sc323/tt-analysis && ENABLE_DEVICE_PROFILER=1 bash run_benchmarks.sh'
```

Running this on another machine, you will have to edit `THESIS`/`TT_METAL`/`CONTAINER`
in `run_benchmarks.sh` to point at your own paths, or skip the
script entirely and run the built binaries (plus
`python3 benchmarks/ttnn_ops/bench_ttnn_ops.py`) directly on a host that
already has the SDK set up.
