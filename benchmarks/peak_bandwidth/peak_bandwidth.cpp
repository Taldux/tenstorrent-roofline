// Measures peak DRAM bandwidth on Wormhole.
//
// Streams a large buffer from DRAM → L1 → DRAM with zero compute, reading
// and writing concurrently.
//
// CHUNK_TILES=8 / CB depth=128 tiles (16x) is the best config found from an
// earlier chunk-size/CB-depth sweep. Since then, I've done five more
// investigations (read/write isolation, buffer-size scaling, deferred write-barrier
// pipelining matching Tenstorrent's own eltwise kernels, and matching their DRAM page size)
// all failed to match the TTNN add/exp kernels (which give ~200GB/s on this 
// same hardware). I personally am not sure if this is a cause of my own code or maybe
// something additional the Tenstorrent engineers did when writing the TTNN kernels.
// Probably both.

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// one tile = 32x32 × 2 bytes (BF16) = 2048 bytes
// 16408 tiles × 2048 bytes ≈ 32.05 MB, large enough to exceed on-chip buffer
constexpr uint32_t NUM_TILES = 16408;
constexpr uint32_t TILE_SIZE_BYTES = tt::constants::TILE_HW * 2; // TILE_HW = 1024 elements, BF16 = 2 bytes
constexpr uint32_t BUFFER_SIZE = NUM_TILES * TILE_SIZE_BYTES;

constexpr uint32_t CHUNK_TILES = 8;
constexpr uint32_t CB_TILES = 128; // 16x CHUNK_TILES
constexpr uint32_t NUM_ITERATIONS = 100;

constexpr double PEAK_BW_GBS = 288.0;  // single chip

// Row counts to test, clamped to the real grid's row count later
const std::vector<uint32_t> ROW_COUNTS = {1, 2, 4, 8, 999};

struct BenchResult {
    double bw_GBs;
    double elapsed_per_iter_s;
};

BenchResult run_config(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& cq,
    tt::tt_metal::IDevice* device,
    tt::tt_metal::CoreCoord grid,
    uint32_t rows_used)
{
    uint32_t num_cores = grid.x * rows_used;
    tt::tt_metal::CoreRange active_cores(
        tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{grid.x - 1, rows_used - 1});

    uint32_t base_tiles = NUM_TILES / num_cores;
    uint32_t remainder  = NUM_TILES % num_cores;
    std::vector<uint32_t> tiles_for_core(num_cores, base_tiles);
    for (uint32_t i = 0; i < remainder; i++) {
        tiles_for_core[i]++;
    }

    ////////
    // DRAM buffers
    ////////

    tt::tt_metal::distributed::DeviceLocalBufferConfig local_cfg{
        .page_size   = TILE_SIZE_BYTES,
        .buffer_type = tt::tt_metal::BufferType::DRAM
    };
    tt::tt_metal::distributed::ReplicatedBufferConfig rep_cfg{.size = BUFFER_SIZE};

    auto src_mesh_buf = tt::tt_metal::distributed::MeshBuffer::create(rep_cfg, local_cfg, &mesh_device);
    auto dst_mesh_buf = tt::tt_metal::distributed::MeshBuffer::create(rep_cfg, local_cfg, &mesh_device);
    auto* src_buffer = src_mesh_buf->get_device_buffer(tt::tt_metal::distributed::MeshCoordinate{0, 0});
    auto* dst_buffer = dst_mesh_buf->get_device_buffer(tt::tt_metal::distributed::MeshCoordinate{0, 0});

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::tt_metal::CircularBufferConfig cb_cfg =
        tt::tt_metal::CircularBufferConfig(
            CB_TILES * TILE_SIZE_BYTES,
            {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
        .set_page_size(tt::CBIndex::c_0, TILE_SIZE_BYTES);
    tt::tt_metal::CreateCircularBuffer(program, active_cores, cb_cfg);

    ////////
    // Kernels and runtime args
    ////////

    auto reader_kernel = tt::tt_metal::CreateKernel(
        program,
        "benchmarks/peak_bandwidth/kernels/dataflow/reader_dram_stream.cpp",
        active_cores,
        tt::tt_metal::DataMovementConfig{
            .processor   = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc         = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = {CHUNK_TILES}
        });

    auto writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "benchmarks/peak_bandwidth/kernels/dataflow/writer_dram_stream.cpp",
        active_cores,
        tt::tt_metal::DataMovementConfig{
            .processor   = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc         = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = {CHUNK_TILES}
        });

    tt::tt_metal::CreateKernel(
        program,
        "benchmarks/peak_bandwidth/kernels/compute/passthrough.cpp",
        active_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = tt::tt_metal::MathFidelity::LoFi,
            .compile_args  = {}
        });

    uint32_t tile_offset = 0;
    for (uint32_t y = 0; y < rows_used; y++) {
        for (uint32_t x = 0; x < grid.x; x++) {
            tt::tt_metal::CoreCoord core = {x, y};
            uint32_t core_idx = y * grid.x + x;
            uint32_t n_tiles  = tiles_for_core[core_idx];

            tt::tt_metal::SetRuntimeArgs(program, reader_kernel, core,
                {src_buffer->address(), n_tiles, tile_offset});
            tt::tt_metal::SetRuntimeArgs(program, writer_kernel, core,
                {dst_buffer->address(), n_tiles, tile_offset});
            // no runtime args needed for compute kernel

            tile_offset += n_tiles;
        }
    }

    ////////
    // Dispatch
    ////////

    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange device_range(mesh_device.shape());
    workload.add_program(device_range, std::move(program));

    // warmup
    tt::tt_metal::distributed::EnqueueMeshWorkload(cq, workload, false);
    tt::tt_metal::distributed::Finish(cq);

    // timed runs
    auto t_start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < NUM_ITERATIONS; i++) {
        tt::tt_metal::distributed::EnqueueMeshWorkload(cq, workload, false);
    }
    tt::tt_metal::distributed::Finish(cq);
    auto t_end = std::chrono::high_resolution_clock::now();

    // flush per-kernel device profiler results
    tt::tt_metal::detail::ReadDeviceProfilerResults(device);

    double elapsed_s        = std::chrono::duration<double>(t_end - t_start).count();
    double elapsed_per_iter = elapsed_s / NUM_ITERATIONS;
    double bytes_per_iter   = 2.0 * BUFFER_SIZE; // one read pass + one write pass
    double bw_GBs           = (bytes_per_iter / elapsed_per_iter) / 1e9;

    return BenchResult{bw_GBs, elapsed_per_iter};
}

int main(int argc, char** argv) {
    std::string csv_path = (argc > 1) ? argv[1] : "";

    constexpr int device_id = 0;
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    tt::tt_metal::IDevice* device = mesh_device->get_device(0, 0);
    tt::tt_metal::distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    tt::tt_metal::CoreCoord grid = device->compute_with_storage_grid_size();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "=== Peak DRAM Bandwidth Benchmark — Core-Count Sweep ===" << std::endl;
    std::cout << "Full grid:            " << grid.x << "x" << grid.y << " (" << grid.x * grid.y << " cores)" << std::endl;
    std::cout << "Buffer size:          " << BUFFER_SIZE / (1024 * 1024) << " MB per direction" << std::endl;
    std::cout << "Chunk / CB tiles:     " << CHUNK_TILES << " / " << CB_TILES << std::endl;
    std::cout << "Iterations:           " << NUM_ITERATIONS << std::endl;
    std::cout << std::endl;

    std::cout << std::left
              << std::setw(12) << "Rows"
              << std::setw(10) << "Cores"
              << std::setw(14) << "BW (GB/s)"
              << std::setw(14) << "Efficiency"
              << "Time/iter" << std::endl;
    std::cout << std::string(66, '-') << std::endl;

    std::vector<std::pair<uint32_t, BenchResult>> results;
    for (uint32_t requested_rows : ROW_COUNTS) {
        uint32_t rows_used = std::min(requested_rows, static_cast<uint32_t>(grid.y));
        // skip duplicate configs
        if (!results.empty() && results.back().first == rows_used) {
            continue;
        }
        BenchResult r = run_config(*mesh_device, cq, device, grid, rows_used);
        double eff = (r.bw_GBs / PEAK_BW_GBS) * 100.0;

        std::cout << std::left
                  << std::setw(12) << rows_used
                  << std::setw(10) << (grid.x * rows_used)
                  << std::setw(14) << r.bw_GBs
                  << std::setw(14) << (std::to_string(static_cast<int>(eff)) + "%")
                  << r.elapsed_per_iter_s * 1e6 << " us" << std::endl;

        results.push_back({rows_used, r});
    }

    {
        auto best = std::max_element(results.begin(), results.end(),
            [](const auto& a, const auto& b) { return a.second.bw_GBs < b.second.bw_GBs; });
        const auto& [rows_used, r] = *best;
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "Best config:         " << rows_used << " rows (" << grid.x * rows_used << " cores)" << std::endl;
        std::cout << "Time per iteration:  " << r.elapsed_per_iter_s * 1e6 << " us" << std::endl;
        std::cout << "Measured bandwidth:  " << r.bw_GBs << " GB/s" << std::endl;
        std::cout << "Efficiency:          " << (r.bw_GBs / PEAK_BW_GBS) * 100.0 << "%" << std::endl;
    }

    if (!csv_path.empty()) {
        std::ofstream f(csv_path);
        f << "rows_used,cores_used,chunk_tiles,cb_tiles,bw_gbs,efficiency_pct,time_per_iter_us\n";
        for (const auto& [rows_used, r] : results) {
            f << rows_used << ","
              << (grid.x * rows_used) << ","
              << CHUNK_TILES << ","
              << CB_TILES << ","
              << r.bw_GBs << ","
              << (r.bw_GBs / PEAK_BW_GBS) * 100.0 << ","
              << r.elapsed_per_iter_s * 1e6 << "\n";
        }
        std::cout << "\nCSV written to " << csv_path << std::endl;
    }

    mesh_device->close();
    return 0;
}
