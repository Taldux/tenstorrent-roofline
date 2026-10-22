// DRAM bandwidth for a handwritten elementwise add, at the same size and
// access pattern as the TTNN add in bench_ttnn_ops.py (8192x8192 BF16, two
// inputs read, one output written), on the full grid.
//
// Variants:
//   copy_128MiB  peak_bandwidth's unchanged copy kernels (1 read, 1 write), at 128 MiB
//   add_128MiB   C = A + B (2 reads, 1 write), real add_tiles() in the compute kernel
//   *_tt_noc     same, but reader on NOC_0 and writer on NOC_1, like tt-metal's
//                ReaderDataMovementConfig / WriterDataMovementConfig on Wormhole
//                (the other variants use RISCV_x_default, which is the opposite)
//
// GB/s counts every byte read and written, same as bench_ttnn_ops.py.
//
// Usage: dram_add [out.csv]

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// 8192 x 8192 BF16 = 65536 tiles = 128 MiB per tensor
constexpr uint32_t NUM_TILES = (8192 * 8192) / tt::constants::TILE_HW;
constexpr uint32_t TILE_SIZE_BYTES = tt::constants::TILE_HW * 2;
constexpr uint32_t BUFFER_SIZE = NUM_TILES * TILE_SIZE_BYTES;

// same chunk / CB depth as peak_bandwidth
constexpr uint32_t CHUNK_TILES = 8;
constexpr uint32_t CB_TILES = 128;
constexpr uint32_t NUM_ITERATIONS = 100;

constexpr double PEAK_BW_GBS = 288.0; // single chip

enum class Variant
{
    Copy,
    Add
};

struct BenchResult
{
    double bw_GBs;
    double elapsed_per_iter_s;
    double bytes_per_iter;
};

tt::tt_metal::CircularBufferConfig make_cb(tt::CBIndex index)
{
    return tt::tt_metal::CircularBufferConfig(
               CB_TILES * TILE_SIZE_BYTES, {{index, tt::DataFormat::Float16_b}})
        .set_page_size(index, TILE_SIZE_BYTES);
}

BenchResult run_config(
    tt::tt_metal::distributed::MeshDevice &mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue &cq,
    tt::tt_metal::IDevice *device,
    Variant variant,
    bool tt_noc)
{
    tt::tt_metal::CoreCoord grid = device->compute_with_storage_grid_size();
    uint32_t num_cores = grid.x * grid.y;
    tt::tt_metal::CoreRange all_cores(
        tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{grid.x - 1, grid.y - 1});

    uint32_t base_tiles = NUM_TILES / num_cores;
    uint32_t remainder = NUM_TILES % num_cores;
    std::vector<uint32_t> tiles_for_core(num_cores, base_tiles);
    for (uint32_t i = 0; i < remainder; i++)
    {
        tiles_for_core[i]++;
    }

    ////////
    // DRAM buffers
    ////////

    tt::tt_metal::distributed::DeviceLocalBufferConfig local_cfg{
        .page_size = TILE_SIZE_BYTES,
        .buffer_type = tt::tt_metal::BufferType::DRAM};
    tt::tt_metal::distributed::ReplicatedBufferConfig rep_cfg{.size = BUFFER_SIZE};

    auto a_mesh_buf = tt::tt_metal::distributed::MeshBuffer::create(rep_cfg, local_cfg, &mesh_device);
    auto b_mesh_buf = tt::tt_metal::distributed::MeshBuffer::create(rep_cfg, local_cfg, &mesh_device);
    auto c_mesh_buf = tt::tt_metal::distributed::MeshBuffer::create(rep_cfg, local_cfg, &mesh_device);
    auto *a_buffer = a_mesh_buf->get_device_buffer(tt::tt_metal::distributed::MeshCoordinate{0, 0});
    auto *b_buffer = b_mesh_buf->get_device_buffer(tt::tt_metal::distributed::MeshCoordinate{0, 0});
    auto *c_buffer = c_mesh_buf->get_device_buffer(tt::tt_metal::distributed::MeshCoordinate{0, 0});

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    ////////
    // Kernels
    ////////

    tt::tt_metal::KernelHandle reader_kernel, writer_kernel, compute_kernel;

    tt::tt_metal::NOC reader_noc = tt_noc ? tt::tt_metal::NOC::NOC_0 : tt::tt_metal::NOC::RISCV_1_default;
    tt::tt_metal::NOC writer_noc = tt_noc ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::RISCV_0_default;

    if (variant == Variant::Copy)
    {
        tt::tt_metal::CreateCircularBuffer(program, all_cores, make_cb(tt::CBIndex::c_0));

        reader_kernel = tt::tt_metal::CreateKernel(
            program,
            "benchmarks/peak_bandwidth/kernels/dataflow/reader_dram_stream.cpp",
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = reader_noc,
                .compile_args = {CHUNK_TILES}});
        writer_kernel = tt::tt_metal::CreateKernel(
            program,
            "benchmarks/peak_bandwidth/kernels/dataflow/writer_dram_stream.cpp",
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = writer_noc,
                .compile_args = {CHUNK_TILES}});
        compute_kernel = tt::tt_metal::CreateKernel(
            program,
            "benchmarks/peak_bandwidth/kernels/compute/passthrough.cpp",
            all_cores,
            tt::tt_metal::ComputeConfig{.math_fidelity = tt::tt_metal::MathFidelity::LoFi});
    }
    else
    {
        tt::tt_metal::CreateCircularBuffer(program, all_cores, make_cb(tt::CBIndex::c_0));
        tt::tt_metal::CreateCircularBuffer(program, all_cores, make_cb(tt::CBIndex::c_1));
        tt::tt_metal::CreateCircularBuffer(program, all_cores, make_cb(tt::CBIndex::c_16));

        reader_kernel = tt::tt_metal::CreateKernel(
            program,
            "benchmarks/dram_add/kernels/dataflow/reader_two_streams.cpp",
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = reader_noc,
                .compile_args = {CHUNK_TILES}});
        writer_kernel = tt::tt_metal::CreateKernel(
            program,
            "benchmarks/dram_add/kernels/dataflow/writer_out.cpp",
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = writer_noc,
                .compile_args = {CHUNK_TILES}});
        compute_kernel = tt::tt_metal::CreateKernel(
            program,
            "benchmarks/dram_add/kernels/compute/add.cpp",
            all_cores,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = tt::tt_metal::MathFidelity::LoFi,
                .compile_args = {CHUNK_TILES}});
    }

    ////////
    // Runtime args
    ////////

    uint32_t tile_offset = 0;
    for (uint32_t y = 0; y < grid.y; y++)
    {
        for (uint32_t x = 0; x < grid.x; x++)
        {
            tt::tt_metal::CoreCoord core = {x, y};
            uint32_t n_tiles = tiles_for_core[y * grid.x + x];

            if (variant == Variant::Copy)
            {
                tt::tt_metal::SetRuntimeArgs(program, reader_kernel, core,
                                             {a_buffer->address(), n_tiles, tile_offset});
            }
            else
            {
                tt::tt_metal::SetRuntimeArgs(program, reader_kernel, core,
                                             {a_buffer->address(), b_buffer->address(), n_tiles, tile_offset});
                tt::tt_metal::SetRuntimeArgs(program, compute_kernel, core, {n_tiles});
            }
            tt::tt_metal::SetRuntimeArgs(program, writer_kernel, core,
                                         {c_buffer->address(), n_tiles, tile_offset});

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
    for (uint32_t i = 0; i < NUM_ITERATIONS; i++)
    {
        tt::tt_metal::distributed::EnqueueMeshWorkload(cq, workload, false);
    }
    tt::tt_metal::distributed::Finish(cq);
    auto t_end = std::chrono::high_resolution_clock::now();

    // flush per-kernel device profiler results
    tt::tt_metal::detail::ReadDeviceProfilerResults(device);

    double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
    double elapsed_per_iter = elapsed_s / NUM_ITERATIONS;
    double streams = (variant == Variant::Copy) ? 2.0 : 3.0;
    double bytes_per_iter = streams * BUFFER_SIZE;
    double bw_GBs = (bytes_per_iter / elapsed_per_iter) / 1e9;

    return BenchResult{bw_GBs, elapsed_per_iter, bytes_per_iter};
}

int main(int argc, char **argv)
{
    std::string csv_path = (argc > 1) ? argv[1] : "";

    constexpr int device_id = 0;
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    tt::tt_metal::IDevice *device = mesh_device->get_device(0, 0);
    tt::tt_metal::distributed::MeshCommandQueue &cq = mesh_device->mesh_command_queue();

    tt::tt_metal::CoreCoord grid = device->compute_with_storage_grid_size();
    uint32_t num_cores = grid.x * grid.y;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "=== DRAM Add Benchmark (8192x8192 BF16) ===" << std::endl;
    std::cout << "Cores:            " << num_cores << " (" << grid.x << "x" << grid.y << ")" << std::endl;
    std::cout << "Tensor size:      " << BUFFER_SIZE / (1024 * 1024) << " MiB" << std::endl;
    std::cout << "Chunk / CB tiles: " << CHUNK_TILES << " / " << CB_TILES << std::endl;
    std::cout << "Iterations:       " << NUM_ITERATIONS << std::endl
              << std::endl;

    std::cout << std::left
              << std::setw(22) << "Variant"
              << std::setw(14) << "BW (GB/s)"
              << std::setw(14) << "Efficiency"
              << "Time/iter" << std::endl;
    std::cout << std::string(66, '-') << std::endl;

    struct VariantConfig
    {
        std::string name;
        Variant variant;
        bool tt_noc;
    };
    const std::vector<VariantConfig> variants = {
        {"copy_128MiB", Variant::Copy, false},
        {"add_128MiB", Variant::Add, false},
        {"copy_128MiB_tt_noc", Variant::Copy, true},
        {"add_128MiB_tt_noc", Variant::Add, true},
    };

    std::vector<std::pair<std::string, BenchResult>> results;
    for (const auto &[name, variant, tt_noc] : variants)
    {
        BenchResult r = run_config(*mesh_device, cq, device, variant, tt_noc);
        std::cout << std::left
                  << std::setw(22) << name
                  << std::setw(14) << r.bw_GBs
                  << std::setw(14) << (r.bw_GBs / PEAK_BW_GBS * 100.0)
                  << r.elapsed_per_iter_s * 1e6 << " us" << std::endl;
        results.push_back({name, r});
    }

    if (!csv_path.empty())
    {
        std::ofstream f(csv_path);
        f << "variant,cores_used,tensor_mib,bytes_per_iter,bw_gbs,efficiency_pct,time_per_iter_us\n";
        for (const auto &[name, r] : results)
        {
            f << name << ","
              << num_cores << ","
              << BUFFER_SIZE / (1024 * 1024) << ","
              << r.bytes_per_iter << ","
              << r.bw_GBs << ","
              << r.bw_GBs / PEAK_BW_GBS * 100.0 << ","
              << r.elapsed_per_iter_s * 1e6 << "\n";
        }
        std::cout << "\nCSV written to " << csv_path << std::endl;
    }

    mesh_device->close();
    return 0;
}
