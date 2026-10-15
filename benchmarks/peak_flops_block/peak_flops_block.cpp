// matmul_block() vs matmul_tiles() compute throughput.
//
// Same setup as peak_flops.cpp, but runs several compute kernels:
//   tiles      the unchanged peak_flops kernel, as a baseline
//   block_HxW  matmul_block() with an H x W output block (1x1 = no reuse)
//
// Usage: peak_flops_block [matrix.csv] [tileswp.csv] [correctness.csv]
// Pass "-" to skip one.

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

constexpr uint32_t INNER_LOOP = 10000;
constexpr uint32_t DRAIN_INTERVAL = 0;
constexpr uint32_t NUM_ITERATIONS = 10;
constexpr uint32_t MAX_DST_TILES = 8;

// Per core matrix engine peak, by math fidelity, at the nominal 1 GHz AI clock
double per_core_peak_tflops(tt::tt_metal::MathFidelity fidelity)
{
    switch (fidelity)
    {
    case tt::tt_metal::MathFidelity::LoFi:
        return 4.0;
    case tt::tt_metal::MathFidelity::HiFi2:
        return 2.0;
    case tt::tt_metal::MathFidelity::HiFi3:
        return 4.0 / 3.0;
    case tt::tt_metal::MathFidelity::HiFi4:
        return 1.0;
    default:
        return 0.0;
    }
}

std::string fidelity_name(tt::tt_metal::MathFidelity f)
{
    switch (f)
    {
    case tt::tt_metal::MathFidelity::LoFi:
        return "LoFi";
    case tt::tt_metal::MathFidelity::HiFi2:
        return "HiFi2";
    case tt::tt_metal::MathFidelity::HiFi3:
        return "HiFi3";
    case tt::tt_metal::MathFidelity::HiFi4:
        return "HiFi4";
    default:
        return "Other";
    }
}

std::string format_name(tt::DataFormat df)
{
    switch (df)
    {
    case tt::DataFormat::Bfp4_b:
        return "BF4";
    case tt::DataFormat::Bfp8_b:
        return "BF8";
    case tt::DataFormat::Float16_b:
        return "BF16";
    default:
        return "Other";
    }
}

// sub_h = sub_w = 0 means the matmul_tiles baseline
struct KernelVariant
{
    std::string name;
    uint32_t sub_h;
    uint32_t sub_w;
    bool is_tiles() const { return sub_h == 0; }
};

// clang-format off
const std::vector<KernelVariant> KERNELS = {
    {"tiles",     0, 0},
    {"block_1x1", 1, 1},
    {"block_1x8", 1, 8},
    {"block_2x4", 2, 4},
    {"block_4x2", 4, 2},
};

const std::vector<KernelVariant> TILE_SWEEP_KERNELS = {
    {"tiles",     0, 0},
    {"block_1x8", 1, 8},
    {"block_2x4", 2, 4},
};
// clang-format on

struct FormatConfig
{
    tt::DataFormat df;
    tt::tt_metal::MathFidelity fidelity;
    uint32_t Mt, Kt, Nt;
};

// Same 3x3 format x fidelity matrix as peak_flops
// clang-format off
const std::vector<FormatConfig> FIDELITY_MATRIX = {
    {tt::DataFormat::Bfp4_b,    tt::tt_metal::MathFidelity::LoFi,  16, 16, 8},
    {tt::DataFormat::Bfp4_b,    tt::tt_metal::MathFidelity::HiFi2, 16, 16, 8},
    {tt::DataFormat::Bfp4_b,    tt::tt_metal::MathFidelity::HiFi4, 16, 16, 8},
    {tt::DataFormat::Bfp8_b,    tt::tt_metal::MathFidelity::LoFi,  16, 16, 8},
    {tt::DataFormat::Bfp8_b,    tt::tt_metal::MathFidelity::HiFi2, 16, 16, 8},
    {tt::DataFormat::Bfp8_b,    tt::tt_metal::MathFidelity::HiFi4, 16, 16, 8},
    {tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::LoFi,  16, 16, 8},
    {tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::HiFi2, 16, 16, 8},
    {tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::HiFi4, 16, 16, 8},
};
// clang-format on

// Mt=Kt=T for the BF16 LoFi tile sweep, Nt fixed at 8
const std::vector<uint32_t> TILE_SWEEP_T = {1, 2, 4, 6, 8, 10, 12, 14, 16};

bool fits(const KernelVariant &kv, const FormatConfig &cfg)
{
    if (kv.is_tiles())
    {
        return cfg.Nt <= MAX_DST_TILES;
    }
    return cfg.Mt % kv.sub_h == 0 && cfg.Nt % kv.sub_w == 0;
}

// Correctness check inputs: every element of A tile (m,k) is a_val(m,k), every
// element of B tile (k,n) is b_val(k,n), so C tile (m,n) = 32 * sum_k a*b everywhere
float a_val(uint32_t m, uint32_t k) { return 0.5f * (1 + (3 * m + k) % 4); }
float b_val(uint32_t k, uint32_t n) { return 0.5f * (1 + (k + 5 * n) % 4); }

uint32_t bf16_bits(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u >> 16;
}

float bf16_to_float(uint32_t bits)
{
    uint32_t u = bits << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

struct BenchResult
{
    double tflops;
    double elapsed_per_iter_s;
    double ns_per_tile_matmul; // per core, one 32x32x32 tile product
};

struct RunOptions
{
    uint32_t inner_loop = INNER_LOOP;
    uint32_t num_iterations = NUM_ITERATIONS;
    bool init_inputs = false;                // write the correctness inputs into A and B
    std::vector<uint32_t> *output = nullptr; // if set, write C to DRAM and read it back
};

BenchResult run_config(
    tt::tt_metal::distributed::MeshDevice &mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue &cq,
    tt::tt_metal::IDevice *device,
    const KernelVariant &kv,
    const FormatConfig &cfg,
    const RunOptions &opt)
{
    const uint32_t Mt = cfg.Mt, Kt = cfg.Kt, Nt = cfg.Nt;
    const uint32_t TILE_SIZE_BYTES = tt::tile_size(cfg.df);

    tt::tt_metal::CoreCoord grid = device->compute_with_storage_grid_size();
    uint32_t num_cores = grid.x * grid.y;
    tt::tt_metal::CoreRange all_cores(
        tt::tt_metal::CoreCoord{0, 0},
        tt::tt_metal::CoreCoord{grid.x - 1, grid.y - 1});

    ////////
    // DRAM and circular buffers
    ////////

    uint32_t size_A = Mt * Kt * TILE_SIZE_BYTES;
    uint32_t size_B = Kt * Nt * TILE_SIZE_BYTES;
    uint32_t size_C = num_cores * Mt * Nt * TILE_SIZE_BYTES;

    tt::tt_metal::distributed::DeviceLocalBufferConfig lcfg{
        .page_size = TILE_SIZE_BYTES, .buffer_type = tt::tt_metal::BufferType::DRAM};

    auto buf_A_mesh = tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{.size = size_A}, lcfg, &mesh_device);
    auto buf_B_mesh = tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{.size = size_B}, lcfg, &mesh_device);
    auto buf_C_mesh = tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{.size = size_C}, lcfg, &mesh_device);

    auto *buf_A = buf_A_mesh->get_device_buffer(tt::tt_metal::distributed::MeshCoordinate{0, 0});
    auto *buf_B = buf_B_mesh->get_device_buffer(tt::tt_metal::distributed::MeshCoordinate{0, 0});
    auto *buf_C = buf_C_mesh->get_device_buffer(tt::tt_metal::distributed::MeshCoordinate{0, 0});

    if (opt.init_inputs)
    {
        // BF16 only, 2 values per uint32
        constexpr uint32_t WORDS_PER_TILE = tt::constants::TILE_HW / 2;
        std::vector<uint32_t> host_A(Mt * Kt * WORDS_PER_TILE);
        std::vector<uint32_t> host_B(Kt * Nt * WORDS_PER_TILE);
        for (uint32_t m = 0; m < Mt; m++)
        {
            for (uint32_t k = 0; k < Kt; k++)
            {
                uint32_t bits = bf16_bits(a_val(m, k));
                std::fill_n(host_A.begin() + (m * Kt + k) * WORDS_PER_TILE, WORDS_PER_TILE, bits | (bits << 16));
            }
        }
        for (uint32_t k = 0; k < Kt; k++)
        {
            for (uint32_t n = 0; n < Nt; n++)
            {
                uint32_t bits = bf16_bits(b_val(k, n));
                std::fill_n(host_B.begin() + (k * Nt + n) * WORDS_PER_TILE, WORDS_PER_TILE, bits | (bits << 16));
            }
        }
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, buf_A_mesh, host_A, true);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, buf_B_mesh, host_B, true);
    }

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::tt_metal::CircularBufferConfig cb_A_cfg =
        tt::tt_metal::CircularBufferConfig(Mt * Kt * TILE_SIZE_BYTES, {{tt::CBIndex::c_0, cfg.df}})
            .set_page_size(tt::CBIndex::c_0, TILE_SIZE_BYTES);
    tt::tt_metal::CircularBufferConfig cb_B_cfg =
        tt::tt_metal::CircularBufferConfig(Kt * Nt * TILE_SIZE_BYTES, {{tt::CBIndex::c_1, cfg.df}})
            .set_page_size(tt::CBIndex::c_1, TILE_SIZE_BYTES);
    tt::tt_metal::CircularBufferConfig cb_C_cfg =
        tt::tt_metal::CircularBufferConfig(Mt * Nt * TILE_SIZE_BYTES, {{tt::CBIndex::c_16, cfg.df}})
            .set_page_size(tt::CBIndex::c_16, TILE_SIZE_BYTES);

    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_A_cfg);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_B_cfg);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_C_cfg);

    ////////
    // Kernels and runtime args, reader and writer are reused from peak_flops
    ////////

    auto reader_kernel = tt::tt_metal::CreateKernel(
        program,
        "benchmarks/peak_flops/kernels/dataflow/reader_load_once.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default});

    auto writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "benchmarks/peak_flops/kernels/dataflow/writer_drain.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default});

    tt::tt_metal::KernelHandle compute_kernel;
    if (kv.is_tiles())
    {
        compute_kernel = tt::tt_metal::CreateKernel(
            program,
            "benchmarks/peak_flops/kernels/compute/matmul_loop.cpp",
            all_cores,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = cfg.fidelity,
                .compile_args = {Mt, Kt, Nt, MAX_DST_TILES, 1, 1}});
    }
    else
    {
        compute_kernel = tt::tt_metal::CreateKernel(
            program,
            "benchmarks/peak_flops_block/kernels/compute/matmul_block_loop.cpp",
            all_cores,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = cfg.fidelity,
                .compile_args = {Mt, Kt, Nt, kv.sub_h, kv.sub_w}});
    }

    uint32_t write_output = (opt.output != nullptr) ? 1 : 0;

    for (uint32_t y = 0; y < grid.y; y++)
    {
        for (uint32_t x = 0; x < grid.x; x++)
        {
            tt::tt_metal::CoreCoord core = {x, y};
            uint32_t core_idx = y * grid.x + x;
            uint32_t c_tile_off = core_idx * Mt * Nt;

            tt::tt_metal::SetRuntimeArgs(program, reader_kernel, core,
                                         {buf_A->address(), buf_B->address(), Mt, Kt, Nt});
            tt::tt_metal::SetRuntimeArgs(program, writer_kernel, core,
                                         {buf_C->address(), Mt * Nt, opt.inner_loop, c_tile_off,
                                          DRAIN_INTERVAL, write_output});
            tt::tt_metal::SetRuntimeArgs(program, compute_kernel, core, {opt.inner_loop});
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
    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < opt.num_iterations; i++)
    {
        tt::tt_metal::distributed::EnqueueMeshWorkload(cq, workload, false);
    }
    tt::tt_metal::distributed::Finish(cq);
    auto t1 = std::chrono::high_resolution_clock::now();

    // flush per-kernel device profiler results
    tt::tt_metal::detail::ReadDeviceProfilerResults(device);

    if (opt.output != nullptr)
    {
        opt.output->resize(size_C / sizeof(uint32_t));
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, *opt.output, buf_C_mesh, true);
    }

    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    double elapsed_per_iter = elapsed_s / opt.num_iterations;

    double M_elem = static_cast<double>(Mt) * tt::constants::TILE_HEIGHT;
    double K_elem = static_cast<double>(Kt) * tt::constants::TILE_HEIGHT;
    double N_elem = static_cast<double>(Nt) * tt::constants::TILE_HEIGHT;
    double flops = 2.0 * M_elem * K_elem * N_elem * opt.inner_loop * num_cores;
    double tflops = (flops / elapsed_per_iter) / 1e12;

    double tile_matmuls_per_core = static_cast<double>(Mt) * Kt * Nt * opt.inner_loop;
    double ns_per_tile_mm = elapsed_per_iter / tile_matmuls_per_core * 1e9;

    return BenchResult{tflops, elapsed_per_iter, ns_per_tile_mm};
}

// The kernels pack output tiles block by block, this maps CB position t back to (m, n)
void cb_index_to_mn(const KernelVariant &kv, const FormatConfig &cfg, uint32_t t, uint32_t &m, uint32_t &n)
{
    uint32_t sh, sw;
    if (kv.is_tiles())
    {
        uint32_t h = MAX_DST_TILES / cfg.Nt;
        sh = h < cfg.Mt ? h : cfg.Mt;
        sw = cfg.Nt;
    }
    else
    {
        sh = kv.sub_h;
        sw = kv.sub_w;
    }
    uint32_t block = t / (sh * sw);
    uint32_t in_block = t % (sh * sw);
    uint32_t blocks_per_row = cfg.Nt / sw;
    m = (block / blocks_per_row) * sh + in_block / sw;
    n = (block % blocks_per_row) * sw + in_block % sw;
}

// Compares every output element of every core with the expected value
void check_output(
    const KernelVariant &kv, const FormatConfig &cfg, uint32_t num_cores,
    const std::vector<uint32_t> &out, uint64_t &wrong, double &max_err)
{
    constexpr uint32_t WORDS_PER_TILE = tt::constants::TILE_HW / 2;
    const uint32_t tiles_per_core = cfg.Mt * cfg.Nt;
    wrong = 0;
    max_err = 0.0;
    for (uint32_t c = 0; c < num_cores; c++)
    {
        for (uint32_t t = 0; t < tiles_per_core; t++)
        {
            uint32_t m, n;
            cb_index_to_mn(kv, cfg, t, m, n);
            double expected = 0.0;
            for (uint32_t k = 0; k < cfg.Kt; k++)
            {
                expected += static_cast<double>(a_val(m, k)) * b_val(k, n);
            }
            expected *= tt::constants::TILE_WIDTH;

            const uint32_t base = (c * tiles_per_core + t) * WORDS_PER_TILE;
            for (uint32_t w = 0; w < WORDS_PER_TILE; w++)
            {
                uint32_t word = out[base + w];
                for (uint32_t half = 0; half < 2; half++)
                {
                    double got = bf16_to_float((word >> (16 * half)) & 0xFFFF);
                    double err = std::fabs(got - expected);
                    if (!(err <= max_err))
                        max_err = err; // also catches NaN
                    if (!(err <= 1e-3 * std::fabs(expected)))
                        wrong++;
                }
            }
        }
    }
}

std::string shape_str(const FormatConfig &cfg)
{
    return std::to_string(cfg.Mt * tt::constants::TILE_HEIGHT) + "x" + std::to_string(cfg.Kt * tt::constants::TILE_HEIGHT) + "x" + std::to_string(cfg.Nt * tt::constants::TILE_HEIGHT);
}

int main(int argc, char **argv)
{
    auto arg = [&](int i) -> std::string
    {
        if (argc <= i)
            return "";
        std::string s = argv[i];
        return s == "-" ? "" : s;
    };
    std::string matrix_path = arg(1);
    std::string tswp_path = arg(2);
    std::string correctness_path = arg(3);

    constexpr int device_id = 0;
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    tt::tt_metal::IDevice *device = mesh_device->get_device(0, 0);
    tt::tt_metal::distributed::MeshCommandQueue &cq = mesh_device->mesh_command_queue();

    tt::tt_metal::CoreCoord grid = device->compute_with_storage_grid_size();
    uint32_t num_cores = grid.x * grid.y;

    std::cout << "=== Peak FLOPS Benchmark — matmul_block vs matmul_tiles ===" << std::endl;
    std::cout << "Cores:      " << num_cores << " (" << grid.x << "x" << grid.y << ")" << std::endl;
    std::cout << "Inner loop: " << INNER_LOOP << " repeats per outer iteration" << std::endl;
    std::cout << "Outer iters:" << NUM_ITERATIONS << std::endl;

    // correctness check, BF16 LoFi, one inner iteration
    if (!correctness_path.empty())
    {
        std::cout << "\n=== Correctness (BF16 LoFi, Mt=Kt=16, Nt=8) ===" << std::endl;
        const FormatConfig cfg{tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::LoFi, 16, 16, 8};
        std::ofstream cf(correctness_path);
        cf << "kernel,sub_h,sub_w,elements_checked,wrong_elements,max_abs_error,pass\n";
        for (const auto &kv : KERNELS)
        {
            std::vector<uint32_t> out;
            RunOptions opt;
            opt.inner_loop = 1;
            opt.num_iterations = 1;
            opt.init_inputs = true;
            opt.output = &out;
            run_config(*mesh_device, cq, device, kv, cfg, opt);

            uint64_t wrong;
            double max_err;
            check_output(kv, cfg, num_cores, out, wrong, max_err);
            uint64_t checked = static_cast<uint64_t>(num_cores) * cfg.Mt * cfg.Nt * tt::constants::TILE_HW;
            bool pass = (wrong == 0);
            std::cout << std::left << std::setw(12) << kv.name
                      << (pass ? "PASS" : "FAIL")
                      << "  wrong " << wrong << "/" << checked
                      << "  max abs err " << max_err << std::endl;
            cf << kv.name << "," << kv.sub_h << "," << kv.sub_w << ","
               << checked << "," << wrong << "," << max_err << "," << (pass ? 1 : 0) << "\n";
        }
        std::cout << "Correctness CSV written to " << correctness_path << std::endl;
    }

    // every kernel x the format x fidelity matrix
    if (!matrix_path.empty())
    {
        std::cout << "\n=== Kernel x Format x Fidelity (Mt=Kt=16, Nt=8) ===" << std::endl;
        std::cout << std::left
                  << std::setw(12) << "Kernel"
                  << std::setw(8) << "Format"
                  << std::setw(8) << "Fid"
                  << std::setw(12) << "TFLOPS"
                  << std::setw(12) << "Efficiency"
                  << std::setw(14) << "ns/tile-mm"
                  << "Time/iter" << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        std::ofstream mf(matrix_path);
        mf << "kernel,sub_h,sub_w,data_format,math_fidelity,Mt,Kt,Nt,tflops,theoretical_peak_tflops,"
              "efficiency_pct,time_per_iter_ms,ns_per_tile_matmul\n";
        for (const auto &cfg : FIDELITY_MATRIX)
        {
            for (const auto &kv : KERNELS)
            {
                if (!fits(kv, cfg))
                    continue;
                BenchResult r = run_config(*mesh_device, cq, device, kv, cfg, RunOptions{});
                double peak = num_cores * per_core_peak_tflops(cfg.fidelity);
                double eff = r.tflops / peak * 100.0;
                std::cout << std::left
                          << std::setw(12) << kv.name
                          << std::setw(8) << format_name(cfg.df)
                          << std::setw(8) << fidelity_name(cfg.fidelity)
                          << std::setw(12) << std::fixed << std::setprecision(2) << r.tflops
                          << std::setw(12) << eff
                          << std::setw(14) << r.ns_per_tile_matmul
                          << r.elapsed_per_iter_s * 1e3 << " ms" << std::endl;
                mf << kv.name << "," << kv.sub_h << "," << kv.sub_w << ","
                   << format_name(cfg.df) << "," << fidelity_name(cfg.fidelity) << ","
                   << cfg.Mt << "," << cfg.Kt << "," << cfg.Nt << ","
                   << r.tflops << "," << peak << "," << eff << ","
                   << r.elapsed_per_iter_s * 1e3 << "," << r.ns_per_tile_matmul << "\n";
            }
        }
        std::cout << "\nMatrix CSV written to " << matrix_path << std::endl;
    }

    // BF16 LoFi tile sweep
    if (!tswp_path.empty())
    {
        std::cout << "\n=== Tile Size Sweep (BF16 LoFi, Mt=Kt=T, Nt=8) ===" << std::endl;
        std::cout << std::left
                  << std::setw(12) << "Kernel"
                  << std::setw(16) << "MxKxN"
                  << std::setw(12) << "TFLOPS"
                  << std::setw(14) << "ns/tile-mm"
                  << "Time/iter" << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        std::ofstream tf(tswp_path);
        tf << "kernel,sub_h,sub_w,tile_dim,tflops,theoretical_peak_tflops,efficiency_pct,"
              "time_per_iter_ms,ns_per_tile_matmul\n";
        for (uint32_t T : TILE_SWEEP_T)
        {
            const FormatConfig cfg{tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::LoFi, T, T, 8};
            for (const auto &kv : TILE_SWEEP_KERNELS)
            {
                if (!fits(kv, cfg))
                    continue;
                BenchResult r = run_config(*mesh_device, cq, device, kv, cfg, RunOptions{});
                double peak = num_cores * per_core_peak_tflops(cfg.fidelity);
                std::cout << std::left
                          << std::setw(12) << kv.name
                          << std::setw(16) << shape_str(cfg)
                          << std::setw(12) << std::fixed << std::setprecision(2) << r.tflops
                          << std::setw(14) << r.ns_per_tile_matmul
                          << r.elapsed_per_iter_s * 1e3 << " ms" << std::endl;
                tf << kv.name << "," << kv.sub_h << "," << kv.sub_w << ","
                   << (T * tt::constants::TILE_HEIGHT) << ","
                   << r.tflops << "," << peak << "," << r.tflops / peak * 100.0 << ","
                   << r.elapsed_per_iter_s * 1e3 << "," << r.ns_per_tile_matmul << "\n";
            }
        }
        std::cout << "\nTile sweep CSV written to " << tswp_path << std::endl;
    }

    mesh_device->close();
    return 0;
}
