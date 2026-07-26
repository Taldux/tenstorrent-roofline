// Measures peak compute throughput across numerical formats on a single
// Wormhole chip: BF4, BF8, and BF16, which is the three formats Tenstorrent's own
// GEMM_FLOPS tech report actually publishes numbers for
//
// Loads small A and B matrices into each core's L1 SRAM once, then
// loop the matmul INNER_LOOP times with zero DRAM traffic. Each format is
// benchmarked at its Tenstorrent standard fidelity pairing (BF4/LoFi,
// BF8/HiFi2, BF16/HiFi4). Then results are printed and
// saved to CSV.

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
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

constexpr uint32_t INNER_LOOP = 10000;
constexpr uint32_t DRAIN_INTERVAL = 0;
constexpr uint32_t WRITE_OUTPUT_TO_DRAM = 0;
constexpr uint32_t NUM_ITERATIONS = 10;

// Per core matrix engine peak, by math fidelity, at the nominal 1 GHz AI clock
double per_core_peak_tflops(tt::tt_metal::MathFidelity fidelity) {
    switch (fidelity) {
        case tt::tt_metal::MathFidelity::LoFi:  return 4.0;
        case tt::tt_metal::MathFidelity::HiFi2: return 2.0;
        case tt::tt_metal::MathFidelity::HiFi3: return 4.0 / 3.0;
        case tt::tt_metal::MathFidelity::HiFi4: return 1.0;
        default:                                return 0.0;
    }
}

// DST register capacity used by the subblocked matmul kernel
constexpr uint32_t MAX_DST_TILES = 8;

struct FormatConfig {
    std::string name; // display name
    tt::DataFormat df; // tile data format
    tt::tt_metal::MathFidelity fidelity;
    uint32_t Mt, Kt, Nt; // matrix dims in tiles
    uint32_t do_matmul = 1; // 1 = issue real matmul_tiles() calls
    uint32_t do_pack   = 1; // 1 = issue real pack_tile() calls
};

double theoretical_peak_tflops(const FormatConfig& cfg, uint32_t num_cores) {
    return num_cores * per_core_peak_tflops(cfg.fidelity);
}

// Mt=Kt=T swept, Nt fixed at 8, extended up to T=16
// clang-format off
const std::vector<FormatConfig> TILE_SWEEP = {
    {"BF16_LoFi_1T",  tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::LoFi, 1,  1,  8},
    {"BF16_LoFi_2T",  tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::LoFi, 2,  2,  8},
    {"BF16_LoFi_4T",  tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::LoFi, 4,  4,  8},
    {"BF16_LoFi_6T",  tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::LoFi, 6,  6,  8},
    {"BF16_LoFi_8T",  tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::LoFi, 8,  8,  8},
    {"BF16_LoFi_10T", tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::LoFi, 10, 10, 8},
    {"BF16_LoFi_12T", tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::LoFi, 12, 12, 8},
    {"BF16_LoFi_14T", tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::LoFi, 14, 14, 8},
    {"BF16_LoFi_16T", tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::LoFi, 16, 16, 8},
};
// clang-format on

// Same Mt=Kt=T/Nt=8 sweep, HiFi4 instead of LoFi. LoFi's plateau is a
// per-instruction FPU issue-rate limit (i think, after doing phase sweep
// LoFI not even budging), HiFi4 does 4x more real compute
// per matmul_tiles() call, so it shouldn't hit that same ceiling and
// should show a classic climbing curve instead.
// clang-format off
const std::vector<FormatConfig> TILE_SWEEP_HIFI4 = {
    {"BF16_HiFi4_1T",  tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::HiFi4, 1,  1,  8},
    {"BF16_HiFi4_2T",  tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::HiFi4, 2,  2,  8},
    {"BF16_HiFi4_4T",  tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::HiFi4, 4,  4,  8},
    {"BF16_HiFi4_6T",  tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::HiFi4, 6,  6,  8},
    {"BF16_HiFi4_8T",  tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::HiFi4, 8,  8,  8},
    {"BF16_HiFi4_10T", tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::HiFi4, 10, 10, 8},
    {"BF16_HiFi4_12T", tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::HiFi4, 12, 12, 8},
    {"BF16_HiFi4_14T", tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::HiFi4, 14, 14, 8},
    {"BF16_HiFi4_16T", tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::HiFi4, 16, 16, 8},
};
// clang-format on

// Full 3x3 formatxfidelity matrix, all at the same Mt=Kt=16/Nt=8 shape
// as CONFIGS. CONFIGS only tests each format at Tenstorrent's recommended
// fidelity (BF4/LoFi, BF8/HiFi2, BF16/HiFi4),whereas this tests every
// combination to check whether the "LoFi can't reach its ceiling" effect
// found for BF16 (see TILE_SWEEP/TILE_SWEEP_HIFI4) is fidelity-specific
// or just particular to BF16.
// clang-format off
const std::vector<FormatConfig> FIDELITY_MATRIX = {
    {"BF4_LoFi",   tt::DataFormat::Bfp4_b,    tt::tt_metal::MathFidelity::LoFi,  16, 16, 8},
    {"BF4_HiFi2",  tt::DataFormat::Bfp4_b,    tt::tt_metal::MathFidelity::HiFi2, 16, 16, 8},
    {"BF4_HiFi4",  tt::DataFormat::Bfp4_b,    tt::tt_metal::MathFidelity::HiFi4, 16, 16, 8},
    {"BF8_LoFi",   tt::DataFormat::Bfp8_b,    tt::tt_metal::MathFidelity::LoFi,  16, 16, 8},
    {"BF8_HiFi2",  tt::DataFormat::Bfp8_b,    tt::tt_metal::MathFidelity::HiFi2, 16, 16, 8},
    {"BF8_HiFi4",  tt::DataFormat::Bfp8_b,    tt::tt_metal::MathFidelity::HiFi4, 16, 16, 8},
    {"BF16_LoFi",  tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::LoFi,  16, 16, 8},
    {"BF16_HiFi2", tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::HiFi2, 16, 16, 8},
    {"BF16_HiFi4", tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::HiFi4, 16, 16, 8},
};
// clang-format on

// Isolates which phase of the hot loop (matmul_tiles() FPU compute vs
// pack_tile() packing) dominates the LoFi 8T plateau, mirroring the
// DO_READ/DO_WRITE isolation used for peak_bandwidth.cpp's DRAM
// investigation
// clang-format off
const std::vector<FormatConfig> PHASE_SWEEP = {
    {"LoFi_8T_both",        tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::LoFi, 8, 8, 8, 1, 1},
    {"LoFi_8T_matmul_only", tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::LoFi, 8, 8, 8, 1, 0},
    {"LoFi_8T_pack_only",   tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::LoFi, 8, 8, 8, 0, 1},
};
// clang-format on

// clang-format off
const std::vector<FormatConfig> CONFIGS = {
    // name   data format              fidelity                            Mt  Kt  Nt
    {"BF4",  tt::DataFormat::Bfp4_b,    tt::tt_metal::MathFidelity::LoFi,   16, 16, 8},
    {"BF8",  tt::DataFormat::Bfp8_b,    tt::tt_metal::MathFidelity::HiFi2,  16, 16, 8},
    {"BF16", tt::DataFormat::Float16_b, tt::tt_metal::MathFidelity::HiFi4,  16, 16, 8},
};
// clang-format on

struct BenchResult {
    double tflops;
    double elapsed_per_iter_s;
    double flops_per_iter;
    double arithmetic_intensity;  // FLOP/byte based on one-time DRAM load + INNER_LOOP compute
};

BenchResult run_config(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& cq,
    tt::tt_metal::IDevice* device,
    const FormatConfig& cfg)
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

    tt::tt_metal::distributed::DeviceLocalBufferConfig lcfg_A{
        .page_size = TILE_SIZE_BYTES, .buffer_type = tt::tt_metal::BufferType::DRAM};
    tt::tt_metal::distributed::DeviceLocalBufferConfig lcfg_B{
        .page_size = TILE_SIZE_BYTES, .buffer_type = tt::tt_metal::BufferType::DRAM};
    tt::tt_metal::distributed::DeviceLocalBufferConfig lcfg_C{
        .page_size = TILE_SIZE_BYTES, .buffer_type = tt::tt_metal::BufferType::DRAM};

    auto buf_A_mesh = tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{.size = size_A}, lcfg_A, &mesh_device);
    auto buf_B_mesh = tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{.size = size_B}, lcfg_B, &mesh_device);
    auto buf_C_mesh = tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{.size = size_C}, lcfg_C, &mesh_device);

    auto* buf_A = buf_A_mesh->get_device_buffer(tt::tt_metal::distributed::MeshCoordinate{0, 0});
    auto* buf_B = buf_B_mesh->get_device_buffer(tt::tt_metal::distributed::MeshCoordinate{0, 0});
    auto* buf_C = buf_C_mesh->get_device_buffer(tt::tt_metal::distributed::MeshCoordinate{0, 0});


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
    // Kernels and runtime args
    ////////

    auto reader_kernel = tt::tt_metal::CreateKernel(
        program,
        "benchmarks/peak_flops/kernels/dataflow/reader_load_once.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc       = tt::tt_metal::NOC::RISCV_1_default
        });

    auto writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "benchmarks/peak_flops/kernels/dataflow/writer_drain.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc       = tt::tt_metal::NOC::RISCV_0_default
        });

    auto compute_kernel = tt::tt_metal::CreateKernel(
        program,
        "benchmarks/peak_flops/kernels/compute/matmul_loop.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = cfg.fidelity,
            .compile_args  = {Mt, Kt, Nt, MAX_DST_TILES, cfg.do_matmul, cfg.do_pack}
        });

    for (uint32_t y = 0; y < grid.y; y++) {
        for (uint32_t x = 0; x < grid.x; x++) {
            tt::tt_metal::CoreCoord core = {x, y};
            uint32_t core_idx    = y * grid.x + x;
            uint32_t c_tile_off  = core_idx * Mt * Nt;

            tt::tt_metal::SetRuntimeArgs(program, reader_kernel, core,
                {buf_A->address(), buf_B->address(), Mt, Kt, Nt});
            tt::tt_metal::SetRuntimeArgs(program, writer_kernel, core,
                {buf_C->address(), Mt * Nt, INNER_LOOP, c_tile_off,
                 DRAIN_INTERVAL, WRITE_OUTPUT_TO_DRAM});
            tt::tt_metal::SetRuntimeArgs(program, compute_kernel, core, {INNER_LOOP});
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
    for (uint32_t i = 0; i < NUM_ITERATIONS; i++) {
        tt::tt_metal::distributed::EnqueueMeshWorkload(cq, workload, false);
    }
    tt::tt_metal::distributed::Finish(cq);
    auto t1 = std::chrono::high_resolution_clock::now();

    // flush per-kernel device profiler results
    tt::tt_metal::detail::ReadDeviceProfilerResults(device);

    double elapsed_s        = std::chrono::duration<double>(t1 - t0).count();
    double elapsed_per_iter = elapsed_s / NUM_ITERATIONS;

    double M_elem = static_cast<double>(Mt) * tt::constants::TILE_HEIGHT;
    double K_elem = static_cast<double>(Kt) * tt::constants::TILE_HEIGHT;
    double N_elem = static_cast<double>(Nt) * tt::constants::TILE_HEIGHT;
    double flops  = 2.0 * M_elem * K_elem * N_elem * INNER_LOOP * num_cores;
    double tflops = (flops / elapsed_per_iter) / 1e12;

    // AI = total FLOPS / DRAM bytes
    double dram_bytes = static_cast<double>(Mt * Kt + Kt * Nt) * TILE_SIZE_BYTES;
    double ai         = (2.0 * M_elem * K_elem * N_elem * INNER_LOOP) / dram_bytes;

    return BenchResult{tflops, elapsed_per_iter, flops, ai};
}


int main(int argc, char** argv) {
    // accept optional output CSV path as argv[1]
    std::string csv_path = (argc > 1) ? argv[1] : "";

    constexpr int device_id = 0;
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    tt::tt_metal::IDevice* device = mesh_device->get_device(0, 0);
    tt::tt_metal::distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    tt::tt_metal::CoreCoord grid = device->compute_with_storage_grid_size();
    uint32_t num_cores = grid.x * grid.y;

    std::cout << "=== Peak FLOPS Benchmark — Format Sweep (BF4/BF8/BF16) ===" << std::endl;
    std::cout << "Cores:      " << num_cores << " (" << grid.x << "x" << grid.y << ")" << std::endl;
    std::cout << "Inner loop: " << INNER_LOOP << " repeats per outer iteration" << std::endl;
    std::cout << "Outer iters:" << NUM_ITERATIONS << std::endl;
    std::cout << std::endl;

    // column headers
    std::cout << std::left
              << std::setw(14) << "Config"
              << std::setw(16) << "MtxKtxNt"
              << std::setw(14) << "TFLOPS"
              << std::setw(14) << "Efficiency"
              << "Time/iter" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    std::vector<std::pair<FormatConfig, BenchResult>> results;

    for (const auto& cfg : CONFIGS) {
        std::string mat_str =
            std::to_string(cfg.Mt * tt::constants::TILE_HEIGHT) + "x"
            + std::to_string(cfg.Kt * tt::constants::TILE_HEIGHT) + "x"
            + std::to_string(cfg.Nt * tt::constants::TILE_HEIGHT);
        BenchResult r = run_config(*mesh_device, cq, device, cfg);
        double peak = theoretical_peak_tflops(cfg, num_cores);
        std::string eff_str = (peak > 0)
            ? (std::to_string(static_cast<int>((r.tflops / peak) * 100.0)) + "%")
            : "N/A";

        std::cout << std::left
                  << std::setw(14) << cfg.name
                  << std::setw(16) << mat_str
                  << std::setw(14) << (std::to_string(static_cast<int>(r.tflops)) + " TFLOPS")
                  << std::setw(14) << eff_str
                  << r.elapsed_per_iter_s * 1e3 << " ms" << std::endl;

        results.push_back({cfg, r});
    }

    {
        const auto& [cfg, r] = results[2]; // BF16
        double peak = theoretical_peak_tflops(cfg, num_cores);
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "FLOPs per iteration: " << r.flops_per_iter / 1e9 << " GFLOPs" << std::endl;
        std::cout << "Time per iteration:  " << r.elapsed_per_iter_s * 1e3 << " ms" << std::endl;
        std::cout << "Measured throughput: " << r.tflops << " TFLOPS (" << cfg.name << ")" << std::endl;
        std::cout << "Peak (" << num_cores << "-core): " << peak << " TFLOPS" << std::endl;
        std::cout << "Efficiency:          " << (r.tflops / peak) * 100.0 << "%" << std::endl;
    }

    // write to csv
    if (!csv_path.empty()) {
        std::ofstream f(csv_path);
        f << "config,data_format,math_fidelity,Mt,Kt,Nt,tflops,theoretical_peak_tflops,efficiency_pct,time_per_iter_ms,arithmetic_intensity\n";
        for (const auto& [cfg, r] : results) {
            double peak = theoretical_peak_tflops(cfg, num_cores);
            // fidelity string
            std::string fid_str;
            switch (cfg.fidelity) {
                case tt::tt_metal::MathFidelity::LoFi:  fid_str = "LoFi";  break;
                case tt::tt_metal::MathFidelity::HiFi2: fid_str = "HiFi2"; break;
                case tt::tt_metal::MathFidelity::HiFi4: fid_str = "HiFi4"; break;
                default:                  fid_str = "Other";
            }
            // format string
            std::string df_str;
            switch (cfg.df) {
                case tt::DataFormat::Bfp4_b:    df_str = "BF4";  break;
                case tt::DataFormat::Bfp8_b:    df_str = "BF8";  break;
                case tt::DataFormat::Float16_b: df_str = "BF16"; break;
                default:                        df_str = "Other";
            }
            f << cfg.name << ","
              << df_str << ","
              << fid_str << ","
              << cfg.Mt << "," << cfg.Kt << "," << cfg.Nt << ","
              << r.tflops << ",";
            if (peak > 0) {
                f << peak << "," << (r.tflops / peak) * 100.0;
            } else {
                f << "NA,NA";
            }
            f << "," << r.elapsed_per_iter_s * 1e3 << ","
              << r.arithmetic_intensity << "\n";
        }
        std::cout << "\nCSV written to " << csv_path << std::endl;
    }

    // tile size sweep, optional CSV output path as argv[2]
    std::string tswp_path = (argc > 2) ? argv[2] : "";
    if (!tswp_path.empty()) {
        std::cout << "\n=== Tile Size Sweep (BF16 LoFi, Mt=Kt=1..16, Nt=8 fixed) ===" << std::endl;
        std::cout << std::left
                  << std::setw(16) << "Config"
                  << std::setw(14) << "MtxKtxNt"
                  << std::setw(14) << "TFLOPS"
                  << "Time/iter" << std::endl;
        std::cout << std::string(66, '-') << std::endl;

        std::vector<std::pair<FormatConfig, BenchResult>> tswp;
        for (const auto& cfg : TILE_SWEEP) {
            BenchResult r = run_config(*mesh_device, cq, device, cfg);
            std::string tdim =
                std::to_string(cfg.Mt * tt::constants::TILE_HEIGHT) + "x"
                + std::to_string(cfg.Kt * tt::constants::TILE_HEIGHT) + "x"
                + std::to_string(cfg.Nt * tt::constants::TILE_HEIGHT);
            std::cout << std::left
                      << std::setw(16) << cfg.name
                      << std::setw(14) << tdim
                      << std::setw(14) << (std::to_string(static_cast<int>(r.tflops)) + " TFLOPS")
                      << r.elapsed_per_iter_s * 1e3 << " ms" << std::endl;
            tswp.push_back({cfg, r});
        }

        std::ofstream tf(tswp_path);
        tf << "config,tile_dim,tflops,theoretical_peak_tflops,efficiency_pct,time_per_iter_ms,arithmetic_intensity\n";
        for (const auto& [cfg, r] : tswp) {
            double peak = theoretical_peak_tflops(cfg, num_cores);  // all BF16 LoFi, always > 0
            double eff = (r.tflops / peak) * 100.0;
            tf << cfg.name << ","
               << (cfg.Mt * tt::constants::TILE_HEIGHT) << ","
               << r.tflops << ","
               << peak << ","
               << eff << ","
               << r.elapsed_per_iter_s * 1e3 << ","
               << r.arithmetic_intensity << "\n";
        }
        std::cout << "\nTile sweep CSV written to " << tswp_path << std::endl;
    }

    // matmul-vs-pack phase isolation, optional CSV output path as argv[3]
    std::string phase_path = (argc > 3) ? argv[3] : "";
    if (!phase_path.empty()) {
        std::cout << "\n=== Phase Isolation (BF16 LoFi, Mt=Nt=Kt=8) ===" << std::endl;
        std::cout << std::left
                  << std::setw(22) << "Config"
                  << std::setw(14) << "TFLOPS"
                  << "Time/iter" << std::endl;
        std::cout << std::string(60, '-') << std::endl;

        std::vector<std::pair<FormatConfig, BenchResult>> phw;
        for (const auto& cfg : PHASE_SWEEP) {
            BenchResult r = run_config(*mesh_device, cq, device, cfg);
            std::cout << std::left
                      << std::setw(22) << cfg.name
                      << std::setw(14) << (std::to_string(static_cast<int>(r.tflops)) + " TFLOPS")
                      << r.elapsed_per_iter_s * 1e3 << " ms" << std::endl;
            phw.push_back({cfg, r});
        }

        std::ofstream pf(phase_path);
        pf << "config,do_matmul,do_pack,tflops,time_per_iter_ms\n";
        for (const auto& [cfg, r] : phw) {
            pf << cfg.name << ","
               << cfg.do_matmul << ","
               << cfg.do_pack << ","
               << r.tflops << ","
               << r.elapsed_per_iter_s * 1e3 << "\n";
        }
        std::cout << "\nPhase sweep CSV written to " << phase_path << std::endl;
    }

    // HiFi4 tile size sweep (same Mt=Kt=T/Nt=8 shape as TILE_SWEEP, LoFi
    // swapped for HiFi4), optional CSV output path as argv[4]
    std::string tswp_hifi4_path = (argc > 4) ? argv[4] : "";
    if (!tswp_hifi4_path.empty()) {
        std::cout << "\n=== Tile Size Sweep (BF16 HiFi4, Mt=Kt=1..16, Nt=8 fixed) ===" << std::endl;
        std::cout << std::left
                  << std::setw(16) << "Config"
                  << std::setw(14) << "MtxKtxNt"
                  << std::setw(14) << "TFLOPS"
                  << "Time/iter" << std::endl;
        std::cout << std::string(66, '-') << std::endl;

        std::vector<std::pair<FormatConfig, BenchResult>> tswp4;
        for (const auto& cfg : TILE_SWEEP_HIFI4) {
            BenchResult r = run_config(*mesh_device, cq, device, cfg);
            std::string tdim =
                std::to_string(cfg.Mt * tt::constants::TILE_HEIGHT) + "x"
                + std::to_string(cfg.Kt * tt::constants::TILE_HEIGHT) + "x"
                + std::to_string(cfg.Nt * tt::constants::TILE_HEIGHT);
            std::cout << std::left
                      << std::setw(16) << cfg.name
                      << std::setw(14) << tdim
                      << std::setw(14) << (std::to_string(static_cast<int>(r.tflops)) + " TFLOPS")
                      << r.elapsed_per_iter_s * 1e3 << " ms" << std::endl;
            tswp4.push_back({cfg, r});
        }

        std::ofstream tf4(tswp_hifi4_path);
        tf4 << "config,tile_dim,tflops,theoretical_peak_tflops,efficiency_pct,time_per_iter_ms,arithmetic_intensity\n";
        for (const auto& [cfg, r] : tswp4) {
            double peak = theoretical_peak_tflops(cfg, num_cores);
            double eff = (r.tflops / peak) * 100.0;
            tf4 << cfg.name << ","
                << (cfg.Mt * tt::constants::TILE_HEIGHT) << ","
                << r.tflops << ","
                << peak << ","
                << eff << ","
                << r.elapsed_per_iter_s * 1e3 << ","
                << r.arithmetic_intensity << "\n";
        }
        std::cout << "\nHiFi4 tile sweep CSV written to " << tswp_hifi4_path << std::endl;
    }

    // Full format x fidelity matrix, optional CSV output path as argv[5]
    std::string matrix_path = (argc > 5) ? argv[5] : "";
    if (!matrix_path.empty()) {
        std::cout << "\n=== Format x Fidelity Matrix (Mt=Kt=16, Nt=8) ===" << std::endl;
        std::cout << std::left
                  << std::setw(14) << "Config"
                  << std::setw(14) << "TFLOPS"
                  << std::setw(14) << "Efficiency"
                  << "Time/iter" << std::endl;
        std::cout << std::string(66, '-') << std::endl;

        std::vector<std::pair<FormatConfig, BenchResult>> matrix;
        for (const auto& cfg : FIDELITY_MATRIX) {
            BenchResult r = run_config(*mesh_device, cq, device, cfg);
            double peak = theoretical_peak_tflops(cfg, num_cores);
            double eff = (r.tflops / peak) * 100.0;
            std::cout << std::left
                      << std::setw(14) << cfg.name
                      << std::setw(14) << (std::to_string(static_cast<int>(r.tflops)) + " TFLOPS")
                      << std::setw(14) << (std::to_string(static_cast<int>(eff)) + "%")
                      << r.elapsed_per_iter_s * 1e3 << " ms" << std::endl;
            matrix.push_back({cfg, r});
        }

        std::ofstream mf(matrix_path);
        mf << "config,data_format,math_fidelity,tflops,theoretical_peak_tflops,efficiency_pct,time_per_iter_ms\n";
        for (const auto& [cfg, r] : matrix) {
            double peak = theoretical_peak_tflops(cfg, num_cores);
            double eff = (r.tflops / peak) * 100.0;
            std::string fmt_name = (cfg.df == tt::DataFormat::Bfp4_b) ? "BF4"
                                  : (cfg.df == tt::DataFormat::Bfp8_b) ? "BF8" : "BF16";
            std::string fid_name = (cfg.fidelity == tt::tt_metal::MathFidelity::LoFi)  ? "LoFi"
                                  : (cfg.fidelity == tt::tt_metal::MathFidelity::HiFi2) ? "HiFi2"
                                  : (cfg.fidelity == tt::tt_metal::MathFidelity::HiFi3) ? "HiFi3" : "HiFi4";
            mf << cfg.name << "," << fmt_name << "," << fid_name << ","
               << r.tflops << "," << peak << "," << eff << ","
               << r.elapsed_per_iter_s * 1e3 << "\n";
        }
        std::cout << "\nFormat x fidelity matrix CSV written to " << matrix_path << std::endl;
    }

    mesh_device->close();
    return 0;
}
