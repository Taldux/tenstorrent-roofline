// Measures on-chip L1/NoC bandwidth
//
// Allocates two L1 buffers on each core.  A single
// dataflow kernel loops INNER_LOOP times, reading TRANSFER_SIZE bytes from
// the core's L1 src into its L1 dst via the loopback kernel.
// This benchmarks the L1 -> NoC -> L1 bandwidth, as a means of showing that
// it is not a limiting factor of device performance.
//
// Measured bandwidth = INNER_LOOP × TRANSFER_SIZE × 2 × num_cores / time

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>

constexpr uint32_t TRANSFER_SIZE = 32 * 2048;   // 64KB transferred per core per innner-loop iteration
constexpr uint32_t INNER_LOOP = 10000;
constexpr uint32_t NUM_ITERATIONS = 10;

int main() {

    ////////
    // Initial setup
    ////////

    constexpr int device_id = 0;
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    tt::tt_metal::IDevice* device = mesh_device->get_device(0, 0);
    tt::tt_metal::distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    tt::tt_metal::CoreCoord grid = device->compute_with_storage_grid_size();
    uint32_t num_cores = grid.x * grid.y;
    tt::tt_metal::CoreRange all_cores(tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{grid.x - 1, grid.y - 1});

    std::cout << "=== L1/NoC Bandwidth Benchmark ===" << std::endl;
    std::cout << "Cores:          " << num_cores << " (" << grid.x << "x" << grid.y << ")" << std::endl;
    std::cout << "Transfer size:  " << TRANSFER_SIZE / 1024 << " KB per core" << std::endl;
    std::cout << "Inner loop:     " << INNER_LOOP << " reads per dispatch" << std::endl;
    std::cout << "Iterations:     " << NUM_ITERATIONS << std::endl;
    std::cout << std::endl;

    ////////
    // L1 buffer allocation
    ////////

    uint32_t l1_total = TRANSFER_SIZE * num_cores;

    tt::tt_metal::distributed::DeviceLocalBufferConfig lcfg_src{
        .page_size = TRANSFER_SIZE, .buffer_type = tt::tt_metal::BufferType::L1};
    tt::tt_metal::distributed::DeviceLocalBufferConfig lcfg_dst{
        .page_size = TRANSFER_SIZE, .buffer_type = tt::tt_metal::BufferType::L1};

    auto src_mesh = tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{.size = l1_total}, lcfg_src, mesh_device.get());
    auto dst_mesh = tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{.size = l1_total}, lcfg_dst, mesh_device.get());

    auto* src_dev = src_mesh->get_device_buffer(tt::tt_metal::distributed::MeshCoordinate{0, 0});
    auto* dst_dev = dst_mesh->get_device_buffer(tt::tt_metal::distributed::MeshCoordinate{0, 0});

    ////////
    // Program and kernel setup
    ////////

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    auto kernel = tt::tt_metal::CreateKernel(
        program,
        "benchmarks/l1_bandwidth/kernels/dataflow/l1_loopback.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc       = tt::tt_metal::NOC::RISCV_1_default
        });

    for (uint32_t y = 0; y < grid.y; y++) {
        for (uint32_t x = 0; x < grid.x; x++) {
            tt::tt_metal::SetRuntimeArgs(program, kernel, tt::tt_metal::CoreCoord{x, y}, {
                src_dev->address(),
                dst_dev->address(),
                TRANSFER_SIZE,
                INNER_LOOP
            });
        }
    }

    ////////
    // L1 buffer allocation
    ////////

    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange device_range(mesh_device->shape());
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


    ////////
    // L1 buffer allocation
    ////////

    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    double per_iter_s = elapsed_s / NUM_ITERATIONS;

    double bytes_per_iter = static_cast<double>(INNER_LOOP) * TRANSFER_SIZE * 2.0 * num_cores;
    double bw_GBs         = (bytes_per_iter / per_iter_s) / 1e9;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "=== Results ===" << std::endl;
    std::cout << "Time per iteration:  " << per_iter_s * 1e3 << " ms" << std::endl;
    std::cout << "Measured bandwidth:  " << bw_GBs << " GB/s" << std::endl;

    mesh_device->close();
    return 0;
}
