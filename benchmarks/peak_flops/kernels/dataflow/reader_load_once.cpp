// Loads matrix A and B tiles from DRAM into L1 SRAM once, then exits.
// The compute kernel loops over the same L1 data for all inner_loop iterations.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {

    uint32_t addr_A = get_arg_val<uint32_t>(0);
    uint32_t addr_B = get_arg_val<uint32_t>(1);
    uint32_t Mt     = get_arg_val<uint32_t>(2);
    uint32_t Kt     = get_arg_val<uint32_t>(3);
    uint32_t Nt     = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_A = tt::CBIndex::c_0;
    constexpr uint32_t cb_B = tt::CBIndex::c_1;

    const uint32_t tile_size = get_tile_size(cb_A);

    const InterleavedAddrGenFast<true> src_A = {
        .bank_base_address = addr_A,
        .page_size = tile_size,
        .data_format = get_dataformat(cb_A),
    };

    const InterleavedAddrGenFast<true> src_B = {
        .bank_base_address = addr_B,
        .page_size = tile_size,
        .data_format = get_dataformat(cb_B),
    };

    uint32_t total_A = Mt * Kt;
    uint32_t total_B = Kt * Nt;

    // Load all A tiles into L1 in one batch, then barrier once
    cb_reserve_back(cb_A, total_A);
    uint32_t l1_write_addr = get_write_ptr(cb_A);

    for (uint32_t i = 0; i < total_A; i++) {
        noc_async_read_page(i, src_A, l1_write_addr + i * tile_size);
    }

    noc_async_read_barrier();
    cb_push_back(cb_A, total_A);

    // Load all B tiles into L1 in one batch, then barrier once
    cb_reserve_back(cb_B, total_B);
    l1_write_addr = get_write_ptr(cb_B);

    for (uint32_t i = 0; i < total_B; i++) {
        noc_async_read_page(i, src_B, l1_write_addr + i * tile_size);
    }

    noc_async_read_barrier();
    cb_push_back(cb_B, total_B);
    
}