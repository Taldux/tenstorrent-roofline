// Drains the output CB each iteration to prevent stalls.
// DRAM writeback is optional and can be throttled or disabled.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr       = get_arg_val<uint32_t>(0);
    uint32_t tiles_per_iter = get_arg_val<uint32_t>(1); // Mt * Nt
    uint32_t inner_loop     = get_arg_val<uint32_t>(2);
    uint32_t tile_offset    = get_arg_val<uint32_t>(3); // per-core start tile in buf_C
    uint32_t drain_interval = get_arg_val<uint32_t>(4); // 0 -> only final iteration if writes enabled
    uint32_t write_output   = get_arg_val<uint32_t>(5); // 0 -> no DRAM writes, just drain CB

    constexpr uint32_t cb_C = tt::CBIndex::c_16;

    const uint32_t tile_size = get_tile_size(cb_C);

    const InterleavedAddrGenFast<true> dst = {
        .bank_base_address = dst_addr,
        .page_size = tile_size,
        .data_format = get_dataformat(cb_C),
    };

    for (uint32_t iter = 0; iter < inner_loop; iter++) {
        cb_wait_front(cb_C, tiles_per_iter);

        bool should_write = false;
        if (write_output != 0) {
            // Always allow a final write
            should_write = (iter + 1 == inner_loop);
            if (drain_interval > 0 && ((iter + 1) % drain_interval == 0)) {
                should_write = true;
            }
        }

        if (should_write) {
            uint32_t l1_addr = get_read_ptr(cb_C);
            for (uint32_t i = 0; i < tiles_per_iter; i++) {
                // Write each tile back to its fixed slot in DRAM
                // tile_offset + i is the global tile index for this core's output slice
                noc_async_write_page(tile_offset + i, dst, l1_addr + i * tile_size);
            }
            noc_async_write_barrier();
        }

        cb_pop_front(cb_C, tiles_per_iter);
    }
}