// Streams tiles of A and B from DRAM into two input CBs, CHUNK_TILES at a time

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CHUNK_TILES = get_compile_time_arg_val(0);

void kernel_main() {

    uint32_t addr_A      = get_arg_val<uint32_t>(0);
    uint32_t addr_B      = get_arg_val<uint32_t>(1);
    uint32_t num_tiles   = get_arg_val<uint32_t>(2);
    uint32_t tile_offset = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_A = tt::CBIndex::c_0;
    constexpr uint32_t cb_B = tt::CBIndex::c_1;
    constexpr uint32_t tile_size = get_tile_size(cb_A);

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

    for (uint32_t i = 0; i < num_tiles; i += CHUNK_TILES) {
        uint32_t n = num_tiles - i;
        if (n > CHUNK_TILES) {
            n = CHUNK_TILES;
        }

        cb_reserve_back(cb_A, n);
        cb_reserve_back(cb_B, n);
        uint32_t l1_A = get_write_ptr(cb_A);
        uint32_t l1_B = get_write_ptr(cb_B);
        for (uint32_t t = 0; t < n; t++) {
            noc_async_read_page(tile_offset + i + t, src_A, l1_A + t * tile_size);
            noc_async_read_page(tile_offset + i + t, src_B, l1_B + t * tile_size);
        }
        noc_async_read_barrier();
        cb_push_back(cb_A, n);
        cb_push_back(cb_B, n);
    }
}
