// Same as peak_bandwidth's writer_dram_stream.cpp, but drains the output CB (c_16)

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CHUNK_TILES = get_compile_time_arg_val(0);

void kernel_main() {

    uint32_t dst_addr    = get_arg_val<uint32_t>(0);
    uint32_t num_tiles   = get_arg_val<uint32_t>(1);
    uint32_t tile_offset = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_C = tt::CBIndex::c_16;
    constexpr uint32_t tile_size = get_tile_size(cb_C);

    const InterleavedAddrGenFast<true> dst = {
        .bank_base_address = dst_addr,
        .page_size = tile_size,
        .data_format = get_dataformat(cb_C),
    };

    for (uint32_t i = 0; i < num_tiles; i += CHUNK_TILES) {
        uint32_t n = num_tiles - i;
        if (n > CHUNK_TILES) {
            n = CHUNK_TILES;
        }

        cb_wait_front(cb_C, n);
        uint32_t l1_addr = get_read_ptr(cb_C);
        for (uint32_t t = 0; t < n; t++) {
            noc_async_write_page(tile_offset + i + t, dst, l1_addr + t * tile_size);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_C, n);
    }

    noc_async_write_barrier();
}
