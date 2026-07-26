// Streams tiles from CB_out0 back to DRAM, one full pass per dispatch.
// See reader_dram_stream.cpp for why this isn't looping multiple passes
// internally anymore.

#include "api/dataflow/dataflow_api.h"

// Must match reader_dram_stream.cpp's CHUNK_TILES
constexpr uint32_t CHUNK_TILES = get_compile_time_arg_val(0);

void kernel_main() {

    uint32_t dst_addr    = get_arg_val<uint32_t>(0);
    uint32_t num_tiles   = get_arg_val<uint32_t>(1);
    uint32_t tile_offset = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    constexpr uint32_t tile_size = get_tile_size(cb_id);

    const InterleavedAddrGenFast<true> dst = {
        .bank_base_address = dst_addr,
        .page_size = tile_size,
        .data_format = get_dataformat(cb_id),
    };

    for (uint32_t i = 0; i < num_tiles; i += CHUNK_TILES) {
        uint32_t tiles_this_chunk = num_tiles - i;
        if (tiles_this_chunk > CHUNK_TILES) {
            tiles_this_chunk = CHUNK_TILES;
        }

        cb_wait_front(cb_id, tiles_this_chunk);

        uint32_t l1_read_addr = get_read_ptr(cb_id);
        for (uint32_t t = 0; t < tiles_this_chunk; t++) {
            noc_async_write_page(tile_offset + i + t, dst, l1_read_addr + t * tile_size);
        }
        noc_async_writes_flushed();

        cb_pop_front(cb_id, tiles_this_chunk);
    }

    // One real, full barrier for the entire per core tile range
    noc_async_write_barrier();
}
