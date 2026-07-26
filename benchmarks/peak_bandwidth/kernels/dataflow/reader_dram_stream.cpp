// Streams tiles from DRAM into the input circular buffer, one full pass
// per dispatch.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CHUNK_TILES = get_compile_time_arg_val(0);

void kernel_main() {

    uint32_t src_addr    = get_arg_val<uint32_t>(0);
    uint32_t num_tiles   = get_arg_val<uint32_t>(1);
    uint32_t tile_offset = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    constexpr uint32_t tile_size = get_tile_size(cb_id);

    const InterleavedAddrGenFast<true> src = {
        .bank_base_address = src_addr,
        .page_size = tile_size,
        .data_format = get_dataformat(cb_id),
    };

    for (uint32_t i = 0; i < num_tiles; i += CHUNK_TILES) {
        uint32_t tiles_this_chunk = num_tiles - i;
        if (tiles_this_chunk > CHUNK_TILES) {
            tiles_this_chunk = CHUNK_TILES;
        }

        cb_reserve_back(cb_id, tiles_this_chunk);

        uint32_t l1_write_addr = get_write_ptr(cb_id);
        for (uint32_t t = 0; t < tiles_this_chunk; t++) {
            noc_async_read_page(tile_offset + i + t, src, l1_write_addr + t * tile_size);
        }
        noc_async_read_barrier();

        cb_push_back(cb_id, tiles_this_chunk);
    }
}
