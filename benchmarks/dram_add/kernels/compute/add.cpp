// C = A + B, CHUNK_TILES tiles per DST acquire

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"

constexpr uint32_t CHUNK_TILES = get_compile_time_arg_val(0);

void kernel_main() {

    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_A = tt::CBIndex::c_0;
    constexpr uint32_t cb_B = tt::CBIndex::c_1;
    constexpr uint32_t cb_C = tt::CBIndex::c_16;

    binary_op_init_common(cb_A, cb_B, cb_C);
    add_tiles_init(cb_A, cb_B);

    for (uint32_t i = 0; i < num_tiles; i += CHUNK_TILES) {
        uint32_t n = num_tiles - i;
        if (n > CHUNK_TILES) {
            n = CHUNK_TILES;
        }

        cb_wait_front(cb_A, n);
        cb_wait_front(cb_B, n);

        tile_regs_acquire();
        for (uint32_t t = 0; t < n; t++) {
            add_tiles(cb_A, cb_B, t, t, t);
        }
        tile_regs_commit();

        cb_reserve_back(cb_C, n);
        tile_regs_wait();
        for (uint32_t t = 0; t < n; t++) {
            pack_tile(t, cb_C);
        }
        tile_regs_release();
        cb_push_back(cb_C, n);

        cb_pop_front(cb_A, n);
        cb_pop_front(cb_B, n);
    }
}
