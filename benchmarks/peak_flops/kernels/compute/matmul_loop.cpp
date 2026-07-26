// Performs tiled matmul on data already in L1 SRAM, looping INNER_LOOP times.
// Never touches DRAM

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"

// Tile dimensions passed as compile-time args from host
constexpr uint32_t Mt = get_compile_time_arg_val(0);
constexpr uint32_t Kt = get_compile_time_arg_val(1);
constexpr uint32_t Nt = get_compile_time_arg_val(2);

// Max output tiles the DST register file can hold at a point
constexpr uint32_t MAX_DST_TILES = get_compile_time_arg_val(3);

// 1 = issue real matmul_tiles()/pack_tile() calls (normal mode); 0 = skip them
// used to isolate the matmul_tiles and pack_tiles calls for profiling 
constexpr uint32_t DO_MATMUL = get_compile_time_arg_val(4);
constexpr uint32_t DO_PACK   = get_compile_time_arg_val(5);

// Sublocking, used to avoid acquiring more DST tiles than the register file can hold
static_assert(Nt <= MAX_DST_TILES,
    "subblock spans full output rows (Nt <= MAX_DST_TILES); "
    "add an N-subblock loop if Nt grows past MAX_DST_TILES");
constexpr uint32_t SUBBLOCK_H = (MAX_DST_TILES / Nt) < Mt ? (MAX_DST_TILES / Nt) : Mt;

void kernel_main() {

    uint32_t inner_loop = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_A = 0;
    constexpr uint32_t cb_B = 1;
    constexpr uint32_t cb_C = 16;

    mm_init(cb_A, cb_B, cb_C);
    cb_wait_front(cb_A, Mt * Kt);
    cb_wait_front(cb_B, Kt * Nt);

    // repeat matmul INNER_LOOP times on the same L1 data
    for (uint32_t iter = 0; iter < inner_loop; iter++) {

        cb_reserve_back(cb_C, Mt * Nt);

        for (uint32_t mb = 0; mb < Mt; mb += SUBBLOCK_H) {
            uint32_t h = (Mt - mb < SUBBLOCK_H) ? (Mt - mb) : SUBBLOCK_H;

            tile_regs_acquire();

            if constexpr (DO_MATMUL) {
                for (uint32_t m = 0; m < h; m++) {
                    for (uint32_t n = 0; n < Nt; n++) {
                        for (uint32_t k = 0; k < Kt; k++) {
                            // matmul_tiles: C[mb+m,n] += A[mb+m,k] * B[k,n]
                            matmul_tiles(
                                cb_A,                   // input CB for A
                                cb_B,                    // input CB for B
                                (mb + m) * Kt + k,       // tile index in A
                                k * Nt + n,               // tile index in B
                                m * Nt + n                // destination register index (subblock-local)
                            );
                        }
                    }
                }
            }

            tile_regs_commit();
            tile_regs_wait();

            if constexpr (DO_PACK) {
                for (uint32_t i = 0; i < h * Nt; i++) {
                    pack_tile(i, cb_C);
                }
            }
            tile_regs_release();
        }

        cb_push_back(cb_C, Mt * Nt);
        // don't pop A and B, they stay there for the next iteration
    }

    // release A and B at the very end
    cb_pop_front(cb_A, Mt * Kt);
    cb_pop_front(cb_B, Kt * Nt);
}