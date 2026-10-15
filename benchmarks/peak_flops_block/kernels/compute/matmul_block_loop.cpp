// Same as peak_flops/kernels/compute/matmul_loop.cpp, but uses matmul_block()
// instead of matmul_tiles(), so input tiles are reused across a SUB_H x SUB_W
// output block instead of being unpacked again for every output tile

#include "api/compute/common.h"
#include "api/compute/matmul.h"

// Matrix dims in tiles
constexpr uint32_t Mt = get_compile_time_arg_val(0);
constexpr uint32_t Kt = get_compile_time_arg_val(1);
constexpr uint32_t Nt = get_compile_time_arg_val(2);

// Output block size in tiles
constexpr uint32_t SUB_H = get_compile_time_arg_val(3);
constexpr uint32_t SUB_W = get_compile_time_arg_val(4);

static_assert(SUB_H * SUB_W <= 8, "block must fit in DST (8 tiles)");
static_assert(Mt % SUB_H == 0 && Nt % SUB_W == 0, "block must divide the matrix");

void kernel_main() {

    uint32_t inner_loop = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_A = 0;
    constexpr uint32_t cb_B = 1;
    constexpr uint32_t cb_C = 16;

    mm_block_init(cb_A, cb_B, cb_C, 0, SUB_W, SUB_H, Kt);
    cb_wait_front(cb_A, Mt * Kt);
    cb_wait_front(cb_B, Kt * Nt);

    // repeat matmul INNER_LOOP times on the same L1 data
    for (uint32_t iter = 0; iter < inner_loop; iter++) {

        cb_reserve_back(cb_C, Mt * Nt);

        for (uint32_t mb = 0; mb < Mt; mb += SUB_H) {
            for (uint32_t nb = 0; nb < Nt; nb += SUB_W) {

                tile_regs_acquire();

                // one call computes the whole block for one k step
                for (uint32_t k = 0; k < Kt; k++) {
                    matmul_block(cb_A, cb_B, mb * Kt + k, k * Nt + nb, 0, 0, SUB_W, SUB_H, Kt);
                }

                tile_regs_commit();
                tile_regs_wait();

                for (uint32_t i = 0; i < SUB_H * SUB_W; i++) {
                    pack_tile(i, cb_C);
                }
                tile_regs_release();
            }
        }

        cb_push_back(cb_C, Mt * Nt);
    }

    cb_pop_front(cb_A, Mt * Kt);
    cb_pop_front(cb_B, Kt * Nt);
}
