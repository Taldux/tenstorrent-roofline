// L1/NoC loopback bandwidth kernel.
// Reads TRANSFER_SIZE bytes from the core's L1 src buffer into its dst buffer
// via NOC, INNER_LOOP times for each dispatch.
// This measures the aggregate NoC bandwidth

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_l1_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_l1_addr = get_arg_val<uint32_t>(1);
    const uint32_t transfer_size = get_arg_val<uint32_t>(2);  // size in bytes
    const uint32_t inner_loop = get_arg_val<uint32_t>(3);  // transfers per dispatch

    const uint64_t src_noc = get_noc_addr(src_l1_addr);

    // read from core's L1 src buffer into its dst buffer via NoC
    for (uint32_t i = 0; i < inner_loop; i++) {
        noc_async_read(src_noc, dst_l1_addr, transfer_size);
        noc_async_read_barrier();
    }
}
