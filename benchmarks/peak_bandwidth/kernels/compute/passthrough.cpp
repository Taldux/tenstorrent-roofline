// Passthrough compute kernel for the bandwidth benchmark.
//
// Since tt-metalium requires a compute kernel when both a
// reader and writer exist, we supply the minimal valid thing, so a kernel that
// does nothing. The reader writes to CB_in (c_0) and the writer reads from
// the same CB (c_0), bypassing CB_out entirely.
 
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    // empty
}
 