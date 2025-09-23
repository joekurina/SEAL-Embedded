#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_ntt_rtl_common.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdio>
#include <cstdlib>

// RTL NTT A Input Kernel
class RTLNTTKernel_A_Input {
private:
    size_t n;                                      // Number of elements to process
    mutable sycl::buffer<uint32_t, 1> vec_acc;    // Input buffer (secret key data)

public:
    // Constructor accepting input and save buffers
    RTLNTTKernel_A_Input(size_t n_val,
                         sycl::buffer<uint32_t, 1>& vec_buf)  // Input (secret key data)
        : n(n_val), vec_acc(vec_buf) {}

    void operator()(sycl::handler& h) const {
        // Accessor for input buffer (read only)
        auto data = vec_acc.get_access<sycl::access::mode::read>(h);

        // Capture necessary variables
        size_t kernel_n = n;

        h.single_task([=]() [[intel::kernel_args_restrict]] {
            sycl::ext::oneapi::experimental::printf("RTLNTTKernel_A_Input: Starting, n=%zu\n", kernel_n);

            // Calculate number of structs needed (4 elements per struct)
            size_t num_structs = kernel_n / 4;

            // Process data in chunks of 4 elements
            for (size_t i = 0; i < num_structs; ++i) {
                if (i == 0) {
                    sycl::ext::oneapi::experimental::printf("RTLNTTKernel_A_Input: Processing first struct\n");
                }
                // Read 4 consecutive elements from input buffer
                uint32_t elem_0 = data[i * 4 + 0];
                uint32_t elem_1 = data[i * 4 + 1];
                uint32_t elem_2 = data[i * 4 + 2];
                uint32_t elem_3 = data[i * 4 + 3];

                // Create RTL input data structure
                NTT_RTL_Input_Data rtl_input;
                rtl_input.port_x_in_0 = static_cast<int32_t>(elem_0);
                rtl_input.port_x_in_1 = static_cast<int32_t>(elem_1);
                rtl_input.port_x_in_2 = static_cast<int32_t>(elem_2);
                rtl_input.port_x_in_3 = static_cast<int32_t>(elem_3);

                // Write to NTT A input pipe
                NTTAInputPipe::write(rtl_input);


                if (i == num_structs - 1) {
                    sycl::ext::oneapi::experimental::printf("RTLNTTKernel_A_Input: Processed last struct %zu\n", i);
                }
            }
            sycl::ext::oneapi::experimental::printf("RTLNTTKernel_A_Input: Completed, wrote %zu structs\n", num_structs);
        }); // End single_task lambda
    } // End operator()
}; // End RTLNTTKernel_A_Input class