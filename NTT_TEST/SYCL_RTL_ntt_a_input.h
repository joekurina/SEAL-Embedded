#pragma once

#include "ntt_common.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdio>
#include <cstdlib>

// RTL NTT A Input Kernel
class RTLNTTKernel_A_Input {
private:
    mutable sycl::buffer<uint32_t, 1> vec_acc;

public:
    // Constructor accepting input buffer
    RTLNTTKernel_A_Input(sycl::buffer<uint32_t, 1>& vec_buf) : vec_acc(vec_buf) {}

    void operator()(sycl::handler& h) const {
        // Accessor for input buffer (read only)
        auto data = vec_acc.get_access<sycl::access::mode::read>(h);

        h.single_task([=]() [[intel::kernel_args_restrict]] {

            // Process data in chunks of 4 elements
            for (size_t i = 0; i < 1024; ++i) {
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
            }
        }); // End single_task lambda
    } // End operator()
}; // End RTLNTTKernel_A_Input class