#pragma once

#include "ntt_common.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdio>
#include <cstdlib>

// RTL NTT A Output Kernel
class RTLNTTKernel_A_Output {
private:
    sycl::buffer<uint32_t, 1>& save_acc;

public:
    RTLNTTKernel_A_Output(sycl::buffer<uint32_t, 1>& save_buf) : save_acc(save_buf) {}

    void operator()(sycl::handler& h) const {
        // Accessor for save buffer (write only)
        auto s_save = save_acc.get_access<sycl::access::mode::write>(h);

        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Process each output structure from RTL NTT A
            for (size_t i = 0; i < 1024; ++i) {
                // Read from NTT A output pipe (blocking read)
                NTT_RTL_Output_Data rtl_output = NTTAOutputPipe::read();

                // Convert RTL output back to individual uint32_t values
                // and write them to the existing pipeline
                uint32_t elem_0 = static_cast<uint32_t>(rtl_output.port_out_q_0);
                uint32_t elem_1 = static_cast<uint32_t>(rtl_output.port_out_q_1);
                uint32_t elem_2 = static_cast<uint32_t>(rtl_output.port_out_q_2);
                uint32_t elem_3 = static_cast<uint32_t>(rtl_output.port_out_q_3);

                // Save the elements to the S buffer
                s_save[i * 4 + 0] = elem_0;
                s_save[i * 4 + 1] = elem_1;
                s_save[i * 4 + 2] = elem_2;
                s_save[i * 4 + 3] = elem_3;
            }
        }); // End single_task lambda
    } // End operator()
}; // End RTLNTTKernel_A_Output class