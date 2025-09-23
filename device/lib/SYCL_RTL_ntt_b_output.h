#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_ntt_rtl_common.h"
#include "SYCL_pipes.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdio>
#include <cstdlib>

// RTL NTT B Output Kernel
class RTLNTTKernel_B_Output {
private:
    size_t n;                                     // Number of elements to process
    mutable sycl::buffer<uint32_t, 1> result_acc; // Result output buffer

public:
    RTLNTTKernel_B_Output(size_t n_val, sycl::buffer<uint32_t, 1>& result_buf)
        : n(n_val), result_acc(result_buf) {}

    void operator()(sycl::handler& h) const {
        // Get write access to the result buffer
        auto out_data_accessor = result_acc.get_access<sycl::access::mode::write>(h);

        // Capture necessary variables for the kernel
        size_t kernel_n = n;

        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Calculate number of structs to process (4K points = 1024 structs)
            size_t num_structs = kernel_n / 4;

            // Process each output structure from RTL NTT B
            for (size_t i = 0; i < num_structs; ++i) {
                // Read from NTT B output pipe (blocking read)
                NTT_RTL_Output_Data rtl_output = NTTBOutputPipe::read();

                // Convert RTL output back to individual uint32_t values
                uint32_t elem_0 = static_cast<uint32_t>(rtl_output.port_out_q_0);
                uint32_t elem_1 = static_cast<uint32_t>(rtl_output.port_out_q_1);
                uint32_t elem_2 = static_cast<uint32_t>(rtl_output.port_out_q_2);
                uint32_t elem_3 = static_cast<uint32_t>(rtl_output.port_out_q_3);

                // Write each element to the existing NTTToAddModPipe
                NTTToAddModPipe::write(elem_0);
                NTTToAddModPipe::write(elem_1);
                NTTToAddModPipe::write(elem_2);
                NTTToAddModPipe::write(elem_3);

                // Also write to the result buffer for output
                out_data_accessor[i * 4 + 0] = elem_0;
                out_data_accessor[i * 4 + 1] = elem_1;
                out_data_accessor[i * 4 + 2] = elem_2;
                out_data_accessor[i * 4 + 3] = elem_3;
            }
        }); // End single_task lambda
    } // End operator()
}; // End RTLNTTKernel_B_Output class