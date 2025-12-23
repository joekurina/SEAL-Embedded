#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_ntt_rtl_common.h"
#include "SYCL_pipes.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdio>
#include <cstdlib>

template <int P>
class RTLNTTKernel_B_Output_Task;

// RTL NTT B Output Kernel (templated on pipeline index)
template <int P>
class RTLNTTKernel_B_OutputT {
private:
    size_t n;                                     // Number of elements to process
    mutable sycl::buffer<u32x4_input, 1> result_acc; // Result output buffer (packed)

public:
    RTLNTTKernel_B_OutputT(size_t n_val, sycl::buffer<u32x4_input, 1>& result_buf)
        : n(n_val), result_acc(result_buf) {}

    void operator()(sycl::handler& h) const {
        // Get write access to the result buffer (packed)
        auto out_data_accessor = result_acc.get_access<sycl::access::mode::write>(h);

        // Capture necessary variables for the kernel
        size_t kernel_n = n;

        h.single_task<RTLNTTKernel_B_Output_Task<P>>([=]() [[intel::kernel_args_restrict]] {
            size_t num_structs = kernel_n / 4;

            // Process each output structure from RTL NTT B
            for (size_t i = 0; i < num_structs; ++i) {
                // Read from NTT B output pipe (blocking read)
                using NttPipes = NTT_PIPE_SET<P>;
                using PipeSet = CKKS_PIPE_SET<P>;

                NTT_RTL_Output_Data rtl_output = NttPipes::NTTBOutputPipe::read();

                // Pack into a 4-lane struct
                u32x4_input packed{};
                packed.element0 = static_cast<uint32_t>(rtl_output.port_out_q_0);
                packed.element1 = static_cast<uint32_t>(rtl_output.port_out_q_1);
                packed.element2 = static_cast<uint32_t>(rtl_output.port_out_q_2);
                packed.element3 = static_cast<uint32_t>(rtl_output.port_out_q_3);

                // Write packed struct to downstream pipe
                PipeSet::NTTToAddModPipe::write(packed);

                // Also write packed result to output buffer
                out_data_accessor[i] = packed;
            }
        }); // End single_task lambda
    } // End operator()
}; // End RTLNTTKernel_B_OutputT class

// Backwards-compatible alias for pipeline P = 0.
using RTLNTTKernel_B_Output = RTLNTTKernel_B_OutputT<0>;