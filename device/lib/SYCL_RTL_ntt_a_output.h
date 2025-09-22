#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_ntt_rtl_common.h"
#include "SYCL_pipes.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdio>
#include <cstdlib>

// RTL NTT A Output Kernel
// This kernel processes RTL NTT A output and connects to the existing pipeline
// It reads from NTT A output pipe and writes to the existing NTTToPolyMultNegPipe
class RTLNTTKernel_A_Output {
private:
    size_t n;  // Number of elements to process

public:
    RTLNTTKernel_A_Output(size_t n_val) : n(n_val) {}

    void operator()(sycl::handler& h) const {
        // Capture necessary variables for the kernel lambda
        size_t kernel_n = n;

        h.single_task([=]() [[intel::kernel_args_restrict]] {
            sycl::ext::oneapi::experimental::printf("RTLNTTKernel_A_Output: Starting, n=%zu\n", kernel_n);

            // Calculate number of structs to process (4K points = 1024 structs)
            size_t num_structs = kernel_n / 4;

            // Process each output structure from RTL NTT A
            for (size_t i = 0; i < num_structs; ++i) {
                if (i == 0) {
                    sycl::ext::oneapi::experimental::printf("RTLNTTKernel_A_Output: Reading first struct from RTL\n");
                }
                // Read from NTT A output pipe (blocking read)
                NTT_RTL_Output_Data rtl_output = NTTAOutputPipe::read();

                // Convert RTL output back to individual uint32_t values
                // and write them to the existing pipeline
                uint32_t elem_0 = static_cast<uint32_t>(rtl_output.port_out_q_0);
                uint32_t elem_1 = static_cast<uint32_t>(rtl_output.port_out_q_1);
                uint32_t elem_2 = static_cast<uint32_t>(rtl_output.port_out_q_2);
                uint32_t elem_3 = static_cast<uint32_t>(rtl_output.port_out_q_3);

                // Write each element to the existing NTTToPolyMultNegPipe
                // This maintains compatibility with the rest of the pipeline
                NTTToPolyMultNegPipe::write(elem_0);
                NTTToPolyMultNegPipe::write(elem_1);
                NTTToPolyMultNegPipe::write(elem_2);
                NTTToPolyMultNegPipe::write(elem_3);

                if (i == num_structs - 1) {
                    sycl::ext::oneapi::experimental::printf("RTLNTTKernel_A_Output: Processed last struct %zu\n", i);
                }
            }
            sycl::ext::oneapi::experimental::printf("RTLNTTKernel_A_Output: Completed, processed %zu structs\n", num_structs);
        }); // End single_task lambda
    } // End operator()
}; // End RTLNTTKernel_A_Output class