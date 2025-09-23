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
    sycl::buffer<uint32_t, 1>& save_acc;

public:
    RTLNTTKernel_A_Output(size_t n_val, sycl::buffer<uint32_t, 1>& save_buf) : n(n_val), save_acc(save_buf) {}

    void operator()(sycl::handler& h) const {
        // Accessor for save buffer (write only)
        auto s_save = save_acc.get_access<sycl::access::mode::write>(h);

        // Capture necessary variables for the kernel lambda
        size_t kernel_n = n;

        // Perform host-side check if save buffer is valid
        //bool save_output = (save_acc.get_range() == sycl::range(kernel_n));

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

                // Also save the elements to the buffer
                //if (save_output) {
                    s_save[i * 4 + 0] = elem_0;
                    s_save[i * 4 + 1] = elem_1;
                    s_save[i * 4 + 2] = elem_2;
                    s_save[i * 4 + 3] = elem_3;
                //}

                if (i == num_structs - 1) {
                    sycl::ext::oneapi::experimental::printf("RTLNTTKernel_A_Output: Processed last struct %zu\n", i);
                }
            }
            sycl::ext::oneapi::experimental::printf("RTLNTTKernel_A_Output: Completed, processed %zu structs\n", num_structs);
        }); // End single_task lambda
    } // End operator()
}; // End RTLNTTKernel_A_Output class