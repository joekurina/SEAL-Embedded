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
    sycl::buffer<u32x4_input, 1>& save_acc;

public:
    RTLNTTKernel_A_Output(size_t n_val, sycl::buffer<u32x4_input, 1>& save_buf) : n(n_val), save_acc(save_buf) {}

    void operator()(sycl::handler& h) const {
        // Accessor for save buffer (write only)
        auto s_save = save_acc.get_access<sycl::access::mode::write>(h);

        // Capture necessary variables for the kernel lambda
        size_t kernel_n = n;

        h.single_task<class RTLNTTKernel_A_Output>([=]() [[intel::kernel_args_restrict]] {
            size_t num_structs = kernel_n / 4;

            // Process each output structure from RTL NTT A
            for (size_t i = 0; i < num_structs; ++i) {
                // Read from NTT A output pipe (blocking read)
                NTT_RTL_Output_Data rtl_output = NTTAOutputPipe::read();

                u32x4_input packed{};
                packed.element0 = static_cast<uint32_t>(rtl_output.port_out_q_0);
                packed.element1 = static_cast<uint32_t>(rtl_output.port_out_q_1);
                packed.element2 = static_cast<uint32_t>(rtl_output.port_out_q_2);
                packed.element3 = static_cast<uint32_t>(rtl_output.port_out_q_3);

                // Write packed elements to the existing NTTToPolyMultNegPipe
                NTTToPolyMultNegPipe::write(packed);

                // Save the NTT(s) state to the provided buffer
                s_save[i] = packed;
            }
        }); // End single_task lambda
    } // End operator()
}; // End RTLNTTKernel_A_Output class