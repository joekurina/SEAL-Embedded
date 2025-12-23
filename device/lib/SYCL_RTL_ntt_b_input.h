#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_ntt_rtl_common.h"
#include "SYCL_pipes.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdio>
#include <cstdlib>

template <int P>
class RTLNTTKernel_B_Input_Task;

// RTL NTT B Input Kernel (templated on pipeline index)
template <int P>
class RTLNTTKernel_B_InputT {
private:
    size_t n;                   // Number of elements to process
    uint8_t mod_sel;            // Modulus selector

public:
    RTLNTTKernel_B_InputT(size_t n_val, uint8_t mod_selector) : n(n_val), mod_sel(mod_selector)  {}

    void operator()(sycl::handler& h) const {
        // Capture necessary variables
        size_t kernel_n = n;
        uint8_t kernel_mod_sel = mod_sel;

        h.single_task<RTLNTTKernel_B_Input_Task<P>>([=]() [[intel::kernel_args_restrict]] {
            size_t num_structs = kernel_n / 4;

            for (size_t blk = 0; blk < num_structs; ++blk) {
                using PipeSet = CKKS_PIPE_SET<P>;
                u32x4_input packed = PipeSet::ScaleReduceToNTTBPipe::read();

                NTT_RTL_Input_Data rtl_input;
                rtl_input.port_x_in_0 = static_cast<int32_t>(packed.element0);
                rtl_input.port_x_in_1 = static_cast<int32_t>(packed.element1);
                rtl_input.port_x_in_2 = static_cast<int32_t>(packed.element2);
                rtl_input.port_x_in_3 = static_cast<int32_t>(packed.element3);

                using NttPipes = NTT_PIPE_SET<P>;
                NttPipes::NTTBModSelectorPipe::write(kernel_mod_sel);
                NttPipes::NTTBInputPipe::write(rtl_input);
            }
        }); // End single_task lambda
    } // End operator()
}; // End RTLNTTKernel_B_InputT class

// Backwards-compatible alias for pipeline P = 0.
using RTLNTTKernel_B_Input = RTLNTTKernel_B_InputT<0>;