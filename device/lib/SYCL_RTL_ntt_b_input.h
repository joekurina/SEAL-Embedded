#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_ntt_rtl_common.h"
#include "SYCL_pipes.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdio>
#include <cstdlib>

// RTL NTT B Input Kernel
class RTLNTTKernel_B_Input {
private:
    size_t n;                   // Number of elements to process
    uint8_t mod_sel;            // Modulus selector

public:
    RTLNTTKernel_B_Input(size_t n_val, uint8_t mod_selector) : n(n_val), mod_sel(mod_selector)  {}

    void operator()(sycl::handler& h) const {
        // Capture necessary variables
        size_t kernel_n = n;
        uint8_t kernel_mod_sel = mod_sel;

        h.single_task<class RTLNTTKenel_B_Input>([=]() [[intel::kernel_args_restrict]] {
            size_t num_structs = kernel_n / 4;

            for (size_t blk = 0; blk < num_structs; ++blk) {
                u32x4_input packed = ScaleReduceToNTTBPipe::read();

                NTT_RTL_Input_Data rtl_input;
                rtl_input.port_x_in_0 = static_cast<int32_t>(packed.element0);
                rtl_input.port_x_in_1 = static_cast<int32_t>(packed.element1);
                rtl_input.port_x_in_2 = static_cast<int32_t>(packed.element2);
                rtl_input.port_x_in_3 = static_cast<int32_t>(packed.element3);

                NTTBModSelectorPipe::write(kernel_mod_sel);
                NTTBInputPipe::write(rtl_input);
            }
        }); // End single_task lambda
    } // End operator()
}; // End RTLNTTKernel_B_Input class