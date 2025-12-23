#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_ntt_rtl_common.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdio>
#include <cstdlib>

template <int P>
class RTLNTTKernel_A_Input_Task;

// RTL NTT A Input Kernel (templated on pipeline index)
template <int P>
class RTLNTTKernel_A_InputT {
private:
    size_t n;                                       // Number of elements to process
    uint8_t mod_sel;                               // Modulus selector
    mutable sycl::buffer<u32x4_input, 1> vec_acc;      // Input buffer (secret key data, packed)

public:
    // Constructor accepting input and save buffers
        RTLNTTKernel_A_InputT(size_t n_val,
                                                    uint8_t mod_selector,
                                                    sycl::buffer<u32x4_input, 1>& vec_buf)  // Input (secret key data)
                        : n(n_val),
                          mod_sel(mod_selector), 
                          vec_acc(vec_buf) {}

    void operator()(sycl::handler& h) const {
        // Accessor for input buffer (read only)
        auto data = vec_acc.get_access<sycl::access::mode::read>(h);

        // Capture necessary variables
        size_t kernel_n = n;
        uint8_t kernel_mod_sel = mod_sel;

        h.single_task<RTLNTTKernel_A_Input_Task<P>>([=]() [[intel::kernel_args_restrict]] {
            size_t num_structs = kernel_n / 4;

            // Process data in packed 4-element blocks
            for (size_t i = 0; i < num_structs; ++i) {
                u32x4_input packed = data[i];

                NTT_RTL_Input_Data rtl_input;
                rtl_input.port_x_in_0 = static_cast<int32_t>(packed.element0);
                rtl_input.port_x_in_1 = static_cast<int32_t>(packed.element1);
                rtl_input.port_x_in_2 = static_cast<int32_t>(packed.element2);
                rtl_input.port_x_in_3 = static_cast<int32_t>(packed.element3);
                
                // Write modulus selector to pipe
                using PipeSet = NTT_PIPE_SET<P>;
                PipeSet::NTTAModSelectorPipe::write(kernel_mod_sel);
                // Write to NTT A input pipe
                PipeSet::NTTAInputPipe::write(rtl_input);
            }
        }); // End single_task lambda
    } // End operator()
}; // End RTLNTTKernel_A_InputT class

// Backwards-compatible alias for pipeline P = 0.
using RTLNTTKernel_A_Input = RTLNTTKernel_A_InputT<0>;