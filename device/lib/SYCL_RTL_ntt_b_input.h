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
    size_t n;             // Number of elements to process

public:
    RTLNTTKernel_B_Input(size_t n_val) : n(n_val) {}

    void operator()(sycl::handler& h) const {
        // Capture necessary variables
        size_t kernel_n = n;

        h.single_task([=]() [[intel::kernel_args_restrict]] {
            //sycl::ext::oneapi::experimental::printf("RTLNTTKernel_B_Input: Starting, n=%zu\n", kernel_n);

            // Calculate number of structs needed (4 elements per struct)
            size_t num_structs = kernel_n / 4;

            // Buffer to accumulate 4 elements before creating a struct
            uint32_t element_buffer[4];
            size_t buffer_index = 0;

            // Read individual elements from the existing pipeline
            for (size_t i = 0; i < kernel_n; ++i) {
                // Read from the existing ScaleReduceToNTTBPipe (blocking read)
                uint32_t pipe_element = ScaleReduceToNTTBPipe::read();

                // Accumulate elements in buffer
                element_buffer[buffer_index] = pipe_element;
                buffer_index++;

                // When we have 4 elements, create an RTL input struct
                if (buffer_index == 4) {
                    // Create RTL input data structure
                    NTT_RTL_Input_Data rtl_input;
                    rtl_input.port_x_in_0 = static_cast<int32_t>(element_buffer[0]);
                    rtl_input.port_x_in_1 = static_cast<int32_t>(element_buffer[1]);
                    rtl_input.port_x_in_2 = static_cast<int32_t>(element_buffer[2]);
                    rtl_input.port_x_in_3 = static_cast<int32_t>(element_buffer[3]);

                    // Write to NTT B input pipe
                    NTTBInputPipe::write(rtl_input);

                    // Reset buffer
                    buffer_index = 0;
                }
            }
            //sycl::ext::oneapi::experimental::printf("RTLNTTKernel_B_Input: Completed, wrote %zu structs\n", num_structs);
        }); // End single_task lambda
    } // End operator()
}; // End RTLNTTKernel_B_Input class