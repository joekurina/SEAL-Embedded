#pragma once

#include "ntt_common.h"
#include "the_nwc_4k_ntt_sycl.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdio>
#include <cstdlib>

// RTL NTT A Main Kernel
class RTLNTTKernel_A {

public:
    RTLNTTKernel_A() {}

    void operator()(sycl::handler& h) const {

        h.single_task<RTLNTTKernel_A>([=]() [[intel::kernel_args_restrict]] {
#ifdef FPGA_EMULATOR
            // Create RTL instance for emulator mode
            reg_test_verifyNTT_multi_DUT* instance = the_nwc_4k_ntt_new_instance();
#endif

            // Get RTL modulus selector for this modulus value
            uint8_t rtl_modulus_selector = 3;

            // Process data structures through the RTL
            [[intel::initiation_interval(1)]]
            while (1) {
                // Read from input pipe (non-blocking)
                bool input_valid = false;
                NTT_RTL_Input_Data pipe_input = NTTAInputPipe::read(input_valid);

                // Prepare RTL input structure
                the_nwc_4k_ntt_input_t rtl_input;
                rtl_input.port_in_v_s = input_valid;  // Set valid flag based on pipe read
                rtl_input.port_in_c_s = rtl_modulus_selector;  // Modulus selector
                rtl_input.port_x_in_0 = pipe_input.port_x_in_0;
                rtl_input.port_x_in_1 = pipe_input.port_x_in_1;
                rtl_input.port_x_in_2 = pipe_input.port_x_in_2;
                rtl_input.port_x_in_3 = pipe_input.port_x_in_3;

                // Call RTL function every iteration
#ifdef FPGA_EMULATOR
                the_nwc_4k_ntt_output_t rtl_output = the_nwc_4k_ntt(instance, rtl_input);
#else
                the_nwc_4k_ntt_output_t rtl_output = the_nwc_4k_ntt(rtl_input);
#endif

                // Check if RTL output is valid and write to output pipe
                if (rtl_output.port_out_v_s == 1) {
                    NTT_RTL_Output_Data pipe_output;
                    pipe_output.port_out_q_0 = rtl_output.port_out_q_0;
                    pipe_output.port_out_q_1 = rtl_output.port_out_q_1;
                    pipe_output.port_out_q_2 = rtl_output.port_out_q_2;
                    pipe_output.port_out_q_3 = rtl_output.port_out_q_3;

                    // Write to output pipe
                    NTTAOutputPipe::write(pipe_output);
                }
            }

#ifdef FPGA_EMULATOR
            // Clean up RTL instance for emulator mode
            the_nwc_4k_ntt_delete_instance(instance);
#endif
        }); // End single_task lambda
    } // End operator()
}; // End RTLNTTKernel_A class