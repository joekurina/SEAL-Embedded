#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_ntt_rtl_common.h"
#include "the_nwc_4k_ntt_sycl.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdio>
#include <cstdlib>

// RTL NTT A Main Kernel
class RTLNTTKernel_A {
private:
    uint32_t mod_value;  // Modulus value for RTL selector

public:
    RTLNTTKernel_A(uint32_t mod_val) : mod_value(mod_val) {}

    void operator()(sycl::handler& h) const {
        // Capture necessary variables for the kernel lambda
        uint32_t kernel_mod_val = mod_value;

        h.single_task<RTLNTTKernel_A>([=]() [[intel::kernel_args_restrict]] {
            //sycl::ext::oneapi::experimental::printf("RTLNTTKernel_A: Starting infinite loop\n");
#ifdef FPGA_EMULATOR
            // Create RTL instance for emulator mode
            reg_test_verifyNTT_multi_DUT* instance = the_nwc_4k_ntt_new_instance();
#endif

            // Get RTL modulus selector for this modulus value
            uint8_t rtl_modulus_selector = get_rtl_modulus_selector(kernel_mod_val);
            //sycl::ext::oneapi::experimental::printf("RTLNTTKernel_A: Using modulus selector %u for mod value %u\n",
            //                                        rtl_modulus_selector, kernel_mod_val);

            // Calculate number of structs to process (4K points = 1024 structs)
            size_t num_structs = NTT_RTL_CAPACITY;
            size_t processed_count = 0;

            // Process data structures through the RTL
            [[intel::initiation_interval(1)]]
            while (1) {
                // Read from input pipe (non-blocking)
                bool input_valid = false;
                NTT_RTL_Input_Data pipe_input = NTTAInputPipe::read(input_valid);

                if (input_valid) {
                    if (processed_count == 0) {
                        //sycl::ext::oneapi::experimental::printf("RTLNTTKernel_A: Received first valid input\n");
                    }
                    processed_count++;
                }

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

                    if (processed_count == num_structs) {
                        //sycl::ext::oneapi::experimental::printf("RTLNTTKernel_A: Processed all %zu structs\n", num_structs);
                        processed_count++; // Increment to avoid printing again
                    }
                }
            }

#ifdef FPGA_EMULATOR
            // Clean up RTL instance for emulator mode
            the_nwc_4k_ntt_delete_instance(instance);
#endif
        }); // End single_task lambda
    } // End operator()
}; // End RTLNTTKernel_A class