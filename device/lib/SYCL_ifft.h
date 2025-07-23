#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_pipes.h"
#include "rtl/the_fft_sycl.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

class IFFTKernel {
public:
    void operator()(sycl::handler& h) const {
        h.single_task([=]() [[intel::kernel_args_restrict]] {
#ifdef FPGA_EMULATOR
            fft_example_DUT* instance = the_fft_new_instance();
#endif
            
            [[intel::initiation_interval(1)]]
            while (1) {
                // Read input data from the entrance pipe
                bool valid = false;
                FFT_Input_Data input_data = EntranceToFFTPipe::read(valid);
                
                // Prepare input for the_fft function
                the_fft_input_t input;
                input.port_v_in_s = valid ? 1 : 0;      // Set valid flag based on pipe read
                input.port_channel_in_s = 1;            // Channel 1
                input.port_data_in_0re = input_data.port_data_in_0re;
                input.port_data_in_0im = input_data.port_data_in_0im;
                input.port_data_in_1re = input_data.port_data_in_1re;
                input.port_data_in_1im = input_data.port_data_in_1im;
                input.port_data_in_2re = input_data.port_data_in_2re;
                input.port_data_in_2im = input_data.port_data_in_2im;
                input.port_data_in_3re = input_data.port_data_in_3re;
                input.port_data_in_3im = input_data.port_data_in_3im;
                
                // Call the RTL FFT function
#ifdef FPGA_EMULATOR
                the_fft_output_t output = the_fft(instance, input);
#else
                the_fft_output_t output = the_fft(input);
#endif
                
                // Process output if valid
                if (output.port_v_out_s == 1) {
                    // Create FFT_Output_Data structure directly from RTL output
                    FFT_Output_Data output_data;
                    output_data.port_data_out_0re = output.port_data_out_0re;
                    output_data.port_data_out_0im = output.port_data_out_0im;
                    output_data.port_data_out_1re = output.port_data_out_1re;
                    output_data.port_data_out_1im = output.port_data_out_1im;
                    output_data.port_data_out_2re = output.port_data_out_2re;
                    output_data.port_data_out_2im = output.port_data_out_2im;
                    output_data.port_data_out_3re = output.port_data_out_3re;
                    output_data.port_data_out_3im = output.port_data_out_3im;
                    
                    // Write the structure to the pipe
                    IFFTToScaleAndReducePipe::write(output_data);
                }
            }
            
#ifdef FPGA_EMULATOR
            the_fft_delete_instance(instance);
#endif
        }); // End of single_task
    } // End of operator()
}; // End of IFFTKernel class