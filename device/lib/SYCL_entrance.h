#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_pipes.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
// Include the FFT SYCL header for the library

// Entrance Kernel that reads from buffers and writes to pipes
class EntranceKernel {
private:
    size_t n;
    mutable sycl::buffer<std::complex<double>, 1> encoding_acc;

public:
    EntranceKernel(size_t n_val,
                   sycl::buffer<std::complex<double>, 1>& encoding_buf)
        : n(n_val), 
          encoding_acc(encoding_buf) {}
    
    void operator()(sycl::handler& h) const {
        // Get access to the buffers
        auto encoding = encoding_acc.get_access<sycl::access::mode::read>(h);

        // Capture kernel variables
        size_t kernel_n = n;

        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Format the encoding data into structs for the IFFT RTL
            // Process data in chunks of 4 complex numbers (no input bit-reversal)
            size_t num_chunks = kernel_n / 4;
            
            for (size_t chunk = 0; chunk < num_chunks; chunk++) {
                FFT_Input_Data input_data;
                size_t base_idx = chunk * 4;
                
                // Pack 4 complex values in sequential order
                input_data.port_data_in_0re = encoding[base_idx].real();
                input_data.port_data_in_0im = encoding[base_idx].imag();
                input_data.port_data_in_1re = encoding[base_idx + 1].real();
                input_data.port_data_in_1im = encoding[base_idx + 1].imag();
                input_data.port_data_in_2re = encoding[base_idx + 2].real();
                input_data.port_data_in_2im = encoding[base_idx + 2].imag();
                input_data.port_data_in_3re = encoding[base_idx + 3].real();
                input_data.port_data_in_3im = encoding[base_idx + 3].imag();
                
                // Send the formatted data to the pipe
                EntranceToFFTPipe::write(input_data);
            }
        });
    }
};