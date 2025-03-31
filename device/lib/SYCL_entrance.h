#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_pipes.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Entrance Kernel that reads from buffers and writes to pipes
class EntranceKernel {
private:
    size_t n;
    mutable sycl::buffer<complex_double, 1> encoding_acc;
    mutable sycl::buffer<int8_t, 1> error_samples_acc;

public:
    EntranceKernel(size_t n_val,
                   sycl::buffer<complex_double, 1>& encoding_buf,
                   sycl::buffer<int8_t, 1>& error_samples_buf)
        : n(n_val), 
          encoding_acc(encoding_buf),
          error_samples_acc(error_samples_buf) {}
    
    void operator()(sycl::handler& h) const {
        // Get access to the buffers
        auto encoding = encoding_acc.get_access<sycl::access::mode::read>(h);
        auto error_samples = error_samples_acc.get_access<sycl::access::mode::read>(h);
        
        // Capture necessary variables
        size_t kernel_n = n;
        
        // Create stream for debug output
        sycl::stream out(1024, 256, h);
        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            out << "Entrance: Starting to write " << kernel_n << " values to pipes\n" << sycl::flush;
            
            // Process elements one at a time
            for (size_t i = 0; i < kernel_n; i++) {
                // Write encoding data to IFFT pipe
                EntranceToIFFTPipe::write(encoding[i]);
                
                // Write error sample to error pipe
                EntranceToScaleErrorPipe::write(error_samples[i]);
                
                // Print progress at regular intervals
                if (i == 0 || i == kernel_n-1 || i % 1000 == 0) {
                    out << "Entrance: Written " << (i+1) << "/" << kernel_n << " values\n" << sycl::flush;
                }
            }
            
            out << "Entrance: All values successfully written to pipes\n" << sycl::flush;
        });
    }
};