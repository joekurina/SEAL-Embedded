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
        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Interleave writing to both pipes
            for (size_t i = 0; i < kernel_n; i++) {
                EntranceToIFFTPipe::write(encoding[i]);
                EntranceToScaleErrorPipe::write(error_samples[i]);
            }
        });
    }
};