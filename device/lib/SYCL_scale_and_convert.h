#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_pipes.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Kernel for scaling and conversion to integers, then adding the error samples
class ScaleAndConvertKernel {
private:
    size_t n;
    double scale;
    mutable sycl::buffer<int64_t, 1> output_acc;

public:
    ScaleAndConvertKernel(size_t n_val, double scale_val, 
                            sycl::buffer<int64_t, 1>& output_buf)
        : n(n_val), scale(scale_val), output_acc(output_buf) {}
    
    void operator()(sycl::handler& h) const {
        // Get access to the output buffer
        auto output = output_acc.get_access<sycl::access::mode::write>(h);
        
        // Capture necessary variables
        size_t kernel_n = n;
        double kernel_scale = scale;
        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Calculate scaling factor
            double n_inv = kernel_scale / static_cast<double>(kernel_n);
            
            // Process each element
            for (size_t i = 0; i < kernel_n; i++) {
                // Read from pipes
                std::complex<double> encoded_value = IFFTToScaleAndConvertPipe::read();
                int8_t error_value = IFFTErrorToScaleAndConvertPipe::read();
                
                // Get real part of complex value
                double real_val = encoded_value.real();
                
                // Scale and round
                double scaled = sycl::round(real_val * n_inv);
                
                // Convert to integer and add error
                int64_t int_val = static_cast<int64_t>(scaled);
                int64_t result = int_val + error_value;
                
                // Write result directly to output buffer
                output[i] = result;
            }
        });
    }
};