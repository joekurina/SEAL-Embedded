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

public:
    ScaleAndConvertKernel(size_t n_val, double scale_val)
        : n(n_val), scale(scale_val) {}
    
    void operator()(sycl::handler& h) const {
        // Capture necessary variables
        size_t kernel_n = n;
        double kernel_scale = scale;
        
        // Create a stream for debug output
        sycl::stream out(1024, 256, h);
        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Calculate scaling factor once
            double n_inv = kernel_scale / static_cast<double>(kernel_n);
            
            out << "ScaleAndConvert: Starting to process " << kernel_n << " values\n" << sycl::flush;
            
            // Process elements in a streaming fashion
            for (size_t i = 0; i < kernel_n; i++) {
                // Read from both input pipes for one element
                complex_double encoded_value = IFFTToScaleAndConvertPipe::read();
                int8_t error_value = EntranceToScaleErrorPipe::read();
                
                // Process the element
                double real_val = encoded_value.real();
                double scaled = sycl::round(real_val * n_inv);
                int64_t int_val = static_cast<int64_t>(scaled);
                int64_t result = int_val + error_value;
                
                // Write result to the output pipe
                ScaleAndConvertToExitPipe::write(result);
                
                // Print progress at regular intervals
                if (i == 0 || i == kernel_n-1 || i % 1000 == 0) {
                    out << "ScaleAndConvert: Processed " << (i+1) << "/" << kernel_n << " values\n" << sycl::flush;
                }
            }
            
            out << "ScaleAndConvert: All values successfully processed and written\n" << sycl::flush;
        });
    }
};