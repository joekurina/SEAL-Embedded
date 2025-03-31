#pragma once

#include "SYCL_ckks_sym.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Kernel for scaling and conversion to integers, then adding the error samples to the encoded values
class ScaleAndConvertKernel {
private:
    size_t n;
    double scale;
    mutable sycl::buffer<std::complex<double>, 1> encoding_acc;
    mutable sycl::buffer<int64_t, 1> pt_with_error_acc;
    mutable sycl::buffer<int8_t, 1> error_samples_acc;

public:
    ScaleAndConvertKernel(size_t n_val, double scale_val,
                          sycl::buffer<std::complex<double>, 1>& encoding_buf,
                          sycl::buffer<int64_t, 1>& pt_with_error_buf,
                          sycl::buffer<int8_t, 1>& error_samples_buf)
        : n(n_val), scale(scale_val), encoding_acc(encoding_buf),
          pt_with_error_acc(pt_with_error_buf), error_samples_acc(error_samples_buf) {}
    
    void operator()(sycl::handler& h) const {
        // Get access to buffers
        auto encoding = encoding_acc.get_access<sycl::access::mode::read>(h);
        auto pt_with_error = pt_with_error_acc.get_access<sycl::access::mode::write>(h);
        auto error_samples = error_samples_acc.get_access<sycl::access::mode::read>(h);
        
        // Capture necessary variables
        size_t kernel_n = n;
        double kernel_scale = scale;
        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Calculate scaling factor
            double n_inv = kernel_scale / static_cast<double>(kernel_n);
            
            // Scale, convert to integers, and add error in one pass
            for (size_t i = 0; i < kernel_n; i++) {
                // Get real part of complex value
                double real_val = encoding[i].real();
                
                // Scale and round
                double scaled = sycl::round(real_val * n_inv);
                
                // Convert to integer
                int64_t int_val = static_cast<int64_t>(scaled);
                
                // Add error sample
                pt_with_error[i] = int_val + error_samples[i];
            }
        });
    }
};