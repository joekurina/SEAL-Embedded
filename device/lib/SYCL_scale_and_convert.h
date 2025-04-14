#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_pipes.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdint>

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

        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Calculate scaling factor
            double n_inv = kernel_scale / static_cast<double>(kernel_n);

            // Process each element
            for (size_t i = 0; i < kernel_n; i++) {
                // Read from input pipes
                std::complex<double> encoded_value = IFFTToScaleAndConvertPipe::read();
                int8_t error_value = IFFTErrorToScaleAndConvertPipe::read();

                // Get real part of complex value
                double real_val = encoded_value.real();

                // Scale and round
                double scaled = sycl::round(real_val * n_inv);

                // Convert to integer and add error
                int64_t int_val = static_cast<int64_t>(scaled);
                int64_t result = int_val + error_value;

                // Write to output pipe
                ScaleToReducePipe::write(result);
            }
        });
    }
};