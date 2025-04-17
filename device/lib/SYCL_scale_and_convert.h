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
        // Create a stream for debugging
        sycl::stream kernel_dbg_stream(1024 * 8, 256, h);

        // Capture kernel variables
        size_t kernel_n = n;
        double kernel_scale = scale;
        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Announce the start of the kernel
            kernel_dbg_stream << "ScaleAndConvertKernel: Starting..." << sycl::endl;

            // Calculate scaling factor
            double n_inv = kernel_scale / static_cast<double>(kernel_n);

            // Process each element
            for (size_t i = 0; i < kernel_n; i++) {
                // Blocking Read from input pipes
                std::complex<double> encoded_value = IFFTToScaleAndConvertPipe::read();
                int8_t error_value = IFFTErrorToScaleAndConvertPipe::read();

                // Get real part of complex value
                double real_val = encoded_value.real();

                // Scale and round
                double scaled = sycl::round(real_val * n_inv);

                // Convert to integer and add error
                int64_t int_val = static_cast<int64_t>(scaled);
                int64_t result = int_val + error_value;

                // *** Non-Blocking Write Loop ***
                // Keep trying to write the 'result' for the current 'i' until successful.
                bool write_succeeded = false;
                bool first_write_attempt = true;

                while (!write_succeeded) {
                    // Debug message only on the first attempt for this 'i' to avoid spam
                    if (first_write_attempt && (i == 0 || i == kernel_n - 1)) {
                        kernel_dbg_stream << "ScaleAndConvertKernel: Loop i=" << i
                                          << ", attempting write of " << result
                                          << " to ScaleToReducePipe..." << sycl::endl;
                    }

                    // Attempt a non-blocking write
                    ScaleToReducePipe::write(result, write_succeeded);

                    if (write_succeeded) {
                        // Write succeeded for the first and last iterations
                        if (i == 0 || i == kernel_n - 1) {
                            kernel_dbg_stream << "ScaleAndConvertKernel: Loop i=" << i
                                              << ", write SUCCESS." << sycl::endl;
                        }
                        // Exit the inner retry loop
                        break;
                    } else {
                        // Write failed, the inner loop will repeat.
                        first_write_attempt = false; // No longer the first attempt
                    }
                } // End of non-blocking write retry loop for index i

                // Increment main loop counter only after the write for index 'i' has succeeded.
                i++;
            } // End of main loop

            // Announce completion
            kernel_dbg_stream << "ScaleAndConvertKernel: Loop finished after " << kernel_n << " iterations." << sycl::endl;
            kernel_dbg_stream << "ScaleAndConvertKernel: Finished." << sycl::endl;
        });
    }
};