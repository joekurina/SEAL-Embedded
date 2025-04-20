#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_pipes.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp> // Provides printf according to user

#include <cstdint> // Original include

// Kernel for scaling and conversion to integers, then adding the error samples
class ScaleAndConvertKernel {
private:
    size_t n;
    double scale;

public:
    ScaleAndConvertKernel(size_t n_val, double scale_val)
        : n(n_val), scale(scale_val) {}

    void operator()(sycl::handler& h) const {

        // Capture kernel variables
        size_t kernel_n = n;
        double kernel_scale = scale;

        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Restore printf: Announce the start of the kernel
            sycl::ext::oneapi::experimental::printf("ScaleAndConvertKernel: Starting...\n");

            // Calculate scaling factor
            double n_inv = kernel_scale / static_cast<double>(kernel_n);

            // Process each element using original loop structure
            for (size_t i = 0; i < kernel_n; /* i incremented below */ ) {
                // Blocking Read from input pipes
                // Assuming std::complex is available via includes
                std::complex<double> encoded_value = IFFTToScaleAndConvertPipe::read();
                int8_t error_value = IFFTErrorToScaleAndConvertPipe::read();

                // Get real part of complex value
                double real_val = encoded_value.real();

                // Scale and round
                double scaled = sycl::round(real_val * n_inv);

                // Convert to integer and add error
                int64_t int_val = static_cast<int64_t>(scaled);
                int64_t result = int_val + error_value;

                // *** Restore Original Non-Blocking Write Loop ***
                bool write_succeeded = false;
                bool first_write_attempt = true;

                while (!write_succeeded) {
                    // Restore printf: Debug message only on the first attempt for this 'i' to avoid spam
                    if (first_write_attempt && (i == 0 || i == kernel_n - 1)) {
                        // Assuming i is size_t (%zu) and result is int64_t (%lld)
                       sycl::ext::oneapi::experimental::printf(
                           "ScaleAndConvertKernel: Loop i=%zu, attempting write of %lld to ScaleToReducePipe...\n",
                           i, result);
                    }

                    // Attempt a non-blocking write
                    ScaleToReducePipe::write(result, write_succeeded);

                    if (write_succeeded) {
                        // Restore printf: Write succeeded for the first and last iterations
                        if (i == 0 || i == kernel_n - 1) {
                            // Assuming i is size_t (%zu)
                           sycl::ext::oneapi::experimental::printf(
                               "ScaleAndConvertKernel: Loop i=%zu, write SUCCESS.\n", i);
                        }
                        // Exit the inner retry loop
                        break;
                    } else {
                        // Write failed, the inner loop will repeat.
                        first_write_attempt = false; // No longer the first attempt
                    }
                } // End of non-blocking write retry loop for index i

                // Increment main loop counter only after the write for index 'i' has succeeded. (Original logic)
                i++;

            } // End of main loop

            // Restore printf: Announce completion
            // Assuming kernel_n is size_t (%zu)
            sycl::ext::oneapi::experimental::printf(
                "ScaleAndConvertKernel: Loop finished after %zu iterations.\n", kernel_n);
            sycl::ext::oneapi::experimental::printf("ScaleAndConvertKernel: Finished.\n");
        });
    }
};