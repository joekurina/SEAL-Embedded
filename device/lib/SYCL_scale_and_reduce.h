#pragma once

#include "SYCL_ckks_sym.h" // For complex_double if needed, etc.
#include "SYCL_pipes.h"    // For IFFTToScaleAndConvertPipe, IFFTErrorToScaleAndConvertPipe
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdint>

// Merged Kernel: Performs Scaling/Conversion and Reduction
class ScaleAndReduceKernel {
private:
    size_t n;
    double scale;             // Argument from ScaleAndConvertKernel
    uint32_t mod_value;       // Argument from ReduceSetPTEKernel
    const uint32_t* const_ratio; // Argument from ReduceSetPTEKernel
    mutable sycl::buffer<uint32_t, 1> out_acc; // Output buffer

public:
    // Constructor takes combined arguments
    ScaleAndReduceKernel(size_t n_val, double scale_val, uint32_t mod_val,
                         const uint32_t* const_ratio_val,
                         sycl::buffer<uint32_t, 1>& out_buf)
        : n(n_val),
          scale(scale_val),
          mod_value(mod_val),
          const_ratio(const_ratio_val),
          out_acc(out_buf) {}

    void operator()(sycl::handler& h) const {
        // Get access to the output buffer
        auto out = out_acc.get_access<sycl::access::mode::write>(h);

        // Capture necessary variables for the kernel lambda
        size_t kernel_n = n;
        double kernel_scale = scale;
        uint32_t kernel_mod_val = mod_value;
        const uint32_t* kernel_const_ratio = const_ratio;

        sycl::ext::oneapi::experimental::printf("ScaleAndReduceKernel: Starting...\n");

        h.single_task([=]() [[intel::kernel_args_restrict]] {
            sycl::ext::oneapi::experimental::printf("ScaleAndReduceKernel: Kernel Started...\n");

            // Pre-calculate scaling factor (from ScaleAndConvertKernel)
            double n_inv = kernel_scale / static_cast<double>(kernel_n);

            // Process each coefficient
            for (size_t i = 0; i < kernel_n; i++) {

                // --- Part 1: Logic from ScaleAndConvertKernel ---

                // Blocking Read from input pipes coming from IFFTKernel
                //sycl::ext::oneapi::experimental::printf("ScaleAndReduceKernel: Reading from IFFT Pipes (i=%zu)...\n", i);
                std::complex<double> encoded_value = IFFTToScaleAndConvertPipe::read();
                int8_t error_value = IFFTErrorToScaleAndConvertPipe::read();

                // Get real part, scale, round, add error
                double real_val = encoded_value.real();
                double scaled = sycl::round(real_val * n_inv);
                int64_t int_val = static_cast<int64_t>(scaled);
                int64_t intermediate_result = int_val + error_value; // This value was previously written to ScaleToReducePipe

                //sycl::ext::oneapi::experimental::printf("ScaleAndReduceKernel: Intermediate value (i=%zu) = %lld\n", i, intermediate_result);


                // --- Part 2: Logic from ReduceSetPTEKernel ---
                // Use 'intermediate_result' directly instead of reading from pipe

                int64_t val = intermediate_result; // Use the calculated value

                // Compute absolute value
                uint64_t coeff_abs = (val < 0) ? static_cast<uint64_t>(-val) : static_cast<uint64_t>(val);

                // Create mask based on sign
                uint32_t mask = static_cast<uint32_t>(val < 0);

                // Split 64-bit value for Barrett reduction
                uint32_t coeff_abs_vec[2];
                coeff_abs_vec[0] = static_cast<uint32_t>(coeff_abs & 0xFFFFFFFF);
                coeff_abs_vec[1] = static_cast<uint32_t>((coeff_abs >> 32) & 0xFFFFFFFF);

                // -- Round 1 (Barrett reduction)
                uint32_t right_hw;
                {
                    uint64_t res_temp = (uint64_t)coeff_abs_vec[0] * (uint64_t)kernel_const_ratio[0];
                    right_hw = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                }
                uint32_t middle_temp[2];
                {
                    uint64_t res_temp = (uint64_t)coeff_abs_vec[0] * (uint64_t)kernel_const_ratio[1];
                    middle_temp[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                    middle_temp[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                }
                uint32_t middle_lw;
                uint32_t middle_lw_carry;
                {
                    middle_lw = right_hw + middle_temp[0];
                    middle_lw_carry = (uint8_t)(middle_lw < right_hw);
                }
                uint32_t middle_hw = middle_temp[1] + middle_lw_carry;

                // -- Round 2 (Barrett reduction)
                uint32_t middle2_temp[2];
                {
                    uint64_t res_temp = (uint64_t)coeff_abs_vec[1] * (uint64_t)kernel_const_ratio[0];
                    middle2_temp[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                    middle2_temp[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                }
                uint32_t middle2_lw;
                uint32_t middle2_lw_carry;
                {
                    middle2_lw = middle_lw + middle2_temp[0];
                    middle2_lw_carry = (uint8_t)(middle2_lw < middle_lw);
                }
                uint32_t middle2_hw = middle2_temp[1] + middle2_lw_carry;
                uint32_t tmp = coeff_abs_vec[1] * kernel_const_ratio[1] + middle_hw + middle2_hw;

                // -- Barrett subtraction
                tmp = coeff_abs_vec[0] - tmp * kernel_mod_val;

                // -- Final reduction if needed
                uint32_t coeff_crt;
                {
                    int32_t is_2q = (int32_t)(tmp >= kernel_mod_val);
                    uint32_t tmp_mask = (uint32_t)(-is_2q);
                    coeff_crt = (uint32_t)(tmp) - (kernel_mod_val & tmp_mask);
                }

                // Compute final result based on sign
                uint32_t final_result = ((kernel_mod_val - coeff_crt) & (-mask)) + (coeff_crt & (mask - 1));

                // Store the final result directly to the output buffer
                out[i] = final_result;

                // Debug message for first and last iterations
                if (i == 0 || i == kernel_n -1) {
                     sycl::ext::oneapi::experimental::printf(
                        "ScaleAndReduceKernel: Loop i=%zu, wrote result %u to output buffer.\n", i, final_result);
                }
            } // End of for loop

            // Debug message after loop completion
             sycl::ext::oneapi::experimental::printf(
                "ScaleAndReduceKernel: Loop finished after %zu iterations.\n", kernel_n);
             sycl::ext::oneapi::experimental::printf(
                "ScaleAndReduceKernel: Finished.\n");
        }); // End single_task
    } // End operator()
}; // End of ScaleAndReduceKernel class