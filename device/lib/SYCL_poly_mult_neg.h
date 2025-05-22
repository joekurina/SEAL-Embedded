#pragma once

#include "SYCL_ckks_sym.h" // Assuming this contains necessary base types
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Combined Kernel for In-place Polynomial Multiplication followed by Negation in NTT form
// Computes: a = -(a * b) mod q
class PolyMultNegNTTKernel {
private:
    size_t n;
    uint32_t mod_value;
    const uint32_t* const_ratio; // For Barrett reduction in multiplication
    mutable sycl::buffer<uint32_t, 1> b_acc; // Input buffer

public:
    PolyMultNegNTTKernel(size_t n_val, uint32_t mod_val, const uint32_t* const_ratio_val,
                         sycl::buffer<uint32_t, 1>& b_buf) // In
        : n(n_val),
          mod_value(mod_val),
          const_ratio(const_ratio_val),
          b_acc(b_buf) {}

    void operator()(sycl::handler& h) const {
        // Get access to buffer
        auto b = b_acc.get_access<sycl::access::mode::read>(h);

        // Capture necessary variables
        size_t kernel_n = n;
        uint32_t kernel_mod_val = mod_value;
        const uint32_t* kernel_const_ratio = const_ratio;

        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Process each coefficient
            for (size_t i = 0; i < kernel_n; i++) {
                // Get initial values
                uint32_t a_val = NTTToPolyMultNegPipe::read(); // Read from the pipe
                uint32_t b_val = b[i];

                // --- Step 1: Polynomial Multiplication (a_val * b_val) mod q ---
                // Logic copied directly from PolyMultNTTKernel with original formatting
                uint32_t mult_result;
                {
                    // 1. Multiply to get wide result
                    uint64_t res_temp = (uint64_t)a_val * (uint64_t)b_val;
                    uint32_t product[2];
                    product[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                    product[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);

                    // 2. Barrett reduction starts here
                    // Round 1
                    uint32_t right_hw;
                    {
                        uint32_t res[2];
                        uint64_t rt_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[0];
                        res[0] = (uint32_t)(rt_temp & 0xFFFFFFFF);
                        res[1] = (uint32_t)((rt_temp >> 32) & 0xFFFFFFFF);
                        right_hw = res[1];
                    }

                    uint32_t middle_temp[2];
                    {
                        uint64_t mt_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[1];
                        middle_temp[0] = (uint32_t)(mt_temp & 0xFFFFFFFF);
                        middle_temp[1] = (uint32_t)((mt_temp >> 32) & 0xFFFFFFFF);
                    }

                    uint32_t middle_lw;
                    uint32_t middle_lw_carry;
                    {
                        middle_lw = right_hw + middle_temp[0];
                        middle_lw_carry = (uint8_t)(middle_lw < right_hw);
                    }

                    uint32_t middle_hw = middle_temp[1] + middle_lw_carry;

                    // Round 2
                    uint32_t middle2_temp[2];
                    {
                        uint64_t mt2_temp = (uint64_t)product[1] * (uint64_t)kernel_const_ratio[0];
                        middle2_temp[0] = (uint32_t)(mt2_temp & 0xFFFFFFFF);
                        middle2_temp[1] = (uint32_t)((mt2_temp >> 32) & 0xFFFFFFFF);
                    }

                    uint32_t middle2_lw;
                    uint32_t middle2_lw_carry;
                    {
                        middle2_lw = middle_lw + middle2_temp[0];
                        middle2_lw_carry = (uint8_t)(middle2_lw < middle_lw);
                    }

                    uint32_t middle2_hw = middle2_temp[1] + middle2_lw_carry;

                    uint32_t tmp = product[1] * kernel_const_ratio[1] + middle_hw + middle2_hw;

                    // Barrett subtraction
                    tmp = product[0] - tmp * kernel_mod_val;

                    // Final reduction if needed
                    // Note: Original PolyMultNTTKernel used '>=' check here which is standard for Barrett.
                    // If result can be exactly 'q', this reduces it to 0.
                    int32_t is_ge_q = (int32_t)(tmp >= kernel_mod_val);
                    uint32_t mask_red = (uint32_t)(-is_ge_q);
                    mult_result = tmp - (kernel_mod_val & mask_red); // Store result of multiplication
                } // End of multiplication logic


                // --- Step 2: Polynomial Negation (-mult_result) mod q ---
                // Logic copied directly from PolyNegModKernel, applied to mult_result
                uint32_t neg_result;
                {
                    uint32_t coeff_to_negate = mult_result; // Use the multiplication result

                    // Compute if coefficient is non-zero
                    int32_t non_zero = (int32_t)(coeff_to_negate != 0);
                    uint32_t mask_neg = (uint32_t)(-non_zero);

                    // Compute negation: if coeff == 0, result = 0; else result = q - coeff
                    neg_result = (kernel_mod_val - coeff_to_negate) & mask_neg;
                } // End of negation logic


                // --- Step 3: Write the negated result to the pipe for further processing
                PolyMultNegToPolyAddModPipe::write(neg_result); // Write to the pipe
            }
        });
    }
};