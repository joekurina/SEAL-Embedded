#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_pipes.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdint>

// Kernel for reducing int64_t values to their representation in ring Z_q
class ReduceSetPTEKernel {
private:
    size_t n;
    uint32_t mod_value;
    const uint32_t* const_ratio;
    mutable sycl::buffer<uint32_t, 1> out_acc; // Output buffer

public:
    // Modify constructor to remove input buffer parameter:
    ReduceSetPTEKernel(size_t n_val, uint32_t mod_val, const uint32_t* const_ratio_val,
                        sycl::buffer<uint32_t, 1>& out_buf) // Output buffer
        : n(n_val), mod_value(mod_val), const_ratio(const_ratio_val),
          out_acc(out_buf) {} // Output buffer initializer

    void operator()(sycl::handler& h) const {
        auto out = out_acc.get_access<sycl::access::mode::write>(h);

        // Capture necessary variables
        size_t kernel_n = n;
        uint32_t kernel_mod_val = mod_value;
        const uint32_t* kernel_const_ratio = const_ratio;

        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Process each coefficient
            for (size_t i = 0; i < kernel_n; i++) {
                // Read input value from the pipe:
                int64_t val = ScaleToReducePipe::read();

                // Compute absolute value
                uint64_t coeff_abs = (val < 0) ? static_cast<uint64_t>(-val) : static_cast<uint64_t>(val);

                // Create mask based on sign (1 if negative, 0 if positive)
                uint32_t mask = static_cast<uint32_t>(val < 0);

                // Split 64-bit value into two 32-bit parts for Barrett reduction
                uint32_t coeff_abs_vec[2];
                coeff_abs_vec[0] = static_cast<uint32_t>(coeff_abs & 0xFFFFFFFF);
                coeff_abs_vec[1] = static_cast<uint32_t>((coeff_abs >> 32) & 0xFFFFFFFF);

                // -- Round 1
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

                // -- Round 2
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
                uint32_t result = ((kernel_mod_val - coeff_crt) & (-mask)) + (coeff_crt & (mask - 1));

                // Store the result
                out[i] = result;
            }
        });
    }
};