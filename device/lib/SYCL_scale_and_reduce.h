#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_pipes.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdint>

// Merged Kernel: Performs Scaling/Conversion and Reduction
class ScaleAndReduceKernel 
{
private:
    size_t n;
    double scale;
    uint32_t mod_value;
    const uint32_t* const_ratio;
    mutable sycl::buffer<int8_t, 1> error_samples_acc;

public:
    // Constructor takes combined arguments
    ScaleAndReduceKernel(size_t n_val, double scale_val, uint32_t mod_val,
                         const uint32_t* const_ratio_val,
                         sycl::buffer<int8_t, 1>& error_samples_buf)
        : n(n_val),
          scale(scale_val),
          mod_value(mod_val),
          const_ratio(const_ratio_val), 
          error_samples_acc(error_samples_buf) {}

    void operator()(sycl::handler& h) const 
    {
        // Get access to the error samples buffer
        auto error_samples = error_samples_acc.get_access<sycl::access::mode::read>(h);

        // Capture necessary variables
        size_t kernel_n = n; 
        double kernel_scale = scale;
        uint32_t kernel_mod_val = mod_value;
        const uint32_t* kernel_const_ratio = const_ratio;

        h.single_task([=]() [[intel::kernel_args_restrict]] 
        {
            // --- Local array to buffer pipe data ---
            std::complex<double> local_encoded_data[PIPE_CAPACITY];

            // --- Processing Phase ---
            double n_inv = kernel_scale / static_cast<double>(kernel_n);

            for (size_t i = 0; i < kernel_n; i++) {
                std::complex<double> encoded_value = IFFTToScaleAndReducePipe::read(); 

                double real_val = encoded_value.real();
                double scaled = sycl::round(real_val * n_inv);
                int64_t int_val = static_cast<int64_t>(scaled);
                int64_t intermediate_result = int_val + error_samples[i];

                int64_t val = intermediate_result;
                uint64_t coeff_abs = (val < 0) ? static_cast<uint64_t>(-val) : static_cast<uint64_t>(val);
                uint32_t mask = static_cast<uint32_t>(val < 0);

                uint32_t coeff_abs_vec[2];
                coeff_abs_vec[0] = static_cast<uint32_t>(coeff_abs & 0xFFFFFFFF);
                coeff_abs_vec[1] = static_cast<uint32_t>((coeff_abs >> 32) & 0xFFFFFFFF);

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

                tmp = coeff_abs_vec[0] - tmp * kernel_mod_val;

                uint32_t coeff_crt;
                {
                    int32_t is_2q = (int32_t)(tmp >= kernel_mod_val);
                    uint32_t tmp_mask = (uint32_t)(-is_2q);
                    coeff_crt = (uint32_t)(tmp) - (kernel_mod_val & tmp_mask);
                }

                uint32_t final_result = ((kernel_mod_val - coeff_crt) & (-mask)) + (coeff_crt & (mask - 1));

                ScaleReduceToNTTBPipe::write(final_result); 
            } // End of for loop
        }); // End single_task
    } // End operator()
}; // End of ScaleAndReduceKernel class