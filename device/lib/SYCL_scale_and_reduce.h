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
            // --- Local array to collect all complex values from IFFT ---
            std::complex<double> local_encoded_data[PIPE_CAPACITY];

            // Bit-reversal function (same as original IFFT)
            auto bitrev = [](size_t input, size_t numbits) -> size_t 
            {
                size_t t = (((input & 0xaaaa) >> 1) | ((input & 0x5555) << 1));
                t        = (((t & 0xcccc) >> 2) | ((t & 0x3333) << 2));
                t        = (((t & 0xf0f0) >> 4) | ((t & 0x0f0f) << 4));
                t        = (((t & 0xff00) >> 8) | ((t & 0x00ff) << 8));
                return (numbits == 0) ? 0 : (t >> (16 - numbits));
            };
            
            size_t kernel_logn = 0;
            size_t temp_n = kernel_n;
            while (temp_n > 1) {
                temp_n >>= 1;
                kernel_logn++;
            }

            // --- Collection Phase: Read FFT_Output_Data structures and extract complex values ---
            size_t num_chunks = kernel_n / 4;
            size_t data_index = 0;
            
            for (size_t chunk = 0; chunk < num_chunks; chunk++) {
                FFT_Output_Data output_data = IFFTToScaleAndReducePipe::read();
                
                // Extract 4 complex values from the structure in sequential order
                local_encoded_data[data_index++] = std::complex<double>(output_data.port_data_out_0re, output_data.port_data_out_0im);
                local_encoded_data[data_index++] = std::complex<double>(output_data.port_data_out_1re, output_data.port_data_out_1im);
                local_encoded_data[data_index++] = std::complex<double>(output_data.port_data_out_2re, output_data.port_data_out_2im);
                local_encoded_data[data_index++] = std::complex<double>(output_data.port_data_out_3re, output_data.port_data_out_3im);
            }

            // --- Processing Phase: Perform scaling and modular reduction ---
            // Original scaling factor, but compensate for RTL FFT's internal 1/N scaling
            double n_inv = kernel_scale / static_cast<double>(kernel_n);

            for (size_t i = 0; i < kernel_n; i++) {
                std::complex<double> encoded_value = local_encoded_data[i]; 

                // Back to original scaling approach
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