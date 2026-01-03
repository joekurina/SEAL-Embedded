#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_pipes.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdint>

template <int P>
class ScaleAndReduceKernelTask;

// Merged Kernel: Performs Scaling/Conversion and Reduction (templated on pipeline index)
template <int P>
class ScaleAndReduceKernelT 
{
private:
    size_t n;
    double scale;
    uint32_t mod_value;
    uint32_t const_ratio0;
    uint32_t const_ratio1;
    mutable sycl::buffer<i8x4_input, 1> error_samples_acc;

public:
    // Constructor takes combined arguments
                ScaleAndReduceKernelT(size_t n_val, double scale_val, uint32_t mod_val,
                                                                                                        uint32_t const_ratio0_val,
                                                                                                        uint32_t const_ratio1_val,
                                                                                                        sycl::buffer<i8x4_input, 1>& error_samples_buf)
        : n(n_val),
          scale(scale_val),
          mod_value(mod_val),
                    const_ratio0(const_ratio0_val),
                    const_ratio1(const_ratio1_val),
          error_samples_acc(error_samples_buf) {}

    void operator()(sycl::handler& h) const 
    {
        // Get access to the error samples buffer
        auto error_blocks = error_samples_acc.get_access<sycl::access::mode::read>(h);

        // Capture necessary variables
        size_t kernel_n = n; 
        double kernel_scale = scale;
        uint32_t kernel_mod_val = mod_value;
        uint32_t kernel_const_ratio0 = const_ratio0;
        uint32_t kernel_const_ratio1 = const_ratio1;

        h.single_task<ScaleAndReduceKernelTask<P>>([=]() [[intel::kernel_args_restrict]] 
        {
            double n_inv = kernel_scale / static_cast<double>(kernel_n);

            for (size_t blk = 0; blk < kernel_n / 4; ++blk) {
                using PipeSet = CKKS_PIPE_SET<P>;
                encoding_buffer_input enc_block = PipeSet::IFFTToScaleAndReducePipe::read();
                i8x4_input err_block = error_blocks[blk];

                std::complex<double> enc_vals[4] = {enc_block.element0, enc_block.element1, enc_block.element2, enc_block.element3};
                int8_t err_vals[4] = {err_block.element0, err_block.element1, err_block.element2, err_block.element3};

                u32x4_input out_block{};

                for (size_t lane = 0; lane < 4; ++lane) {
                    double real_val = enc_vals[lane].real();
                    double scaled = sycl::round(real_val * n_inv);
                    int64_t int_val = static_cast<int64_t>(scaled);
                    int64_t intermediate_result = int_val + err_vals[lane];

                    int64_t val = intermediate_result;
                    uint64_t coeff_abs = (val < 0) ? static_cast<uint64_t>(-val) : static_cast<uint64_t>(val);
                    uint32_t mask = static_cast<uint32_t>(val < 0);

                    uint32_t coeff_abs_vec[2];
                    coeff_abs_vec[0] = static_cast<uint32_t>(coeff_abs & 0xFFFFFFFF);
                    coeff_abs_vec[1] = static_cast<uint32_t>((coeff_abs >> 32) & 0xFFFFFFFF);

                    uint32_t right_hw;
                    {
                        uint64_t res_temp = (uint64_t)coeff_abs_vec[0] * (uint64_t)kernel_const_ratio0;
                        right_hw = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                    }
                    uint32_t middle_temp[2];
                    {
                        uint64_t res_temp = (uint64_t)coeff_abs_vec[0] * (uint64_t)kernel_const_ratio1;
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
                        uint64_t res_temp = (uint64_t)coeff_abs_vec[1] * (uint64_t)kernel_const_ratio0;
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
                    uint32_t tmp = coeff_abs_vec[1] * kernel_const_ratio1 + middle_hw + middle2_hw;

                    tmp = coeff_abs_vec[0] - tmp * kernel_mod_val;

                    uint32_t coeff_crt;
                    {
                        int32_t is_2q = (int32_t)(tmp >= kernel_mod_val);
                        uint32_t tmp_mask = (uint32_t)(-is_2q);
                        coeff_crt = (uint32_t)(tmp) - (kernel_mod_val & tmp_mask);
                    }

                    uint32_t final_result = ((kernel_mod_val - coeff_crt) & (-mask)) + (coeff_crt & (mask - 1));

                    reinterpret_cast<uint32_t*>(&out_block)[lane] = final_result;
                }

                PipeSet::ScaleReduceToNTTBPipe::write(out_block); 
            } // End of block loop
        }); // End single_task
    } // End operator()
}; // End of ScaleAndReduceKernelT class

// Backwards-compatible alias for pipeline P = 0.
using ScaleAndReduceKernel = ScaleAndReduceKernelT<0>;