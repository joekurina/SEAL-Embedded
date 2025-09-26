#pragma once

#include "SYCL_ckks_sym.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "SYCL_pipes.h"
#include <cstdio>
#include <cstdlib>

class NTTKernel_A {
private:
    size_t n;
    size_t logn;
    uint32_t mod_value;
    uint32_t root;
    const uint32_t* const_ratio;
    mutable sycl::buffer<uint32_t, 1> vec_acc;    // Input buffer
    mutable sycl::buffer<uint32_t, 1> save_acc;   // Save destination buffer
    
    
public:
    // Constructor accepting both primary and save buffers
    NTTKernel_A(size_t n_val, size_t logn_val, uint32_t mod_val, uint32_t root_val,
                const uint32_t* const_ratio_val,
                sycl::buffer<uint32_t, 1>& vec_buf,  // Input (NTT input)
                sycl::buffer<uint32_t, 1>& save_buf) // Out (Save)
        : n(n_val), logn(logn_val), mod_value(mod_val), root(root_val),
            const_ratio(const_ratio_val),
            vec_acc(vec_buf),  // Initialize primary buffer member
            save_acc(save_buf) // Initialize save buffer member
            {}

    void operator()(sycl::handler& h) const {
        // Accessor for primary buffer (read only)
        auto data = vec_acc.get_access<sycl::access::mode::read>(h);
        // Accessor for save buffer (write only)
        auto s_save = save_acc.get_access<sycl::access::mode::write>(h);

        // Capture necessary variables for the kernel lambda
        size_t kernel_n = n;
        size_t kernel_logn = logn;
        uint32_t kernel_mod_val = mod_value;
        uint32_t kernel_root = root;
        const uint32_t* kernel_const_ratio = const_ratio;

        // Perform host-side check if save buffer is valid before launching kernel
        bool save_output = (save_acc.get_range() == sycl::range(kernel_n));

        h.single_task([=]() [[intel::kernel_args_restrict]] {
            size_t hsize = 1;
            size_t tt = kernel_n / 2;
            uint32_t output_data[PIPE_CAPACITY];

            // Loop over stages
            for (size_t i = 0; i < kernel_logn; i++, hsize *= 2, tt /= 2) {
                for (size_t j = 0, kstart = 0; j < hsize; j++, kstart += 2 * tt) {
                    // Compute twiddle factor exponent
                    uint32_t power = hsize + j;
                    uint32_t s;

                    if (power == 0) {
                        s = 1;
                    } else if (power == (1 << (kernel_logn - 1))) {
                        s = kernel_root;
                    } else {
                        // Inline exponentiation: calculate s = root^power mod mod_val
                        uint32_t current_power = kernel_root;
                        uint32_t result = 1;
                        size_t shift_count = kernel_logn - 1;

                        while (true) {
                            if (power & ((uint32_t)1 << shift_count)) {
                                // Equivalent to mul_mod(current_power, result, mod)
                                uint32_t product[2];
                                uint64_t res_temp = (uint64_t)current_power * (uint64_t)result;
                                product[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                product[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);

                                // Equivalent to barrett_reduce_wide(product, mod)
                                // Round 1
                                uint32_t right_hw;
                                {
                                    uint32_t res[2];
                                    uint64_t res_temp_br1 = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[0]; // Unique temp var
                                    res[0] = (uint32_t)(res_temp_br1 & 0xFFFFFFFF);
                                    res[1] = (uint32_t)((res_temp_br1 >> 32) & 0xFFFFFFFF);
                                    right_hw = res[1];
                                }
                                uint32_t middle_temp[2];
                                {
                                    uint64_t res_temp_br2 = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[1]; // Unique temp var
                                    middle_temp[0] = (uint32_t)(res_temp_br2 & 0xFFFFFFFF);
                                    middle_temp[1] = (uint32_t)((res_temp_br2 >> 32) & 0xFFFFFFFF);
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
                                    uint64_t res_temp_br3 = (uint64_t)product[1] * (uint64_t)kernel_const_ratio[0]; // Unique temp var
                                    middle2_temp[0] = (uint32_t)(res_temp_br3 & 0xFFFFFFFF);
                                    middle2_temp[1] = (uint32_t)((res_temp_br3 >> 32) & 0xFFFFFFFF);
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
                                // Final reduction
                                int32_t is_ge_q = (int32_t)(tmp >= kernel_mod_val);
                                uint32_t mask = (uint32_t)(-is_ge_q);
                                result = (uint32_t)(tmp) - (kernel_mod_val & mask);
                            }

                            power &= ~((uint32_t)1 << shift_count);
                            if (power == 0) {
                                s = result;
                                break;
                            }

                            // Equivalent to mul_mod(current_power, current_power, mod)
                            uint32_t product_sq[2];
                            uint64_t res_temp_sq = (uint64_t)current_power * (uint64_t)current_power;
                            product_sq[0] = (uint32_t)(res_temp_sq & 0xFFFFFFFF);
                            product_sq[1] = (uint32_t)((res_temp_sq >> 32) & 0xFFFFFFFF);

                            // Equivalent to barrett_reduce_wide(product_sq, mod)
                            // Round 1
                            uint32_t right_hw_sq;
                            {
                                uint32_t res[2];
                                uint64_t res_temp_br4 = (uint64_t)product_sq[0] * (uint64_t)kernel_const_ratio[0]; // Unique temp var
                                res[0] = (uint32_t)(res_temp_br4 & 0xFFFFFFFF);
                                res[1] = (uint32_t)((res_temp_br4 >> 32) & 0xFFFFFFFF);
                                right_hw_sq = res[1];
                            }
                            uint32_t middle_temp_sq[2];
                            {
                                uint64_t res_temp_br5 = (uint64_t)product_sq[0] * (uint64_t)kernel_const_ratio[1]; // Unique temp var
                                middle_temp_sq[0] = (uint32_t)(res_temp_br5 & 0xFFFFFFFF);
                                middle_temp_sq[1] = (uint32_t)((res_temp_br5 >> 32) & 0xFFFFFFFF);
                            }
                            uint32_t middle_lw_sq;
                            uint32_t middle_lw_carry_sq;
                            {
                                middle_lw_sq = right_hw_sq + middle_temp_sq[0];
                                middle_lw_carry_sq = (uint8_t)(middle_lw_sq < right_hw_sq);
                            }
                            uint32_t middle_hw_sq = middle_temp_sq[1] + middle_lw_carry_sq;
                            
                            // Round 2
                            uint32_t middle2_temp_sq[2];
                            {
                                uint64_t res_temp_br6 = (uint64_t)product_sq[1] * (uint64_t)kernel_const_ratio[0]; // Unique temp var
                                middle2_temp_sq[0] = (uint32_t)(res_temp_br6 & 0xFFFFFFFF);
                                middle2_temp_sq[1] = (uint32_t)((res_temp_br6 >> 32) & 0xFFFFFFFF);
                            }
                            uint32_t middle2_lw_sq;
                            uint32_t middle2_lw_carry_sq;
                            {
                                middle2_lw_sq = middle_lw_sq + middle2_temp_sq[0];
                                middle2_lw_carry_sq = (uint8_t)(middle2_lw_sq < middle_lw_sq);
                            }
                            uint32_t middle2_hw_sq = middle2_temp_sq[1] + middle2_lw_carry_sq;
                            uint32_t tmp_sq = product_sq[1] * kernel_const_ratio[1] + middle_hw_sq + middle2_hw_sq;
                            // Barrett subtraction
                            tmp_sq = product_sq[0] - tmp_sq * kernel_mod_val;
                            // Final reduction
                            int32_t is_ge_q_sq = (int32_t)(tmp_sq >= kernel_mod_val);
                            uint32_t mask_sq = (uint32_t)(-is_ge_q_sq);
                            current_power = (uint32_t)(tmp_sq) - (kernel_mod_val & mask_sq);

                            shift_count--;
                        } // End while(true)
                    } // End else (inline exponentiation)

                    // Process each pair in the current group
                    for (size_t k = kstart; k < (kstart + tt); k++) {
                        uint32_t u = data[k];

                        // Equivalent to mul_mod(data[k + tt], s, mod)
                        uint32_t v;
                        {
                            uint32_t product_v[2];
                            uint64_t res_temp_v = (uint64_t)data[k + tt] * (uint64_t)s;
                            product_v[0] = (uint32_t)(res_temp_v & 0xFFFFFFFF);
                            product_v[1] = (uint32_t)((res_temp_v >> 32) & 0xFFFFFFFF);

                            // Equivalent to barrett_reduce_wide(product_v, mod)
                            // Round 1
                            uint32_t right_hw_v;
                            {
                                uint32_t res[2];
                                uint64_t res_temp_br7 = (uint64_t)product_v[0] * (uint64_t)kernel_const_ratio[0]; // Unique temp var
                                res[0] = (uint32_t)(res_temp_br7 & 0xFFFFFFFF);
                                res[1] = (uint32_t)((res_temp_br7 >> 32) & 0xFFFFFFFF);
                                right_hw_v = res[1];
                            }
                            uint32_t middle_temp_v[2];
                            {
                                uint64_t res_temp_br8 = (uint64_t)product_v[0] * (uint64_t)kernel_const_ratio[1]; // Unique temp var
                                middle_temp_v[0] = (uint32_t)(res_temp_br8 & 0xFFFFFFFF);
                                middle_temp_v[1] = (uint32_t)((res_temp_br8 >> 32) & 0xFFFFFFFF);
                            }
                            uint32_t middle_lw_v;
                            uint32_t middle_lw_carry_v;
                            {
                                middle_lw_v = right_hw_v + middle_temp_v[0];
                                middle_lw_carry_v = (uint8_t)(middle_lw_v < right_hw_v);
                            }
                            uint32_t middle_hw_v = middle_temp_v[1] + middle_lw_carry_v;
                            // Round 2
                            uint32_t middle2_temp_v[2];
                            {
                                uint64_t res_temp_br9 = (uint64_t)product_v[1] * (uint64_t)kernel_const_ratio[0]; // Unique temp var
                                middle2_temp_v[0] = (uint32_t)(res_temp_br9 & 0xFFFFFFFF);
                                middle2_temp_v[1] = (uint32_t)((res_temp_br9 >> 32) & 0xFFFFFFFF);
                            }
                            uint32_t middle2_lw_v;
                            uint32_t middle2_lw_carry_v;
                            {
                                middle2_lw_v = middle_lw_v + middle2_temp_v[0];
                                middle2_lw_carry_v = (uint8_t)(middle2_lw_v < middle_lw_v);
                            }
                            uint32_t middle2_hw_v = middle2_temp_v[1] + middle2_lw_carry_v;
                            uint32_t tmp_v = product_v[1] * kernel_const_ratio[1] + middle_hw_v + middle2_hw_v;
                            // Barrett subtraction
                            tmp_v = product_v[0] - tmp_v * kernel_mod_val;
                            // Final reduction
                            int32_t is_ge_q_v = (int32_t)(tmp_v >= kernel_mod_val);
                            uint32_t mask_v = (uint32_t)(-is_ge_q_v);
                            v = (uint32_t)(tmp_v) - (kernel_mod_val & mask_v);
                        }

                        // Equivalent to add_mod(u, v, mod)
                        uint32_t result_add = u + v;
                        if (result_add >= kernel_mod_val) {
                            result_add -= kernel_mod_val;
                        }

                        // Equivalent to sub_mod(u, v, mod)
                        uint32_t result_sub;
                        { // Scope for negation intermediate vars
                            uint32_t negated_v; // Renamed from 'negated'
                            int32_t non_zero = (int32_t)(v != 0);
                            uint32_t neg_mask = (uint32_t)(-non_zero); // Renamed from 'mask'
                            negated_v = (kernel_mod_val - v) & neg_mask;
                            result_sub = u + negated_v;
                            if (result_sub >= kernel_mod_val) {
                                result_sub -= kernel_mod_val;
                            }
                        }

                        output_data[k] = result_add;
                        output_data[k + tt] = result_sub; 

                    } // End loop k
                } // End loop j
            } // End loop i (stages)

            // Write the results to the pipe
            for (size_t i = 0; i < kernel_n; ++i) {
                NTTToPolyMultNegPipe::write(output_data[i]);
            }

            if (save_output) {
                // Write the results to the save buffer
                for (size_t i = 0; i < kernel_n; ++i) {
                    s_save[i] = output_data[i];
                }
            }
        }); // End single_task lambda
    } // End operator()
}; // End of NTTKernel_A class