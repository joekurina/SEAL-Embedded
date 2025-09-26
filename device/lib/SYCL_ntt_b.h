#pragma once

#include "SYCL_ckks_sym.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "SYCL_pipes.h"
#include <cstdio>
#include <cstdlib>

// NTT Kernel functor class
class NTTKernel_B {
private:
    size_t n;
    size_t logn;
    uint32_t mod_value;
    uint32_t root;
    const uint32_t* const_ratio;
    mutable sycl::buffer<uint32_t, 1> result_out_acc;
    mutable sycl::buffer<uint32_t, 1> ntt_b_input_buffer; // Intermediate buffer for testing/debugging
    mutable sycl::buffer<uint32_t, 1> ntt_b_output_buffer; // Intermediate buffer for testing/debugging 

public:
    NTTKernel_B(size_t n_val, size_t logn_val, uint32_t mod_val, uint32_t root_val,
                const uint32_t* const_ratio_val, 
                sycl::buffer<uint32_t, 1>& result_output_buf,
                sycl::buffer<uint32_t, 1>& ntt_b_input_buffer, // Intermediate buffer for testing/debugging
                sycl::buffer<uint32_t, 1>& ntt_b_output_buffer  // Intermediate buffer for testing/debugging
               ) 
        : n(n_val), logn(logn_val), mod_value(mod_val), root(root_val),
            const_ratio(const_ratio_val), result_out_acc(result_output_buf),
            ntt_b_input_buffer(ntt_b_input_buffer),
            ntt_b_output_buffer(ntt_b_output_buffer) {} 
    
    void operator()(sycl::handler& h) const {
        // Get write access to the output buffer
        auto out_data_accessor = result_out_acc.get_access<sycl::access::mode::write>(h); 
        // Accessor for intermediate buffer (write only)
        auto ntt_b_input_acc = ntt_b_input_buffer.get_access<sycl::access::mode::write>(h);
        // Accessor for intermediate buffer (write only)
        auto ntt_b_output_acc = ntt_b_output_buffer.get_access<sycl::access::mode::write>(h);


        // Capture necessary variables
        size_t kernel_n = n;
        size_t kernel_logn = logn;
        uint32_t kernel_mod_val = mod_value;
        uint32_t kernel_root = root;
        const uint32_t* kernel_const_ratio = const_ratio;
        
        h.single_task([=]() [[intel::kernel_args_restrict]] { 
            // Local array to store data read from pipe before processing
            uint32_t local_data[PIPE_CAPACITY];
            
            // Non-Blocking Pipe Read Loop
            size_t items_read = 0;
            while (items_read < kernel_n) {
                bool read_success = false;
                uint32_t pipe_val = ScaleReduceToNTTBPipe::read(read_success); 
                
                if (read_success) {
                    if (items_read < kernel_n) { 
                        local_data[items_read] = pipe_val;
                    }
                    items_read++;
                }
            }
            
            // Write input data to intermediate buffer for debugging
            for (size_t i = 0; i < kernel_n; ++i) {
                ntt_b_input_acc[i] = local_data[i];
            }

            size_t hsize = 1;
            size_t tt = kernel_n / 2;
            
            // Loop over stages
            for (size_t i = 0; i < kernel_logn; i++, hsize *= 2, tt /= 2) {
                for (size_t j = 0, kstart = 0; j < hsize; j++, kstart += 2 * tt) {
                    // Compute twiddle factor exponent
                    uint32_t power = hsize + j;
                    uint32_t s;
                    
                    if (power == 0) {
                        s = 1;
                    } else if (power == (1u << (kernel_logn - 1))) {
                        s = kernel_root;
                    } else {
                        // Inline exponentiation: calculate s = root^power mod mod_val
                        uint32_t current_power = kernel_root;
                        uint32_t result = 1;
                        size_t shift_count = kernel_logn - 1;
                        uint32_t temp_power_loop_var = power; // Using a temp var to modify 'power' for the loop only

                        while (true) {
                            if (temp_power_loop_var & ((uint32_t)1 << shift_count)) {
                                // Equivalent to mul_mod(current_power, result, mod)
                                uint32_t product[2];
                                
                                // Equivalent to mul_uint_wide(current_power, result, product)
                                uint64_t res_temp = (uint64_t)current_power * (uint64_t)result;
                                product[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                product[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                                
                                // Equivalent to barrett_reduce_wide(product, mod)
                                uint32_t right_hw;
                                {
                                    uint32_t res[2];
                                    uint64_t res_temp_br = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[0];
                                    res[0] = (uint32_t)(res_temp_br & 0xFFFFFFFF);
                                    res[1] = (uint32_t)((res_temp_br >> 32) & 0xFFFFFFFF);
                                    right_hw = res[1];
                                }
                                uint32_t middle_temp[2];
                                {
                                    uint64_t res_temp_br = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[1];
                                    middle_temp[0] = (uint32_t)(res_temp_br & 0xFFFFFFFF);
                                    middle_temp[1] = (uint32_t)((res_temp_br >> 32) & 0xFFFFFFFF);
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
                                    uint64_t res_temp_br = (uint64_t)product[1] * (uint64_t)kernel_const_ratio[0];
                                    middle2_temp[0] = (uint32_t)(res_temp_br & 0xFFFFFFFF);
                                    middle2_temp[1] = (uint32_t)((res_temp_br >> 32) & 0xFFFFFFFF);
                                }
                                uint32_t middle2_lw;
                                uint32_t middle2_lw_carry;
                                {
                                    middle2_lw = middle_lw + middle2_temp[0];
                                    middle2_lw_carry = (uint8_t)(middle2_lw < middle_lw);
                                }
                                uint32_t middle2_hw = middle2_temp[1] + middle2_lw_carry;
                                uint32_t tmp = product[1] * kernel_const_ratio[1] + middle_hw + middle2_hw;
                                tmp = product[0] - tmp * kernel_mod_val;
                                int32_t is_2q = (int32_t)(tmp >= kernel_mod_val);
                                uint32_t mask = (uint32_t)(-is_2q);
                                result = (uint32_t)(tmp) - (kernel_mod_val & mask);
                            }
                            
                            temp_power_loop_var &= ~((uint32_t)1 << shift_count);
                            if (temp_power_loop_var == 0) {
                                s = result;
                                break;
                            }
                            
                            uint32_t product[2]; 
                            uint64_t res_temp = (uint64_t)current_power * (uint64_t)current_power;
                            product[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                            product[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                            
                            uint32_t right_hw; 
                            {
                                uint32_t res[2];
                                uint64_t res_temp_br = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[0];
                                res[0] = (uint32_t)(res_temp_br & 0xFFFFFFFF);
                                res[1] = (uint32_t)((res_temp_br >> 32) & 0xFFFFFFFF);
                                right_hw = res[1];
                            }
                            uint32_t middle_temp[2]; 
                            {
                                uint64_t res_temp_br = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[1];
                                middle_temp[0] = (uint32_t)(res_temp_br & 0xFFFFFFFF);
                                middle_temp[1] = (uint32_t)((res_temp_br >> 32) & 0xFFFFFFFF);
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
                                uint64_t res_temp_br = (uint64_t)product[1] * (uint64_t)kernel_const_ratio[0];
                                middle2_temp[0] = (uint32_t)(res_temp_br & 0xFFFFFFFF);
                                middle2_temp[1] = (uint32_t)((res_temp_br >> 32) & 0xFFFFFFFF);
                            }
                            uint32_t middle2_lw; 
                            uint32_t middle2_lw_carry; 
                            {
                                middle2_lw = middle_lw + middle2_temp[0];
                                middle2_lw_carry = (uint8_t)(middle2_lw < middle_lw);
                            }
                            uint32_t middle2_hw = middle2_temp[1] + middle2_lw_carry; 
                            uint32_t tmp = product[1] * kernel_const_ratio[1] + middle_hw + middle2_hw;
                            tmp = product[0] - tmp * kernel_mod_val;
                            int32_t is_2q = (int32_t)(tmp >= kernel_mod_val);
                            uint32_t mask = (uint32_t)(-is_2q);
                            current_power = (uint32_t)(tmp) - (kernel_mod_val & mask);
                            
                            if (shift_count > 0) {
                                shift_count--;
                            } else if (temp_power_loop_var != 0) {
                                s = result; 
                                break; 
                            }
                        } // End of while(true) for exponentiation
                    } // End of 's' calculation
                    
                    for (size_t k = kstart; k < (kstart + tt); k++) {
                        uint32_t u = local_data[k];
                        
                        uint32_t v;
                        {
                            uint32_t product[2];
                            uint64_t res_temp = (uint64_t)local_data[k + tt] * (uint64_t)s;
                            product[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                            product[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                            
                            uint32_t right_hw;
                            {
                                uint32_t res[2];
                                uint64_t res_temp_br = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[0];
                                res[0] = (uint32_t)(res_temp_br & 0xFFFFFFFF);
                                res[1] = (uint32_t)((res_temp_br >> 32) & 0xFFFFFFFF);
                                right_hw = res[1];
                            }
                            uint32_t middle_temp[2];
                            {
                                uint64_t res_temp_br = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[1];
                                middle_temp[0] = (uint32_t)(res_temp_br & 0xFFFFFFFF);
                                middle_temp[1] = (uint32_t)((res_temp_br >> 32) & 0xFFFFFFFF);
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
                                uint64_t res_temp_br = (uint64_t)product[1] * (uint64_t)kernel_const_ratio[0];
                                middle2_temp[0] = (uint32_t)(res_temp_br & 0xFFFFFFFF);
                                middle2_temp[1] = (uint32_t)((res_temp_br >> 32) & 0xFFFFFFFF);
                            }
                            uint32_t middle2_lw;
                            uint32_t middle2_lw_carry;
                            {
                                middle2_lw = middle_lw + middle2_temp[0];
                                middle2_lw_carry = (uint8_t)(middle2_lw < middle_lw);
                            }
                            uint32_t middle2_hw = middle2_temp[1] + middle2_lw_carry;
                            uint32_t tmp = product[1] * kernel_const_ratio[1] + middle_hw + middle2_hw;
                            tmp = product[0] - tmp * kernel_mod_val;
                            int32_t is_2q = (int32_t)(tmp >= kernel_mod_val);
                            uint32_t mask = (uint32_t)(-is_2q);
                            v = (uint32_t)(tmp) - (kernel_mod_val & mask);
                        }
                        
                        uint32_t result_add = u + v;
                        if (result_add >= kernel_mod_val) result_add -= kernel_mod_val;
                        local_data[k] = result_add;
                        
                        uint32_t negated;
                        {
                            int32_t non_zero = (int32_t)(v != 0);
                            uint32_t mask = (uint32_t)(-non_zero);
                            negated = (kernel_mod_val - v) & mask;
                        }
                        
                        uint32_t result_sub = u + negated;
                        if (result_sub >= kernel_mod_val) result_sub -= kernel_mod_val;
                        local_data[k + tt] = result_sub;
                    }
                }
            }
            // --- End of NTT Computation ---

            // Write the results to the pipe
            for(size_t i = 0; i < kernel_n; ++i) 
            {
                NTTToAddModPipe::write(local_data[i]);
            }

            // Write the results to the output buffer
            for(size_t i = 0; i < kernel_n; ++i)
            {
                out_data_accessor[i] = local_data[i];
            }

            // Write output data to intermediate buffer for debugging
            for (size_t i = 0; i < kernel_n; ++i) {
                ntt_b_output_acc[i] = local_data[i];
            }
            
        }); // End of single_task
    } // End of operator()
}; // End of NTTKernel_B class