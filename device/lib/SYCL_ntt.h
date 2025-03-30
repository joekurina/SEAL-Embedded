#pragma once

#include "SYCL_ckks_sym.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdio>
#include <cstdlib>

// NTT Kernel functor class
class NTTKernel {
private:
    size_t n;
    size_t logn;
    uint32_t mod_value;
    const uint32_t* const_ratio;
    mutable sycl::buffer<uint32_t, 1> vec_acc;

    // Internal function to get the NTT root
    uint32_t get_root() const {
        uint32_t root;
        switch (n) {
            case 4096:
                switch (mod_value) {
                    case 134012929: root = 7470; break;
                    case 134111233: root = 3856; break;
                    case 134176769: root = 24149; break;
                    case 1053818881: root = 503422; break;
                    case 1054015489: root = 16768; break;
                    case 1054212097: root = 7305; break;
                    default: {
                        printf("Error! Need first power of root for ntt, n = 4K\n");
                        printf("Modulus value = %d", mod_value);
                        exit(1);
                    }
                }
                break;
            case 8192:
                switch (mod_value) {
                    case 1053818881: root = 374229; break;
                    case 1054015489: root = 123363; break;
                    case 1054212097: root = 79941; break;
                    case 1055260673: root = 38869; break;
                    case 1056178177: root = 162146; break;
                    case 1056440321: root = 81884; break;
                    default: {
                        printf("Error! Need first power of root for ntt, n = 8K\n");
                        printf("Modulus value = %d", mod_value);
                        exit(1);
                    }
                }
                break;
            case 16384:
                switch (mod_value) {
                    case 1053818881: root = 13040; break;
                    case 1054015489: root = 507; break;
                    case 1054212097: root = 1595; break;
                    case 1055260673: root = 68507; break;
                    case 1056178177: root = 3073; break;
                    case 1056440321: root = 6854; break;
                    case 1058209793: root = 44467; break;
                    case 1060175873: root = 16117; break;
                    case 1060700161: root = 27607; break;
                    case 1060765697: root = 222391; break;
                    case 1061093377: root = 105471; break;
                    case 1062469633: root = 310222; break;
                    case 1062535169: root = 2005; break;
                    default: {
                        printf("Error! Need first power of root for ntt, n = 16K\n");
                        printf("Modulus value = %d", mod_value);
                        exit(1);
                    }
                }
                break;
            default: {
                printf("Error! Need first power of root for ntt\n");
                printf("Modulus value = %d", mod_value);
                exit(1);
            }
        }
        return root;
    }

public:
    NTTKernel(size_t n_val, size_t logn_val, uint32_t mod_val, 
                const uint32_t* const_ratio_val, sycl::buffer<uint32_t, 1>& vec_buf)
        : n(n_val), logn(logn_val), mod_value(mod_val), const_ratio(const_ratio_val), vec_acc(vec_buf) {}
    
    void operator()(sycl::handler& h) const {
        // Get access to the buffer
        auto data = vec_acc.get_access<sycl::access::mode::read_write>(h);
        
        // Capture necessary variables
        size_t kernel_n = n;
        size_t kernel_logn = logn;
        uint32_t kernel_mod_val = mod_value;
        const uint32_t* kernel_const_ratio = const_ratio;
        
        // Get the root directly within the kernel
        uint32_t kernel_root = get_root();
        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
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
                                
                                // Equivalent to mul_uint_wide(current_power, result, product)
                                uint64_t res_temp = (uint64_t)current_power * (uint64_t)result;
                                product[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                product[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                                
                                // Equivalent to barrett_reduce_wide(product, mod)
                                // Which calls barrett_reduce_64input_32modulus(product, mod)
                                
                                // Round 1
                                uint32_t right_hw;
                                {
                                    uint32_t res[2];
                                    uint64_t res_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[0];
                                    res[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                    res[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                                    right_hw = res[1];
                                }

                                uint32_t middle_temp[2];
                                {
                                    uint64_t res_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[1];
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

                                // Round 2
                                uint32_t middle2_temp[2];
                                {
                                    uint64_t res_temp = (uint64_t)product[1] * (uint64_t)kernel_const_ratio[0];
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

                                uint32_t tmp = product[1] * kernel_const_ratio[1] + middle_hw + middle2_hw;

                                // Barrett subtraction
                                tmp = product[0] - tmp * kernel_mod_val;
                                
                                // Equivalent to shift_result
                                int32_t is_2q = (int32_t)(tmp >= kernel_mod_val);
                                uint32_t mask = (uint32_t)(-is_2q);
                                result = (uint32_t)(tmp) - (kernel_mod_val & mask);
                            }
                            
                            power &= ~((uint32_t)1 << shift_count);
                            if (power == 0) {
                                s = result;
                                break;
                            }
                            
                            // Equivalent to mul_mod(current_power, current_power, mod)
                            uint32_t product[2];
                            
                            // Equivalent to mul_uint_wide(current_power, current_power, product)
                            uint64_t res_temp = (uint64_t)current_power * (uint64_t)current_power;
                            product[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                            product[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                            
                            // Equivalent to barrett_reduce_wide(product, mod)
                            // Which calls barrett_reduce_64input_32modulus(product, mod)
                            
                            // Round 1
                            uint32_t right_hw;
                            {
                                uint32_t res[2];
                                uint64_t res_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[0];
                                res[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                res[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                                right_hw = res[1];
                            }

                            uint32_t middle_temp[2];
                            {
                                uint64_t res_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[1];
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

                            // Round 2
                            uint32_t middle2_temp[2];
                            {
                                uint64_t res_temp = (uint64_t)product[1] * (uint64_t)kernel_const_ratio[0];
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

                            uint32_t tmp = product[1] * kernel_const_ratio[1] + middle_hw + middle2_hw;

                            // Barrett subtraction
                            tmp = product[0] - tmp * kernel_mod_val;
                            
                            // Equivalent to shift_result
                            int32_t is_2q = (int32_t)(tmp >= kernel_mod_val);
                            uint32_t mask = (uint32_t)(-is_2q);
                            current_power = (uint32_t)(tmp) - (kernel_mod_val & mask);
                            
                            shift_count--;
                        }
                    }
                    
                    // Process each pair in the current group
                    for (size_t k = kstart; k < (kstart + tt); k++) {
                        uint32_t u = data[k];
                        
                        // Equivalent to mul_mod(data[k + tt], s, mod)
                        uint32_t v;
                        {
                            uint32_t product[2];
                            
                            // Equivalent to mul_uint_wide(data[k + tt], s, product)
                            uint64_t res_temp = (uint64_t)data[k + tt] * (uint64_t)s;
                            product[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                            product[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                            
                            // Equivalent to barrett_reduce_wide(product, mod)
                            // Which calls barrett_reduce_64input_32modulus(product, mod)
                            
                            // Round 1
                            uint32_t right_hw;
                            {
                                uint32_t res[2];
                                uint64_t res_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[0];
                                res[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                                res[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                                right_hw = res[1];
                            }

                            uint32_t middle_temp[2];
                            {
                                uint64_t res_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[1];
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

                            // Round 2
                            uint32_t middle2_temp[2];
                            {
                                uint64_t res_temp = (uint64_t)product[1] * (uint64_t)kernel_const_ratio[0];
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

                            uint32_t tmp = product[1] * kernel_const_ratio[1] + middle_hw + middle2_hw;

                            // Barrett subtraction
                            tmp = product[0] - tmp * kernel_mod_val;
                            
                            // Equivalent to shift_result
                            int32_t is_2q = (int32_t)(tmp >= kernel_mod_val);
                            uint32_t mask = (uint32_t)(-is_2q);
                            v = (uint32_t)(tmp) - (kernel_mod_val & mask);
                        }
                        
                        // Equivalent to add_mod(u, v, mod)
                        uint32_t result_add = u + v;
                        if (result_add >= kernel_mod_val) result_add -= kernel_mod_val;
                        data[k] = result_add;
                        
                        // Equivalent to sub_mod(u, v, mod)
                        // First, equivalent to neg_mod(v, mod)
                        uint32_t negated;
                        {
                            int32_t non_zero = (int32_t)(v != 0);
                            uint32_t mask = (uint32_t)(-non_zero);
                            negated = (kernel_mod_val - v) & mask;
                        }
                        
                        // Then, equivalent to add_mod(u, negated, mod)
                        uint32_t result_sub = u + negated;
                        if (result_sub >= kernel_mod_val) result_sub -= kernel_mod_val;
                        data[k + tt] = result_sub;
                    }
                }
            }
        });
    }
};