#pragma once

#include "SYCL_ckks_sym.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "SYCL_pipes.h" // Include pipes definition
#include <cstdio>
#include <cstdlib>

// NTT Kernel functor class
class NTTKernel_1 {
private:
    size_t n;
    size_t logn;
    uint32_t mod_value;
    const uint32_t* const_ratio;
    mutable sycl::buffer<uint32_t, 1> result_out_acc; // MODIFIED: For output

public:
    NTTKernel_1(size_t n_val, size_t logn_val, uint32_t mod_val, 
                const uint32_t* const_ratio_val, 
                sycl::buffer<uint32_t, 1>& result_output_buf) // MODIFIED: Constructor takes output buffer
        : n(n_val), logn(logn_val), mod_value(mod_val), 
            const_ratio(const_ratio_val), result_out_acc(result_output_buf) {} // MODIFIED
    
    void operator()(sycl::handler& h) const {
        // Get write access to the output buffer
        auto out_data_accessor = result_out_acc.get_access<sycl::access::mode::write>(h); // ADDED
        
        // Capture necessary variables
        size_t kernel_n = n;
        size_t kernel_logn = logn;
        uint32_t kernel_mod_val = mod_value;
        const uint32_t* kernel_const_ratio = const_ratio;

        sycl::ext::oneapi::experimental::printf("NTTKernel_1: Starting kernel...\n"); // Original commentary preserved
        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            sycl::ext::oneapi::experimental::printf("NTTKernel_1: Kernel Started...\n"); // Original commentary preserved
            
            // Local array to store data read from pipe before processing
            uint32_t local_data[PIPE_CAPACITY]; // MODIFIED: Was 4096, now PIPE_CAPACITY

            sycl::ext::oneapi::experimental::printf("NTTKernel_1: Input Pipe Read Loop...\n"); // Original commentary preserved
            // Read data from the input pipe
            // non-blocking while-loop
            size_t items_read = 0;
            while (items_read < kernel_n) {
                //sycl::ext::oneapi::experimental::printf("NTTKernel_1: Reading Input Pipe...\n");
                bool read_success = false; // Added for non-blocking read
                uint32_t pipe_val = ScaleReduceToNTT1Pipe::read(read_success); // Added for non-blocking read
                if (read_success) { // Added for non-blocking read
                    if (items_read < PIPE_CAPACITY) { // Check against local_data actual size
                        local_data[items_read] = pipe_val;
                    }
                    // Debug message for first and last iterations
                    if (items_read == 0 || items_read == kernel_n - 1) {
                        sycl::ext::oneapi::experimental::printf("NTTKernel_1: Read value %u from input pipe at index %zu\n", local_data[items_read], items_read);
                    }
                    items_read++;
                }
            }
            
            // Calculate the NTT root directly in the device kernel
            uint32_t kernel_root;
            
            // Root selection based on polynomial degree and modulus
            // Case for n = 4096
            if (kernel_n == 4096) {
                if (kernel_mod_val == 134012929) kernel_root = 7470;
                else if (kernel_mod_val == 134111233) kernel_root = 3856;
                else if (kernel_mod_val == 134176769) kernel_root = 24149;
                else if (kernel_mod_val == 1053818881) kernel_root = 503422;
                else if (kernel_mod_val == 1054015489) kernel_root = 16768;
                else if (kernel_mod_val == 1054212097) kernel_root = 7305;
                else kernel_root = 1; // Default fallback, invalid but prevents crashing
            }
            /*
            // Case for n = 8192
            else if (kernel_n == 8192) {
                if (kernel_mod_val == 1053818881) kernel_root = 374229;
                else if (kernel_mod_val == 1054015489) kernel_root = 123363;
                else if (kernel_mod_val == 1054212097) kernel_root = 79941;
                else if (kernel_mod_val == 1055260673) kernel_root = 38869;
                else if (kernel_mod_val == 1056178177) kernel_root = 162146;
                else if (kernel_mod_val == 1056440321) kernel_root = 81884;
                else kernel_root = 1; // Default fallback
            }
            // Case for n = 16384
            else if (kernel_n == 16384) {
                if (kernel_mod_val == 1053818881) kernel_root = 13040;
                else if (kernel_mod_val == 1054015489) kernel_root = 507;
                else if (kernel_mod_val == 1054212097) kernel_root = 1595;
                else if (kernel_mod_val == 1055260673) kernel_root = 68507;
                else if (kernel_mod_val == 1056178177) kernel_root = 3073;
                else if (kernel_mod_val == 1056440321) kernel_root = 6854;
                else if (kernel_mod_val == 1058209793) kernel_root = 44467;
                else if (kernel_mod_val == 1060175873) kernel_root = 16117;
                else if (kernel_mod_val == 1060700161) kernel_root = 27607;
                else if (kernel_mod_val == 1060765697) kernel_root = 222391;
                else if (kernel_mod_val == 1061093377) kernel_root = 105471;
                else if (kernel_mod_val == 1062469633) kernel_root = 310222;
                else if (kernel_mod_val == 1062535169) kernel_root = 2005;
                else kernel_root = 1; // Default fallback
            }
            else {
                kernel_root = 1; // Default fallback
            }
            */
            else {
                kernel_root = 1; // Default fallback
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

            // Write the final NTT result from local_data to the output buffer
            sycl::ext::oneapi::experimental::printf("NTTKernel_1: Finished processing and wrote to output buffer.\n");
            for(size_t i_out = 0; i_out < kernel_n; ++i_out) { // Original loop variable 'i' for output
                if (i_out < PIPE_CAPACITY) { 
                    out_data_accessor[i_out] = local_data[i_out];
                }
                // Original debug message
                if (i_out == 0 || i_out == kernel_n - 1) {
                   sycl::ext::oneapi::experimental::printf("NTTKernel_1: Wrote value %u to output buffer from index %zu\n", local_data[i_out], i_out);
                }
            }
            sycl::ext::oneapi::experimental::printf("NTTKernel_1: Finished processing and wrote to output buffer.\n");
        });
    }
};

// --- Modified NTTKernel_2 (Writes to two buffers) ---
class NTTKernel_2 {
private:
    size_t n;
    size_t logn;
    uint32_t mod_value;
    const uint32_t* const_ratio;
    mutable sycl::buffer<uint32_t, 1> vec_acc;    // Primary In/Out buffer
    mutable sycl::buffer<uint32_t, 1> save_acc;   // Save destination buffer


public:
    // Constructor accepting both primary and save buffers
    NTTKernel_2(size_t n_val, size_t logn_val, uint32_t mod_val,
                const uint32_t* const_ratio_val,
                sycl::buffer<uint32_t, 1>& vec_buf,  // In/Out (Primary)
                sycl::buffer<uint32_t, 1>& save_buf) // Out only (Save)
        : n(n_val), logn(logn_val), mod_value(mod_val),
            const_ratio(const_ratio_val),
            vec_acc(vec_buf),  // Initialize primary buffer member
            save_acc(save_buf) // Initialize save buffer member
            {}

    void operator()(sycl::handler& h) const {
        // Accessor for primary buffer (read/write)
        auto data = vec_acc.get_access<sycl::access::mode::read_write>(h);
        // Accessor for save buffer (write only)
        auto s_save = save_acc.get_access<sycl::access::mode::write>(h);

        // Capture necessary variables for the kernel lambda
        size_t kernel_n = n;
        size_t kernel_logn = logn;
        uint32_t kernel_mod_val = mod_value;
        const uint32_t* kernel_const_ratio = const_ratio;

        // Perform host-side check if save buffer is valid before launching kernel
        bool save_output = (save_acc.get_range() == sycl::range(kernel_n));

        sycl::ext::oneapi::experimental::printf("NTTKernel_2: Starting kernel (Dual output)...\n");

        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Calculate the NTT root directly in the device kernel
            uint32_t kernel_root;

            // Root selection based on polynomial degree and modulus
            // Case for n = 4096
            if (kernel_n == 4096) {
                if (kernel_mod_val == 134012929) kernel_root = 7470;
                else if (kernel_mod_val == 134111233) kernel_root = 3856;
                else if (kernel_mod_val == 134176769) kernel_root = 24149;
                else if (kernel_mod_val == 1053818881) kernel_root = 503422;
                else if (kernel_mod_val == 1054015489) kernel_root = 16768;
                else if (kernel_mod_val == 1054212097) kernel_root = 7305;
                else kernel_root = 1; // Default fallback, invalid but prevents crashing
            }
            /*
            // Case for n = 8192
            else if (kernel_n == 8192) {
                if (kernel_mod_val == 1053818881) kernel_root = 374229;
                else if (kernel_mod_val == 1054015489) kernel_root = 123363;
                else if (kernel_mod_val == 1054212097) kernel_root = 79941;
                else if (kernel_mod_val == 1055260673) kernel_root = 38869;
                else if (kernel_mod_val == 1056178177) kernel_root = 162146;
                else if (kernel_mod_val == 1056440321) kernel_root = 81884;
                else kernel_root = 1; // Default fallback
            }
            // Case for n = 16384
            else if (kernel_n == 16384) {
                if (kernel_mod_val == 1053818881) kernel_root = 13040;
                else if (kernel_mod_val == 1054015489) kernel_root = 507;
                else if (kernel_mod_val == 1054212097) kernel_root = 1595;
                else if (kernel_mod_val == 1055260673) kernel_root = 68507;
                else if (kernel_mod_val == 1056178177) kernel_root = 3073;
                else if (kernel_mod_val == 1056440321) kernel_root = 6854;
                else if (kernel_mod_val == 1058209793) kernel_root = 44467;
                else if (kernel_mod_val == 1060175873) kernel_root = 16117;
                else if (kernel_mod_val == 1060700161) kernel_root = 27607;
                else if (kernel_mod_val == 1060765697) kernel_root = 222391;
                else if (kernel_mod_val == 1061093377) kernel_root = 105471;
                else if (kernel_mod_val == 1062469633) kernel_root = 310222;
                else if (kernel_mod_val == 1062535169) kernel_root = 2005;
                else kernel_root = 1; // Default fallback
            }
            else {
                kernel_root = 1; // Default fallback
            }
            */
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

                        // Write results to BOTH buffers ***
                        data[k] = result_add;       // Write result to primary buffer
                        data[k + tt] = result_sub;  // Write result to primary buffer

                        // Conditionally write to save buffer if requested
                        if (save_output) {
                            s_save[k] = result_add;      // Write result to save buffer
                            s_save[k + tt] = result_sub; // Write result to save buffer
                        }
                    } // End loop k
                } // End loop j
            } // End loop i (stages)

            sycl::ext::oneapi::experimental::printf("NTTKernel_2: Finished NTT computation (Dual output).\n");
        }); // End single_task lambda
    } // End operator()
   }; // End of NTTKernel_2 class