#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_pipes.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdint>

// Merged Kernel: Performs Scaling/Conversion and Reduction
class ScaleAndReduceKernel {
private:
    size_t n;
    double scale;
    uint32_t mod_value;
    const uint32_t* const_ratio;
    // mutable sycl::buffer<uint32_t, 1> out_acc; // REMOVED: Output buffer

public:
    // Constructor takes combined arguments
    ScaleAndReduceKernel(size_t n_val, double scale_val, uint32_t mod_val,
                         const uint32_t* const_ratio_val /*,
                         sycl::buffer<uint32_t, 1>& out_buf REMOVED */ )
        : n(n_val),
          scale(scale_val),
          mod_value(mod_val),
          const_ratio(const_ratio_val) /*,
          out_acc(out_buf) REMOVED */ {}

    void operator()(sycl::handler& h) const {
        // Get access to the output buffer
        // auto out = out_acc.get_access<sycl::access::mode::write>(h); // REMOVED

        // Capture necessary variables for the kernel lambda
        size_t kernel_n = n; 
        double kernel_scale = scale;
        uint32_t kernel_mod_val = mod_value;
        const uint32_t* kernel_const_ratio = const_ratio;

        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // --- Local arrays to buffer pipe data ---
            std::complex<double> local_encoded_data[PIPE_CAPACITY];
            int8_t local_error_data[PIPE_CAPACITY];

            // --- Pre-filling Phase: Using two sequential inner while loops for non-blocking reads ---
            size_t items_read_and_stored = 0;
            while (items_read_and_stored < kernel_n) { // Loop up to the runtime kernel_n
                std::complex<double> current_encoded_value;
                bool encoded_value_acquired = false;
                
                // Loop 1: non-blocking read for the encoded value
                while (!encoded_value_acquired) {
                    current_encoded_value = IFFTToScaleAndReducePipe::read(encoded_value_acquired);
                }

                int8_t current_error_value;
                bool error_value_acquired = false;

                // Loop 2: non-blocking read for the error value
                while (!error_value_acquired) {
                    current_error_value = IFFTErrorToScaleAndReducePipe::read(error_value_acquired);
                }

                // At this point, both current_encoded_value and current_error_value have been successfully read
                if (items_read_and_stored < PIPE_CAPACITY) { 
                    local_encoded_data[items_read_and_stored] = current_encoded_value;
                    local_error_data[items_read_and_stored] = current_error_value;
                }
                // Debug message for first and last iterations
                if (items_read_and_stored == 0 || items_read_and_stored == kernel_n - 1) {
                    sycl::ext::oneapi::experimental::printf(
                        "ScaleAndReduceKernel: Read encoded value %f and error %d at index %zu\n",
                        current_encoded_value.real(), current_error_value, items_read_and_stored);
                }
                items_read_and_stored++;
            } // End of while (items_read_and_stored < kernel_n)

            // --- Processing Phase ---
            double n_inv = kernel_scale / static_cast<double>(kernel_n);

            for (size_t i = 0; i < kernel_n; i++) {
                std::complex<double> encoded_value = local_encoded_data[i]; 
                int8_t error_value = local_error_data[i];

                double real_val = encoded_value.real();
                double scaled = sycl::round(real_val * n_inv);
                int64_t int_val = static_cast<int64_t>(scaled);
                int64_t intermediate_result = int_val + error_value;

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

                // out[i] = final_result; // Writing to buffer
                ScaleReduceToNTT1Pipe::write(final_result); 
                
                if (i == 0 || i == kernel_n -1) {
                     sycl::ext::oneapi::experimental::printf(
                         "ScaleAndReduceKernel: Loop i=%zu, input encoded_real=%f, error=%d, wrote result %u to ScaleReduceToNTT1Pipe.\n",
                         i, encoded_value.real(), error_value, final_result);
                }
            } // End of for loop

            sycl::ext::oneapi::experimental::printf(
                "ScaleAndReduceKernel: Loop finished after %zu iterations.\n", kernel_n);
            sycl::ext::oneapi::experimental::printf(
                "ScaleAndReduceKernel: Finished.\n");
        }); // End single_task
    } // End operator()
}; // End of ScaleAndReduceKernel class