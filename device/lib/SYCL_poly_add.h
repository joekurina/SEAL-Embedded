#pragma once

#include "SYCL_ckks_sym.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "SYCL_pipes.h"

// Kernel for modular addition of two polynomials
class PolyAddModKernel {
private:
    size_t n;
    uint32_t mod_value;
    mutable sycl::buffer<u32x4_input, 1> output_acc;

public:
    PolyAddModKernel(size_t n_val, uint32_t mod_val,
                        sycl::buffer<u32x4_input, 1>& output_buf)
        : n(n_val), mod_value(mod_val), output_acc(output_buf) {}
    
    void operator()(sycl::handler& h) const {
        // Get access to the output buffer
        auto output = output_acc.get_access<sycl::access::mode::write>(h);
        
        // Capture necessary variables
        size_t kernel_n = n;
        uint32_t kernel_mod_val = mod_value;
        
        h.single_task<class PolyAddModKernel>([=]() [[intel::kernel_args_restrict]] {
            
            // Process coefficients in packed 4-lane blocks
            for (size_t blk = 0; blk < kernel_n / 4; ++blk) {
                u32x4_input a_block = PolyMultNegToPolyAddModPipe::read();
                u32x4_input b_block = NTTToAddModPipe::read();

                u32x4_input out_block{};

                for (size_t lane = 0; lane < 4; ++lane) {
                    uint32_t coeff1 = reinterpret_cast<const uint32_t*>(&a_block)[lane];
                    uint32_t coeff2 = reinterpret_cast<const uint32_t*>(&b_block)[lane];

                    uint32_t sum = coeff1 + coeff2;
                    int32_t is_ge_q = (int32_t)(sum >= kernel_mod_val);
                    uint32_t mask = (uint32_t)(-is_ge_q);
                    uint32_t result = sum - (kernel_mod_val & mask);

                    reinterpret_cast<uint32_t*>(&out_block)[lane] = result;
                }

                output[blk] = out_block;
            }
        });
    }
};