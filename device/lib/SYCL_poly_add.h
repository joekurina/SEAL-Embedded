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
    mutable sycl::buffer<uint32_t, 1> p1_acc;
    //mutable sycl::buffer<uint32_t, 1> p2_acc;

public:
    PolyAddModKernel(size_t n_val, uint32_t mod_val,
                        sycl::buffer<uint32_t, 1>& p1_buf)
                        //sycl::buffer<uint32_t, 1>& p2_buf)
        : n(n_val), mod_value(mod_val), p1_acc(p1_buf) {}
        //, p2_acc(p2_buf) {}
    
    void operator()(sycl::handler& h) const {
        // Get access to buffers
        auto p1 = p1_acc.get_access<sycl::access::mode::write>(h);
        //auto p2 = p2_acc.get_access<sycl::access::mode::read>(h);
        
        // Capture necessary variables
        size_t kernel_n = n;
        uint32_t kernel_mod_val = mod_value;
        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            
            // Process each coefficient
            for (size_t i = 0; i < kernel_n; i++) {
                // Get coefficients
                //uint32_t coeff1 = p1[i];
                uint32_t coeff1 = PolyMultNegToPolyAddModPipe::read(); // Read from the pipe
                //uint32_t coeff2 = p2[i];
                uint32_t coeff2 = NTTToAddModPipe::read(); // Read from the pipe
                
                // Add coefficients
                uint32_t sum = coeff1 + coeff2;
                
                // Reduce modulo q: 
                // If sum >= q, subtract q
                int32_t is_ge_q = (int32_t)(sum >= kernel_mod_val);
                uint32_t mask = (uint32_t)(-is_ge_q);
                uint32_t result = sum - (kernel_mod_val & mask);
                
                // Store the result
                p1[i] = result;
            }
        });
    }
};