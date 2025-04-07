#include "SYCL_poly_neg.h"
#include <iostream>

class PolyNegModKernel {
private:
    size_t n;
    uint32_t mod_value;
    mutable buffer<uint32_t, 1> p_acc;

public:
    PolyNegModKernel(size_t n_val, uint32_t mod_val, buffer<uint32_t, 1>& p_buf)
        : n(n_val), mod_value(mod_val), p_acc(p_buf) {}
    
    void operator()(handler& h) const {
        // Get access to buffer
        auto p = p_acc.get_access<access::mode::read_write>(h);
        
        // Capture necessary variables
        size_t kernel_n = n;
        uint32_t kernel_mod_val = mod_value;
        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Apply negation to each coefficient
            for (size_t i = 0; i < kernel_n; i++) {
                // Get the coefficient
                uint32_t coeff = p[i];
                
                // Compute if coefficient is non-zero
                int32_t non_zero = (int32_t)(coeff != 0);
                uint32_t mask = (uint32_t)(-non_zero);
                
                // Compute negation: if coeff == 0, result = 0; else result = q - coeff
                uint32_t result = (kernel_mod_val - coeff) & mask;
                
                // Store the result
                p[i] = result;
            }
        });
    }
};

// Function to negate polynomial coefficients modulo q
void poly_negate_mod(queue q, uint32_t *p, size_t n, uint32_t mod_value) {
    // Create SYCL buffer for the polynomial
    buffer<uint32_t, 1> p_buf(p, range<1>(n));

    try {
        // Print device info
        std::cout << "Running Polynomial Negation on device: "
                  << q.get_device().get_info<info::device::name>().c_str()
                  << std::endl;
        
        // Submit and execute the kernel
        q.submit(PolyNegModKernel(n, mod_value, p_buf)).wait();
        
    } catch (exception const &e) {
        std::cerr << "Caught a synchronous SYCL exception in poly_neg_mod: "
                  << e.what() << "\n";
        std::exit(1);
    }
}