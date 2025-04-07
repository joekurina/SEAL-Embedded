#include "SYCL_poly_add.h"
#include <iostream>

// Kernel for modular addition of two polynomials
class PolyAddModKernel {
private:
    size_t n;
    uint32_t mod_value;
    mutable buffer<uint32_t, 1> p1_acc;
    mutable buffer<uint32_t, 1> p2_acc;

public:
    PolyAddModKernel(size_t n_val, uint32_t mod_val,
                        buffer<uint32_t, 1>& p1_buf,
                        buffer<uint32_t, 1>& p2_buf)
        : n(n_val), mod_value(mod_val), p1_acc(p1_buf), p2_acc(p2_buf) {}
    
    void operator()(handler& h) const {
        // Get access to buffers
        auto p1 = p1_acc.get_access<access::mode::read_write>(h);
        auto p2 = p2_acc.get_access<access::mode::read>(h);
        
        // Capture necessary variables
        size_t kernel_n = n;
        uint32_t kernel_mod_val = mod_value;
        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Process each coefficient
            for (size_t i = 0; i < kernel_n; i++) {
                // Get coefficients
                uint32_t coeff1 = p1[i];
                uint32_t coeff2 = p2[i];
                
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

// Function to add two polynomials modulo q
void poly_add_mod(queue q, uint32_t *p1, const uint32_t *p2, size_t n, uint32_t mod_value) {
    // Create SYCL buffers
    buffer<uint32_t, 1> p1_buf(p1, range<1>(n));
    buffer<uint32_t, 1> p2_buf(const_cast<uint32_t*>(p2), range<1>(n));

    try {      
        // Print device info
        std::cout << "Running Polynomial Addition on device: "
                  << q.get_device().get_info<info::device::name>().c_str()
                  << std::endl;
        
        // Submit and execute the kernel
        q.submit(PolyAddModKernel(n, mod_value, p1_buf, p2_buf)).wait();
        
    } catch (exception const &e) {
        std::cerr << "Caught a synchronous SYCL exception in poly_add_mod_inpl: "
                  << e.what() << "\n";
        std::exit(1);
    }
}