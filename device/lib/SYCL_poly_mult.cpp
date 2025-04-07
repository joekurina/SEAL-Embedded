#include "SYCL_poly_mult.h"
#include <iostream>

class PolyMultNTTKernel {
private:
    size_t n;
    uint32_t mod_value;
    const uint32_t* const_ratio;
    mutable buffer<uint32_t, 1> a_acc;
    mutable buffer<uint32_t, 1> b_acc;

public:
    PolyMultNTTKernel(size_t n_val, uint32_t mod_val, const uint32_t* const_ratio_val,
                        buffer<uint32_t, 1>& a_buf,
                        buffer<uint32_t, 1>& b_buf)
        : n(n_val), mod_value(mod_val), const_ratio(const_ratio_val), 
            a_acc(a_buf), b_acc(b_buf) {}
    
    void operator()(handler& h) const {
        // Get access to buffers
        auto a = a_acc.get_access<access::mode::read_write>(h);
        auto b = b_acc.get_access<access::mode::read>(h);
        
        // Capture necessary variables
        size_t kernel_n = n;
        uint32_t kernel_mod_val = mod_value;
        const uint32_t* kernel_const_ratio = const_ratio;
        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Component-wise multiplication
            for (size_t i = 0; i < kernel_n; i++) {
                // Get values
                uint32_t a_val = a[i];
                uint32_t b_val = b[i];
                
                // Multiply: a[i] = (a[i] * b[i]) mod q
                
                // 1. Multiply to get wide result
                uint32_t product[2];
                uint64_t res_temp = (uint64_t)a_val * (uint64_t)b_val;
                product[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                product[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                
                // 2. Barrett reduction starts here
                
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
                
                // Final reduction if needed
                int32_t is_2q = (int32_t)(tmp >= kernel_mod_val);
                uint32_t mask = (uint32_t)(-is_2q);
                a[i] = (uint32_t)(tmp) - (kernel_mod_val & mask);
            }
        });
    }
};

// Function to perform polynomial multiplication in NTT form
void ntt_form_poly_mod_mult(
    queue q, 
    uint32_t *a, 
    const uint32_t *b, 
    size_t n, 
    uint32_t mod_value, 
    const uint32_t* const_ratio
) {
    // Create SYCL buffers
    buffer<uint32_t, 1> a_buf(a, range<1>(n));
    buffer<uint32_t, 1> b_buf(const_cast<uint32_t*>(b), range<1>(n));
    
    try {
        // Print device info
        std::cout << "Running NTT Polynomial Multiplication on device: "
                  << q.get_device().get_info<info::device::name>().c_str()
                  << std::endl;
        
        // Submit and execute the kernel
        q.submit(PolyMultNTTKernel(n, mod_value, const_ratio, a_buf, b_buf)).wait();
        
    } catch (exception const &e) {
        std::cerr << "Caught a synchronous SYCL exception in poly_mult_mod_ntt_form_inpl: "
                  << e.what() << "\n";
        std::exit(1);
    }
}