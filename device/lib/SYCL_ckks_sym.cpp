#include "SYCL_ckks_sym.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <memory>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Include the pipe definitions
#include "SYCL_pipes.h"

// Include the SYCL kernel headers
#include "SYCL_entrance.h"
#include "SYCL_exit.h"
#include "SYCL_ifft.h"
#include "SYCL_scale_and_convert.h"
#include "SYCL_ntt.h"
#include "SYCL_poly_mult.h"
#include "SYCL_poly_add.h"
#include "SYCL_poly_neg.h"
#include "SYCL_reduce_pte.h"

// Function prototypes for internal functions used in SYCL_combined_encrypt
void pipeline(
                sycl::queue q, 
                sycl::device device, 
                size_t n, 
                size_t logn, 
                double scale, 
                sycl::buffer<complex_double, 1>& encoding_buf, 
                sycl::buffer<int8_t, 1>& error_samples_buf,
                sycl::buffer<int64_t, 1>& pt_with_error_buf );
void ntt_1(size_t n, size_t logn, uint32_t mod_value, const uint32_t* const_ratio, uint32_t *vec);
void ntt_2(size_t n, size_t logn, uint32_t mod_value, const uint32_t* const_ratio, uint32_t *vec);
void ntt_form_poly_mod_mult(uint32_t *a, const uint32_t *b, size_t n, uint32_t mod_value, const uint32_t* const_ratio);
void poly_negate_mod(uint32_t *p, size_t n, uint32_t mod_value);
void reduce_pte(const int64_t *conj_vals_int, size_t n, uint32_t mod_value, const uint32_t* const_ratio, uint32_t *out);
void poly_add_mod(uint32_t *p1, const uint32_t *p2, size_t n, uint32_t mod_value);

// Implementation of the C-compatible function from SYCL_ckks_sym.h
extern "C" void SYCL_combined_encrypt(
    /* parms related values */
    size_t n,                           // Polynomial degree
    size_t logn,                        // Log of polynomial degree
    double scale,                       // Scale value
    uint32_t mod_value,                 // Modulus value (q)
    const uint32_t* const_ratio,        // Const ratio for Barrett reduction
    complex_double* encoding_buffer,    // Buffer for encoding
    uint32_t* expanded_s,               // Expanded secret key
    uint32_t* uniform_poly,             // Uniform polynomial (c1)
    int8_t* error_samples,              // Error samples
    int64_t* pt_with_error,             // Plaintext + error
    uint32_t* ntt_pte,                  // Scratch space for NTT
    uint32_t* c0_s,                     // Output: 1st ciphertext component
    uint32_t* c1,                       // Output: 2nd ciphertext component
    uint32_t* s_save,                   // Optional: Save expanded s (for testing)
    uint32_t* c1_save                   // Optional: Save c1 (for testing)
) {
    // Create SYCL buffers from the input pointers.
    sycl::buffer<complex_double, 1> encoding_buf(encoding_buffer, sycl::range<1>(n));
    sycl::buffer<int64_t, 1> pt_with_error_buf(pt_with_error, sycl::range<1>(n));
    sycl::buffer<int8_t, 1> error_samples_buf(error_samples, sycl::range<1>(n));
    sycl::buffer<uint32_t, 1> expanded_s_buf(expanded_s, sycl::range<1>(n));
    sycl::buffer<uint32_t, 1> uniform_poly_buf(uniform_poly, sycl::range<1>(n));

    // Get host-access pointers for the expanded secret key and uniform poly.
    auto expanded_s_ptr = expanded_s_buf.get_host_access().get_pointer();
    auto uniform_poly_ptr = uniform_poly_buf.get_host_access().get_pointer();
    
    // ==============================================================
    //   Generate ciphertext components
    // ==============================================================
    
    // Create a SYCL queue using the FPGA emulator selector.
    sycl::queue q{sycl::ext::intel::fpga_emulator_selector_v};
    auto device = q.get_device();
    
    // Use the pipelined implementation for IFFT and ScaleAndConvert
    pipeline(q, device, n, logn, scale, encoding_buf, error_samples_buf, pt_with_error_buf);

    // 1. Copy uniform polynomial to c1 output.
    std::memcpy(c1, uniform_poly_ptr, n * sizeof(uint32_t));
    
    // 2. Save c1 if requested for testing.
    if (c1_save != nullptr) {
        std::memcpy(c1_save, uniform_poly_ptr, n * sizeof(uint32_t));
    }
    
    // 3. Copy expanded secret key to c0_s.
    std::memcpy(c0_s, expanded_s_ptr, n * sizeof(uint32_t));
    
    // 4. Apply NTT to the secret key.
    // Updated call passing explicit parameters.
    ntt_1(n, logn, mod_value, const_ratio, c0_s);
    
    // 5. Save NTT(s) for later decryption if requested.
    if (s_save != nullptr) {
        std::memcpy(s_save, c0_s, n * sizeof(uint32_t));
    }
    
    // 6. Calculate [a*s]_Rq using polynomial multiplication in NTT form.
    ntt_form_poly_mod_mult(c0_s, c1, n, mod_value, const_ratio);
    
    // 7. Negate [a*s]_Rq to get [-a*s]_Rq.
    poly_negate_mod(c0_s, n, mod_value);
    
    // 8. Process plaintext + error into ntt_pte.
    reduce_pte(pt_with_error, n, mod_value, const_ratio, ntt_pte);

    // 9. Apply NTT to plaintext + error.
    ntt_2(n, logn, mod_value, const_ratio, ntt_pte);
    
    // 10. Add to ciphertext.
    poly_add_mod(c0_s, ntt_pte, n, mod_value);
}

// Implementation of the SYCL pipeline function
void pipeline(
    sycl::queue q,
    sycl::device device,
    size_t n,                           // Polynomial degree
    size_t logn,                        // Log of polynomial degree
    double scale,                       // Scale value
    sycl::buffer<complex_double, 1>& encoding_buf,
    sycl::buffer<int8_t, 1>& error_samples_buf,
    sycl::buffer<int64_t, 1>& pt_with_error_buf
) {
    try {
        std::cout << "Running pipelined kernels on device: "
                  << device.get_info<sycl::info::device::name>().c_str()
                  << std::endl;
                  
        // Submit the entrance kernel to read from buffers and write to pipes
        auto entrance_event = q.submit([&](sycl::handler &h) {
            EntranceKernel(n, encoding_buf, error_samples_buf)(h);
        });
        
        // Submit the IFFT kernel that reads from and writes to pipes
        auto ifft_event = q.submit([&](sycl::handler &h) {
            h.depends_on(entrance_event);
            IFFTKernel(n, logn)(h);
        });
        
        // Submit the Scale and Convert kernel that reads from pipes and writes to a pipe
        auto scale_event = q.submit([&](sycl::handler &h) {
            h.depends_on(ifft_event);
            ScaleAndConvertKernel(n, scale)(h);
        });
        
        // Submit the exit kernel that reads from a pipe and writes to a buffer
        auto exit_event = q.submit([&](sycl::handler &h) {
            h.depends_on(scale_event);
            ExitKernel(n, pt_with_error_buf)(h);
        });
        
        // Wait for all operations to complete
        exit_event.wait();
        
        std::cout << "Pipelined operations completed successfully" << std::endl;
    } catch (sycl::exception const &e) {
        std::cerr << "Caught a synchronous SYCL exception in pipeline: "
                  << e.what() << "\n";
        std::exit(1);
    }
}

// Implementation of the first ntt function
void ntt_1(size_t n, size_t logn, uint32_t mod_value, const uint32_t* const_ratio, uint32_t *vec) {
    // Optionally, add input validation assertions as needed

    // Create a SYCL buffer for the vector
    sycl::buffer<uint32_t, 1> vec_buf(vec, sycl::range<1>(n));

    // Choose the device selector (using the FPGA emulator selector)
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;

    try {
        sycl::queue q{selector};
        std::cout << "Running First NTT on device: "
                  << q.get_device().get_info<sycl::info::device::name>().c_str()
                  << std::endl;

        // Submit work using the updated NTTKernel that has integrated root calculation
        q.submit([&](sycl::handler &h) {
            // Note: No longer need to call get_ntt_root() externally
            NTTKernel_1(n, logn, mod_value, const_ratio, vec_buf)(h);
        }).wait();
    } catch (sycl::exception const &e) {
        std::cerr << "Caught a synchronous SYCL exception in ntt: "
                  << e.what() << "\n";
        std::exit(1);
    }
}

// Implementation of the second ntt function
void ntt_2(size_t n, size_t logn, uint32_t mod_value, const uint32_t* const_ratio, uint32_t *vec) {
    // Optionally, add input validation assertions as needed

    // Create a SYCL buffer for the vector
    sycl::buffer<uint32_t, 1> vec_buf(vec, sycl::range<1>(n));

    // Choose the device selector (using the FPGA emulator selector)
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;

    try {
        sycl::queue q{selector};
        std::cout << "Running Second NTT on device: "
                  << q.get_device().get_info<sycl::info::device::name>().c_str()
                  << std::endl;

        // Submit work using the updated NTTKernel that has integrated root calculation
        q.submit([&](sycl::handler &h) {
            // Note: No longer need to call get_ntt_root() externally
            NTTKernel_2(n, logn, mod_value, const_ratio, vec_buf)(h);
        }).wait();
    } catch (sycl::exception const &e) {
        std::cerr << "Caught a synchronous SYCL exception in ntt: "
                  << e.what() << "\n";
        std::exit(1);
    }
}

// Public function to perform polynomial multiplication in NTT form
void ntt_form_poly_mod_mult(uint32_t *a, const uint32_t *b, size_t n, uint32_t mod_value, const uint32_t* const_ratio) {
    // Create SYCL buffers
    sycl::buffer<uint32_t, 1> a_buf(a, sycl::range<1>(n));
    sycl::buffer<uint32_t, 1> b_buf(const_cast<uint32_t*>(b), sycl::range<1>(n));
    
    // Create SYCL queue
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
    
    try {
        sycl::queue q{selector};
        
        // Print device info
        std::cout << "Running NTT Polynomial Multiplication on device: "
                  << q.get_device().get_info<sycl::info::device::name>().c_str()
                  << std::endl;
        
        // Submit and execute the kernel
        q.submit(PolyMultNTTKernel(n, mod_value, const_ratio, a_buf, b_buf)).wait();
        
    } catch (sycl::exception const &e) {
        std::cerr << "Caught a synchronous SYCL exception in poly_mult_mod_ntt_form_inpl: "
                  << e.what() << "\n";
        std::exit(1);
    }
}

// Public function to negate polynomial coefficients modulo q
void poly_negate_mod(uint32_t *p, size_t n, uint32_t mod_value) {
    // Create SYCL buffer for the polynomial
    sycl::buffer<uint32_t, 1> p_buf(p, sycl::range<1>(n));
    
    // Create SYCL queue
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
    
    try {
        sycl::queue q{selector};
        
        // Print device info
        std::cout << "Running Polynomial Negation on device: "
                  << q.get_device().get_info<sycl::info::device::name>().c_str()
                  << std::endl;
        
        // Submit and execute the kernel
        q.submit(PolyNegModKernel(n, mod_value, p_buf)).wait();
        
    } catch (sycl::exception const &e) {
        std::cerr << "Caught a synchronous SYCL exception in poly_neg_mod: "
                  << e.what() << "\n";
        std::exit(1);
    }
}
    
// Public function to perform modular reduction of int64_t values
void reduce_pte(const int64_t *conj_vals_int, size_t n, uint32_t mod_value, 
                const uint32_t* const_ratio, uint32_t *out) {
    // Create SYCL buffers
    sycl::buffer<int64_t, 1> conj_vals_int_buf(const_cast<int64_t*>(conj_vals_int), sycl::range<1>(n));
    sycl::buffer<uint32_t, 1> out_buf(out, sycl::range<1>(n));
    
    // Create SYCL queue
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
    
    try {
        sycl::queue q{selector};
        
        // Print device info
        std::cout << "Running Modular Reduction on device: "
                    << q.get_device().get_info<sycl::info::device::name>().c_str()
                    << std::endl;
        
        // Submit and execute the kernel
        q.submit(ReduceSetPTEKernel(n, mod_value, const_ratio, conj_vals_int_buf, out_buf)).wait();
        
    } catch (sycl::exception const &e) {
        std::cerr << "Caught a synchronous SYCL exception in reduce_set_pte: "
                    << e.what() << "\n";
        std::exit(1);
    }
}
    
// Public function to add two polynomials modulo q
void poly_add_mod(uint32_t *p1, const uint32_t *p2, size_t n, uint32_t mod_value) {
    // Create SYCL buffers
    sycl::buffer<uint32_t, 1> p1_buf(p1, sycl::range<1>(n));
    sycl::buffer<uint32_t, 1> p2_buf(const_cast<uint32_t*>(p2), sycl::range<1>(n));
    
    // Create SYCL queue
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
    
    try {
        sycl::queue q{selector};
        
        // Print device info
        std::cout << "Running Polynomial Addition on device: "
                    << q.get_device().get_info<sycl::info::device::name>().c_str()
                    << std::endl;
        
        // Submit and execute the kernel
        q.submit(PolyAddModKernel(n, mod_value, p1_buf, p2_buf)).wait();
        
    } catch (sycl::exception const &e) {
        std::cerr << "Caught a synchronous SYCL exception in poly_add_mod_inpl: "
                    << e.what() << "\n";
        std::exit(1);
    }
}