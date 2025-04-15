#include "SYCL_ckks_sym.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Include the pipe definitions
#include "SYCL_pipes.h"

// Include the SYCL kernel headers
//#include "SYCL_entrance.h"
//#include "SYCL_exit.h"
#include "SYCL_ifft.h"
#include "SYCL_scale_and_convert.h"
#include "SYCL_ntt.h"
#include "SYCL_poly_mult.h"
#include "SYCL_poly_add.h"
#include "SYCL_poly_neg.h"
#include "SYCL_reduce_pte.h"

#include <iostream> // For std::cout, std::cerr, std::endl

using namespace sycl;

// Forward declare all kernel names in global scope
class IFFTKernel;
class ScaleAndConvertKernel;
class NTTKernel_1;
class NTTKernel_2;
class PolyMultNTTKernel;
class PolyAddModKernel;
class PolyNegModKernel;
class ReduceSetPTEKernel;

// Function prototypes
void pipeline(
    queue q,
    size_t n,
    size_t logn,
    double scale,
    uint32_t mod_value,
    const uint32_t* const_ratio,
    buffer<std::complex<double>, 1>& encoding_buf,
    buffer<int8_t, 1>& error_samples_buf,
    buffer<uint32_t, 1>& ntt_pte_buf
);
void ntt_1(queue q, size_t n, size_t logn, uint32_t mod_value, const uint32_t* const_ratio, uint32_t *vec);
void ntt_2(queue q, size_t n, size_t logn, uint32_t mod_value, const uint32_t* const_ratio, uint32_t *vec);
void ntt_form_poly_mod_mult(queue q, uint32_t *a, const uint32_t *b, size_t n, uint32_t mod_value, const uint32_t* const_ratio);
void poly_negate_mod(queue q, uint32_t *p, size_t n, uint32_t mod_value);
void poly_add_mod(queue q, uint32_t *p1, const uint32_t *p2, size_t n, uint32_t mod_value);

// Implementation of the C-compatible function
extern "C" void SYCL_combined_encrypt(
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
    // 1. Copy uniform polynomial to c1 output.
    std::memcpy(c1, uniform_poly, n * sizeof(uint32_t));

    // 2. Save c1 if requested for testing.
    if (c1_save != nullptr) {
        std::memcpy(c1_save, uniform_poly, n * sizeof(uint32_t));
    }
    
    // 3. Copy expanded secret key to c0_s.
    std::memcpy(c0_s, expanded_s, n * sizeof(uint32_t));

    // Create SYCL buffers from the input pointers
    buffer<std::complex<double>, 1> encoding_buf(encoding_buffer, range<1>(n));
    buffer<int8_t, 1> error_samples_buf(error_samples, range<1>(n));
    buffer<uint32_t, 1> ntt_pte_buf(ntt_pte, range(n));

    // Create a SYCL selector
    #if FPGA_HARDWARE
        auto selector = ext::intel::fpga_selector_v;
    #else
        auto selector = ext::intel::fpga_emulator_selector_v;
    #endif

    // Create a SYCL queue
    queue q{selector, property::queue::enable_profiling()};
    
    // 4. Execute the pipeline to perform IFFT, scaling, and conversion
    pipeline(
        q, 
        n, 
        logn, 
        scale,
        mod_value,
        const_ratio,
        encoding_buf,
        error_samples_buf,
        ntt_pte_buf
    );

    // 5. Apply NTT to the secret key.
    // Updated call passing explicit parameters.
    ntt_1(q, n, logn, mod_value, const_ratio, c0_s);
    
    // 6. Save NTT(s) for later decryption if requested.
    if (s_save != nullptr) {
        std::memcpy(s_save, c0_s, n * sizeof(uint32_t));
    }
    
    // 7. Calculate [a*s]_Rq using polynomial multiplication in NTT form.
    ntt_form_poly_mod_mult(q, c0_s, c1, n, mod_value, const_ratio);
    
    // 8. Negate [a*s]_Rq to get [-a*s]_Rq.
    poly_negate_mod(q, c0_s, n, mod_value);

    // 9. Apply NTT to plaintext + error.
    ntt_2(q, n, logn, mod_value, const_ratio, ntt_pte);
    
    // 10. Add to ciphertext.
    poly_add_mod(q, c0_s, ntt_pte, n, mod_value);

}

// Function to perform the pipeline of kernels using pipes
void pipeline(
    queue q,
    size_t n,
    size_t logn,
    double scale,
    uint32_t mod_value,
    const uint32_t* const_ratio,
    buffer<std::complex<double>, 1>& encoding_buf,
    buffer<int8_t, 1>& error_samples_buf,
    buffer<uint32_t, 1>& ntt_pte_buf
) {
    std::cout << "[Pipeline] Starting..." << std::endl;
    try {

        // Submit the IFFT kernel
        std::cout << "[Pipeline] Submitting IFFTKernel..." << std::endl;
        auto ifft_event = q.submit([&](handler &h) {
            IFFTKernel(n, logn, encoding_buf, error_samples_buf)(h);
        });
        std::cout << "[Pipeline] Submitted IFFTKernel." << std::endl;

        // Submit the ScaleAndConvert kernel
        std::cout << "[Pipeline] Submitting ScaleAndConvertKernel..." << std::endl;
        auto scale_event = q.submit([&](handler &h) {
            h.depends_on(ifft_event);
            ScaleAndConvertKernel(n, scale)(h);
        });
        std::cout << "[Pipeline] Submitted ScaleAndConvertKernel." << std::endl;

        // Submit the ReduceSetPTE kernel
        std::cout << "[Pipeline] Submitting ReduceSetPTEKernel..." << std::endl;
        auto reduce_event = q.submit([&](handler &h) {
            h.depends_on(scale_event);
            ReduceSetPTEKernel(n, mod_value, const_ratio, ntt_pte_buf)(h);
        });
        std::cout << "[Pipeline] Submitted ReduceSetPTEKernel." << std::endl;

        // Wait for the last kernel in the sequence to complete
        std::cout << "[Pipeline] Waiting for reduce_event..." << std::endl;
        reduce_event.wait();
        std::cout << "[Pipeline] reduce_event completed." << std::endl;

        std::cout << "[Pipeline] Pipeline execution completed successfully." << std::endl;

    } catch (exception const &e) {
        std::cout << "[Pipeline] EXCEPTION CAUGHT!" << std::endl;
        std::cerr << "Caught a synchronous SYCL exception in pipeline: "
                  << e.what() << std::endl;
        std::exit(1);
    }
    std::cout << "[Pipeline] Exiting." << std::endl;
}


// Function to perofrm the first NTT
void ntt_1(queue q, size_t n, size_t logn, uint32_t mod_value, const uint32_t* const_ratio, uint32_t *vec) {
    // Create a SYCL buffer for the vector
    buffer<uint32_t, 1> vec_buf(vec, range<1>(n));

    try {
        std::cout << "Running First NTT on device: "
                  << q.get_device().get_info<info::device::name>().c_str()
                  << std::endl;

        // Submit work using the NTTKernel_1
        q.submit([&](handler &h) {
            NTTKernel_1(n, logn, mod_value, const_ratio, vec_buf)(h);
        }).wait();
    } catch (exception const &e) {
        std::cerr << "Caught a synchronous SYCL exception in ntt: "
                  << e.what() << "\n";
        std::exit(1);
    }
}

// Function to perofrm the second NTT
void ntt_2(queue q, size_t n, size_t logn, uint32_t mod_value, const uint32_t* const_ratio, uint32_t *vec) {
    // Create a SYCL buffer for the vector
    buffer<uint32_t, 1> vec_buf(vec, range<1>(n));

    try {
        std::cout << "Running Second NTT on device: "
                  << q.get_device().get_info<info::device::name>().c_str()
                  << std::endl;

        // Submit work using the NTTKernel_2
        q.submit([&](handler &h) {
            NTTKernel_2(n, logn, mod_value, const_ratio, vec_buf)(h);
        }).wait();
    } catch (exception const &e) {
        std::cerr << "Caught a synchronous SYCL exception in ntt: "
                  << e.what() << "\n";
        std::exit(1);
    }
}

// Function to perform polynomial multiplication in NTT form
void ntt_form_poly_mod_mult(queue q, uint32_t *a, const uint32_t *b, size_t n, uint32_t mod_value, const uint32_t* const_ratio) {
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