#include "SYCL_ckks_sym.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Include the SYCL kernel headers
#include "SYCL_pipeline.h"
#include "SYCL_ntt.h"
#include "SYCL_poly_mult.h"
#include "SYCL_poly_add.h"
#include "SYCL_poly_neg.h"
#include "SYCL_reduce_pte.h"

using namespace sycl;

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
    // Create SYCL buffers from the input pointers
    buffer<std::complex<double>, 1> encoding_buf(encoding_buffer, range<1>(n));
    buffer<int8_t, 1> error_samples_buf(error_samples, range<1>(n));
    buffer<int64_t, 1> pt_with_error_buf(pt_with_error, range<1>(n));
    buffer<uint32_t, 1> expanded_s_buf(expanded_s, range<1>(n));
    buffer<uint32_t, 1> uniform_poly_buf(uniform_poly, range<1>(n));

    // Get host-access pointers for the expanded secret key and uniform poly
    auto expanded_s_ptr = expanded_s_buf.get_host_access().get_pointer();
    auto uniform_poly_ptr = uniform_poly_buf.get_host_access().get_pointer();
    
    // Create a SYCL selector
    #if FPGA_HARDWARE
        auto selector = ext::intel::fpga_selector_v;
    #else
        auto selector = ext::intel::fpga_emulator_selector_v;
    #endif

    // Create a SYCL queue
    queue q{selector, property::queue::enable_profiling()};
    
    // Execute the pipeline to perform IFFT, scaling, and conversion
    pipeline(q, n, logn, scale, encoding_buf, error_samples_buf, pt_with_error_buf);

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
    ntt_1(q, n, logn, mod_value, const_ratio, c0_s);
    
    // 5. Save NTT(s) for later decryption if requested.
    if (s_save != nullptr) {
        std::memcpy(s_save, c0_s, n * sizeof(uint32_t));
    }
    
    // 6. Calculate [a*s]_Rq using polynomial multiplication in NTT form.
    ntt_form_poly_mod_mult(q, c0_s, c1, n, mod_value, const_ratio);
    
    // 7. Negate [a*s]_Rq to get [-a*s]_Rq.
    poly_negate_mod(q, c0_s, n, mod_value);
    
    // 8. Process plaintext + error into ntt_pte.
    reduce_pte(q, pt_with_error, n, mod_value, const_ratio, ntt_pte);

    // 9. Apply NTT to plaintext + error.
    ntt_2(q, n, logn, mod_value, const_ratio, ntt_pte);
    
    // 10. Add to ciphertext.
    poly_add_mod(q, c0_s, ntt_pte, n, mod_value);
}