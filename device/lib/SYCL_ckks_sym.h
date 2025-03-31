#pragma once

#include <stddef.h>
#include <stdint.h>

// Define complex_double type
#ifdef __cplusplus
#include <complex>
typedef std::complex<double> complex_double;
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * SYCL-accelerated combined encode and encrypt function for CKKS symmetric encryption.
 * This is the C interface to the SYCL implementation using only standard C types
 * with unpacked struct values.
 * 
 * @param n                  Polynomial degree
 * @param logn               Log of polynomial degree
 * @param scale              CKKS scale value
 * @param mod_value          Modulus value (q)
 * @param const_ratio        Pointer to modulus const_ratio array
 * @param encoding_buffer    Buffer containing encoded values
 * @param expanded_s         Expanded secret key buffer
 * @param uniform_poly       Uniform polynomial (c1) buffer
 * @param error_samples      Error samples buffer
 * @param pt_with_error      Buffer for plaintext + error
 * @param ntt_pte            Scratch space for NTT of plaintext+error
 * @param c0_s               Output: 1st ciphertext component
 * @param c1                 Output: 2nd ciphertext component
 * @param s_save             Optional: Save expanded s (for testing)
 * @param c1_save            Optional: Save c1 (for testing)
 */
void SYCL_combined_encrypt(
    /* parms related values */
    size_t n,                       // Polynomial degree
    size_t logn,                    // Log of polynomial degree
    double scale,                   // Scale value
    
    /* modulus related values */
    uint32_t mod_value,             // Modulus value (q)
    const uint32_t* const_ratio,    // Const ratio for Barrett reduction
    
    /* data buffers */
    complex_double* encoding_buffer, // Buffer for encoding
    uint32_t* expanded_s,           // Expanded secret key
    uint32_t* uniform_poly,         // Uniform polynomial (c1)
    int8_t* error_samples,          // Error samples
    int64_t* pt_with_error,         // Plaintext + error
    uint32_t* ntt_pte,              // Scratch space for NTT
    uint32_t* c0_s,                 // Output: 1st ciphertext component
    uint32_t* c1,                   // Output: 2nd ciphertext component
    uint32_t* s_save,               // Optional: Save expanded s (for testing)
    uint32_t* c1_save               // Optional: Save c1 (for testing)
);

#ifdef __cplusplus
}
#endif