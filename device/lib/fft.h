#pragma once

#include <stddef.h>  // for size_t

#ifdef __cplusplus
extern "C" {
#endif

/**
 * A simple struct to hold one complex value in a C-compatible way.
 * The fields are named `re` and `im`.
 */
typedef struct fft_complex
{
    double re;
    double im;
} fft_complex;

/**
 * Bit-reversal function for the FFT/IFFT.
 *
 * Example: bitrev(6, 3) => 0b110 -> reversed -> 0b011 = 3
 *
 * Requires: numbits <= 16
 *
 * @param[in] input    Value to bit-reverse
 * @param[in] numbits  Number of bits required to represent input (<=16)
 * @returns            'input' in bit-reversed order
 */
size_t bitrev(size_t input, size_t numbits);

/**
 * Generates roots for the FFT from scratch (in bit-reversed order).
 *
 * @param[in]  n      The FFT size (number of roots to generate)
 * @param[in]  logn   Floor(log2(n))
 * @param[out] roots  Array of `fft_complex` of length n (bit-reversed order).
 */
void calc_fft_roots(size_t n, size_t logn, fft_complex* roots);

/**
 * Generates roots for the IFFT from scratch (in bit-reversed order).
 *
 * @param[in]  n           The IFFT size
 * @param[in]  logn        Floor(log2(n))
 * @param[out] ifft_roots  Array of `fft_complex` of length n (bit-reversed order).
 */
void calc_ifft_roots(size_t n, size_t logn, fft_complex* ifft_roots);

/**
 * In-place Inverse Fast-Fourier Transform using the Harvey butterfly.
 * Does NOT divide the final result by `n`.
 *
 * @param[in,out] vec   Array of `fft_complex` of length n (input/output)
 * @param[in]     n     IFFT size (polynomial degree)
 * @param[in]     logn  Floor(log2(n))
 * @param[in]     roots [Optional]. As set by `calc_ifft_roots` or similar.
 */
void ifft_inpl(fft_complex* vec, size_t n, size_t logn, const fft_complex* roots);

/**
 * In-place forward Fast-Fourier Transform using the Harvey butterfly.
 * `roots` is ignored (and may be null) if some OTF approach is chosen.
 *
 * @param[in,out] vec   Array of `fft_complex` of length n (input/output)
 * @param[in]     n     FFT size (polynomial degree)
 * @param[in]     logn  Floor(log2(n))
 * @param[in]     roots [Optional]. As set by `calc_fft_roots` or similar.
 */
void fft_inpl(fft_complex* vec, size_t n, size_t logn, const fft_complex* roots);

#ifdef __cplusplus
} // extern "C"
#endif
