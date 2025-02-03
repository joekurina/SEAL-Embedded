#pragma once

#include <stddef.h>      // for size_t
#include <complex.h>     // for _Complex double

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Define a typedef so code can refer to `fft_complex` while really using
 * C99’s `_Complex double`. This can help avoid confusion in the rest of the code.
 */
typedef _Complex double fft_complex;

/**
 * Bit-reversal function.
 *
 * Reverses the lowest `numbits` bits of the input.
 */
size_t bitrev(size_t input, size_t numbits);

/**
 * Generate the FFT roots (twiddle factors) in `_Complex double` form.
 * The roots are generated in bit-reversed order.
 */
void calc_fft_roots(size_t n, size_t logn, fft_complex *roots);

/**
 * Generate the IFFT roots (twiddle factors) in `_Complex double` form.
 * The roots are generated in bit-reversed order.
 */
void calc_ifft_roots(size_t n, size_t logn, fft_complex *ifft_roots);

/**
 * In-place IFFT implementation using `_Complex double`.
 *
 * The function performs an inverse FFT on the array `vec` of length `n`
 * with `logn` stages, optionally using precomputed roots.
 */
void ifft_inpl(fft_complex *vec, size_t n, size_t logn, const fft_complex *roots);

/**
 * In-place FFT implementation using `_Complex double`.
 *
 * The function performs a forward FFT on the array `vec` of length `n`
 * with `logn` stages, optionally using precomputed roots.
 */
void fft_inpl(fft_complex *vec, size_t n, size_t logn, const fft_complex *roots);

#ifdef __cplusplus
}
#endif
