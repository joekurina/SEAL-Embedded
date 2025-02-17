#pragma once

#include <stddef.h>      // for size_t


/**
 * Define a typedef so code can call `fft_complex` but it’s really `_Complex double` in C or std::complex<double> in C++.
 * This might help avoid confusion in other parts of the code.
 */
#ifdef __cplusplus
#include <complex>
typedef std::complex<double> fft_complex;
extern "C" {
#else
#include <complex.h>
typedef _Complex double fft_complex;
#endif

/**
 * Bit-reversal function if other code calls it. 
 * (Remove it from here if only used internally.)
 */
size_t bitrev(size_t input, size_t numbits);

/**
 * Generate the FFT roots in `_Complex double` form (bit-reversed).
 */
//void calc_fft_roots(size_t n, size_t logn, fft_complex *roots);

/**
 * Generate the IFFT roots in `_Complex double` form (bit-reversed).
 */
//void calc_ifft_roots(size_t n, size_t logn, fft_complex *ifft_roots);

/**
 * In-place IFFT using `_Complex double`.
 */
void ifft_inpl(fft_complex *vec, size_t n, size_t logn, const fft_complex *roots);

/**
 * In-place FFT using `_Complex double`.
 */
void fft_inpl(fft_complex *vec, size_t n, size_t logn, const fft_complex *roots);

#ifdef __cplusplus
}
#endif
