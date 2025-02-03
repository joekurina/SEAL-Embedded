// fft-lib.cc
// Implementation of FFT functions for the FFT library.
// Bridges between C99 _Complex double and C++ std::complex<double>.

#include "fft-lib.h"    // uses typedef _Complex double fft_complex
#include <cassert>      // for assert()
#include <cmath>        // for sin, cos, M_PI
#include <complex>      // for std::complex<double>
#include <vector>       // for std::vector

#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Bit reversal helper function.
// Typical 16-bit reversal approach.
size_t bitrev(size_t input, size_t numbits)
{
    size_t t = (((input & 0xaaaa) >> 1) | ((input & 0x5555) << 1));
    t        = (((t & 0xcccc) >> 2) | ((t & 0x3333) << 2));
    t        = (((t & 0xf0f0) >> 4) | ((t & 0x0f0f) << 4));
    t        = (((t & 0xff00) >> 8) | ((t & 0x00ff) << 8));
    return (numbits == 0) ? 0 : (t >> (16 - numbits));
}

// If <complex.h> isn’t included in C++ mode, define creal and cimag.
#ifndef creal
#  define creal(c) __real__(c)
#endif
#ifndef cimag
#  define cimag(c) __imag__(c)
#endif

// Convert from C++ std::complex<double> to C99 _Complex double.
static inline _Complex double to_c99(const std::complex<double>& z)
{
    _Complex double tmp = 0;  // initialize to 0+0i
    __real__ tmp = z.real();
    __imag__ tmp = z.imag();
    return tmp;
}

// Convert from C99 _Complex double to C++ std::complex<double>.
static inline std::complex<double> from_c99(_Complex double c)
{
    return std::complex<double>(creal(c), cimag(c));
}

// Compute the k-th root of unity on the fly.
static std::complex<double> calc_root_otf(size_t k, size_t m)
{
    double angle = 2.0 * M_PI * static_cast<double>(k) / static_cast<double>(m);
    return std::complex<double>(std::cos(angle), std::sin(angle));
}

// Compute FFT roots for forward FFT.
void calc_fft_roots(size_t n, size_t logn, fft_complex* roots)
{
    assert(n >= 4 && roots);

    size_t m = (n << 1);
    for (size_t i = 0; i < n; i++)
    {
        // Optionally, perform bit reversal on the index.
        size_t br = bitrev(i, logn);
        std::complex<double> z = calc_root_otf(br, m);
        roots[i] = to_c99(z);
    }
}

// Compute FFT roots for inverse FFT.
void calc_ifft_roots(size_t n, size_t logn, fft_complex* ifft_roots)
{
    assert(n >= 4 && ifft_roots);

    size_t m = (n << 1);
    for (size_t i = 0; i < n; i++)
    {
        // Calculate the conjugate of the forward FFT root (or adjust bit-reversal as needed).
        std::complex<double> z = calc_root_otf((i - 1), m);
        z = std::conj(z);
        ifft_roots[i] = to_c99(z);
    }
}

// In-place inverse FFT implementation.
void ifft_inpl(fft_complex* vec, size_t n, size_t logn, const fft_complex* roots)
{
    // 1) Copy the input array from _Complex double to std::complex<double>.
    std::vector<std::complex<double>> data(n);
    for (size_t i = 0; i < n; i++)
        data[i] = from_c99(vec[i]);

    // 2) Copy roots if provided.
    std::vector<std::complex<double>> r(n);
    if (roots)
    {
        for (size_t i = 0; i < n; i++)
            r[i] = from_c99(roots[i]);
    }

    // Reset root index for each call.
    size_t root_idx = 1;

    // 3) Perform the inverse FFT (IFFT) using the “Harvey butterfly” approach.
    size_t tt = 1, h = n / 2;
    for (size_t round = 0; round < logn; round++, tt *= 2, h /= 2)
    {
        for (size_t j = 0, kstart = 0; j < h; j++, kstart += 2 * tt)
        {
            std::complex<double> s;
            if (roots)
            {
                // Use the precomputed roots.
                s = r[root_idx++];
            }
            else
            {
                // Compute on the fly.
                size_t br = bitrev(h + j, logn);
                s = std::conj(calc_root_otf(br, n << 1));
            }

            for (size_t k = kstart; k < kstart + tt; k++)
            {
                auto u = data[k];
                auto w = data[k + tt];
                data[k]    = u + w;
                data[k+tt] = (u - w) * s;
            }
        }
    }

    // 4) Copy the result back to the caller’s array.
    for (size_t i = 0; i < n; i++)
        vec[i] = to_c99(data[i]);
}

// In-place forward FFT implementation.
void fft_inpl(fft_complex* vec, size_t n, size_t logn, const fft_complex* roots)
{
    // 1) Copy the input array from _Complex double to std::complex<double>.
    std::vector<std::complex<double>> data(n);
    for (size_t i = 0; i < n; i++)
        data[i] = from_c99(vec[i]);

    // 2) Copy roots if provided.
    std::vector<std::complex<double>> r(n);
    if (roots)
    {
        for (size_t i = 0; i < n; i++)
            r[i] = from_c99(roots[i]);
    }

    // Reset root index for each call.
    size_t root_idx = 1;

    // 3) Perform the FFT using a butterfly algorithm.
    size_t h = 1, tt = n / 2;
    for (size_t round = 0; round < logn; round++, h *= 2, tt /= 2)
    {
        for (size_t j = 0, kstart = 0; j < h; j++, kstart += 2 * tt)
        {
            std::complex<double> s;
            if (roots)
            {
                s = r[root_idx++];
            }
            else
            {
                size_t br = bitrev(h + j, logn);
                s = calc_root_otf(br, n << 1);
            }

            for (size_t k = kstart; k < kstart + tt; k++)
            {
                auto u = data[k];
                auto w = data[k + tt] * s;
                data[k]    = u + w;
                data[k+tt] = u - w;
            }
        }
    }

    // 4) Copy the result back.
    for (size_t i = 0; i < n; i++)
        vec[i] = to_c99(data[i]);
}

#ifdef __cplusplus
} // extern "C"
#endif
