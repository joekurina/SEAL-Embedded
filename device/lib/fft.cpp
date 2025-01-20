// fft.cpp
// Compiled as C++ but implements the same functions from fft.h
// bridging _Complex double <-> std::complex<double>.

#include "fft.h"    // has "typedef _Complex double fft_complex;"
#include <cassert>  // if we want to do asserts
#include <cmath>    // for sin, cos, M_PI
#include <complex>  // for std::complex<double>
#include <vector>   // optional for dynamic arrays

#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// If code calls bitrev(...) from outside, we must define it here.
size_t bitrev(size_t input, size_t numbits)
{
    // typical 16-bit reversal approach:
    size_t t = (((input & 0xaaaa) >> 1) | ((input & 0x5555) << 1));
    t        = (((t & 0xcccc) >> 2) | ((t & 0x3333) << 2));
    t        = (((t & 0xf0f0) >> 4) | ((t & 0x0f0f) << 4));
    t        = (((t & 0xff00) >> 8) | ((t & 0x00ff) << 8));
    return (numbits == 0) ? 0 : (t >> (16 - numbits));
}

/**
 * If we're not including <complex.h> in C++ mode, we may not have macros
 * `creal(c)` and `cimag(c)`. Define them with compiler extensions:
 */
#ifndef creal
#  define creal(c) __real__(c)
#endif
#ifndef cimag
#  define cimag(c) __imag__(c)
#endif

/** 
 * Convert std::complex<double> -> _Complex double without using `I`.
 */
static inline _Complex double to_c99(const std::complex<double> &z)
{
    _Complex double tmp = 0; // 0 + 0i
    __real__ tmp = z.real();
    __imag__ tmp = z.imag();
    return tmp;
}

/**
 * Convert _Complex double -> std::complex<double>.
 * We use our macros `creal(c)` and `cimag(c)`.
 */
static inline std::complex<double> from_c99(_Complex double c)
{
    return std::complex<double>(creal(c), cimag(c));
}

// Basic root calculation
static std::complex<double> calc_root_otf(size_t k, size_t m)
{
    double angle = 2.0 * M_PI * (double)k / (double)m;
    return std::complex<double>(std::cos(angle), std::sin(angle));
}

// --- Public API from fft.h ---

void calc_fft_roots(size_t n, size_t logn, fft_complex* roots)
{
    assert(n >= 4 && roots);

    size_t m = (n << 1);
    for (size_t i = 0; i < n; i++)
    {
        // original approach: bit-reverse i if that's the storage scheme
        size_t br = bitrev(i, logn);
        std::complex<double> z = calc_root_otf(br, m);
        roots[i] = to_c99(z);
    }
}

void calc_ifft_roots(size_t n, size_t logn, fft_complex* ifft_roots)
{
    assert(n >= 4 && ifft_roots);

    size_t m = (n << 1);
    for (size_t i = 0; i < n; i++)
    {
        // conj of the forward root at (i - 1), or bitreversed logic if needed
        std::complex<double> z = calc_root_otf((i - 1), m);
        z = std::conj(z);
        ifft_roots[i] = to_c99(z);
    }
}

void ifft_inpl(fft_complex* vec, size_t n, size_t logn, const fft_complex* roots)
{
    // 1) copy input array from _Complex double to C++
    std::vector<std::complex<double>> data(n);
    for (size_t i = 0; i < n; i++)
        data[i] = from_c99(vec[i]);

    // 2) copy roots if present (one-shot or load-full)
    std::vector<std::complex<double>> r(n);
    if (roots)
    {
        for (size_t i = 0; i < n; i++)
            r[i] = from_c99(roots[i]);
    }

    // reset root index each call
    size_t root_idx = 1;

    // 3) do the IFFT logic
    // For the "Harvey butterfly," we do:
    //   s = conj( calc_root_otf(bitrev(h + j, logn), 2n ) )  if OTF
    //   s = r[root_idx++]  if one-shot or load-full
    size_t tt = 1, h = n / 2;
    for (size_t round = 0; round < logn; round++, tt *= 2, h /= 2)
    {
        for (size_t j = 0, kstart = 0; j < h; j++, kstart += 2*tt)
        {
            std::complex<double> s;
            if (roots) 
            {
                // "one-shot" approach
                s = r[root_idx++];
            }
            else
            {
                // "on-the-fly" approach
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

    // 4) copy results back
    for (size_t i = 0; i < n; i++)
        vec[i] = to_c99(data[i]);
}

void fft_inpl(fft_complex* vec, size_t n, size_t logn, const fft_complex* roots)
{
    // 1) copy input array from _Complex double to std::complex
    std::vector<std::complex<double>> data(n);
    for (size_t i = 0; i < n; i++)
        data[i] = from_c99(vec[i]);

    // 2) copy roots if present
    std::vector<std::complex<double>> r(n);
    if (roots)
    {
        for (size_t i = 0; i < n; i++)
            r[i] = from_c99(roots[i]);
    }

    // reset root_idx each call
    size_t root_idx = 1;

    // 3) do the FFT logic
    //   s = calc_root_otf(bitrev(h + j, logn), 2n) if OTF
    //   s = r[root_idx++] if one-shot or load-full
    size_t h = 1, tt = n / 2;
    for (size_t round = 0; round < logn; round++, h *= 2, tt /= 2)
    {
        for (size_t j = 0, kstart = 0; j < h; j++, kstart += 2*tt)
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

    // 4) copy results back
    for (size_t i = 0; i < n; i++)
        vec[i] = to_c99(data[i]);
}

#ifdef __cplusplus
} // extern "C"
#endif
