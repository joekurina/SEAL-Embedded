// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

/**
@file fft.cpp

Note: All roots are mod 2n, where n is the number of elements
(e.g. polynomial degree) in the transformed vector.
See the paper for more details.
*/

#include "fft.h"        // Includes the struct fft_complex from fft.h
#include "defines.h"    // For se_assert(...) or PolySizeType
#include "util_print.h" // For printing/debug macros, if needed

#include <math.h>       // For cos, sin, M_PI, etc.
#include <stdio.h>      // For printf, etc.
#include <stdlib.h>     // For exit()

#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ------------------------------------------------------------------
// 1) Helpers for complex arithmetic on fft_complex
// ------------------------------------------------------------------

static inline fft_complex make_fft(double re, double im)
{
    fft_complex c;
    c.re = re;
    c.im = im;
    return c;
}

static inline fft_complex add_fft(fft_complex a, fft_complex b)
{
    fft_complex r;
    r.re = a.re + b.re;
    r.im = a.im + b.im;
    return r;
}

static inline fft_complex sub_fft(fft_complex a, fft_complex b)
{
    fft_complex r;
    r.re = a.re - b.re;
    r.im = a.im - b.im;
    return r;
}

static inline fft_complex mul_fft(fft_complex a, fft_complex b)
{
    // (a.re + i a.im) * (b.re + i b.im)
    // = (a.re*b.re - a.im*b.im) + i(a.re*b.im + a.im*b.re)
    fft_complex r;
    r.re = a.re*b.re - a.im*b.im;
    r.im = a.re*b.im + a.im*b.re;
    return r;
}

static inline fft_complex conj_fft(fft_complex c)
{
    fft_complex r;
    r.re = c.re;
    r.im = -(c.im);
    return r;
}

// ------------------------------------------------------------------
// 2) Angle and root calculations
// ------------------------------------------------------------------

/**
Helper function to calculate the angle of a particular root

@param[in] k  Index of root
@param[in] m  Degree of roots (i.e. 2n)
@returns      Angle of the root
*/
static double calc_angle(size_t k, size_t m)
{
    // 2 * pi * k / m
    return 2.0 * M_PI * (double)k / (double)m;
}

/**
Helper function that calculates one FFT root on-the-fly
using the angle from calc_angle(...)
*/
static fft_complex calc_root_otf(size_t k, size_t m)
{
    k &= (m - 1);  // just in case
    double angle = calc_angle(k, m);
    double re = cos(angle);
    double im = sin(angle);
    return make_fft(re, im);
}

// ------------------------------------------------------------------
// 3) Public API implementations (matching fft.h)
// ------------------------------------------------------------------

size_t bitrev(size_t input, size_t numbits)
{
    // Same bit reversal logic as before
    size_t t = (((input & 0xaaaa) >> 1) | ((input & 0x5555) << 1));
    t        = (((t & 0xcccc) >> 2) | ((t & 0x3333) << 2));
    t        = (((t & 0xf0f0) >> 4) | ((t & 0x0f0f) << 4));
    t        = (((t & 0xff00) >> 8) | ((t & 0x00ff) << 8));
    return (numbits == 0) ? 0 : (t >> (16 - numbits));
}

void calc_fft_roots(size_t n, size_t logn, fft_complex *roots)
{
    se_assert(n >= 4);
    se_assert(roots);

    // m = 2n
    PolySizeType m = (PolySizeType)(n << 1);
    for (size_t i = 0; i < n; i++)
    {
        size_t br = bitrev(i, logn);
        roots[i] = calc_root_otf(br, m);
    }
}

void calc_ifft_roots(size_t n, size_t logn, fft_complex *ifft_roots)
{
    se_assert(n >= 4);
    se_assert(ifft_roots);

    PolySizeType m = (PolySizeType)(n << 1);
    for (size_t i = 0; i < n; i++)
    {
        // conj of the forward root, but with i-1 (or i-1 + 1)
        fft_complex root = calc_root_otf(bitrev(i - 1, logn), m);
        ifft_roots[i]    = conj_fft(root);
        // ifft_roots[i] = conj_fft(calc_root_otf(bitrev(i - 1, logn) + 1, m));
    }
}

void ifft_inpl(fft_complex *vec, size_t n, size_t logn, const fft_complex *roots)
{
#if defined(SE_IFFT_LOAD_FULL) || defined(SE_IFFT_ONE_SHOT)
    se_assert(roots);
    size_t root_idx = 1;  // some indexing logic if you store roots in bit-rev order
#elif defined(SE_IFFT_OTF)
    (void)roots;          // not used
    size_t m = n << 1;    // 2n for angle
#else
    printf("IFFT option not found!\n");
    exit(0);
#endif

    // Example: n=8, logn=3 -> see the multi-line comment from your original code
    size_t tt = 1;        // size of each butterfly
    size_t h  = n / 2;    // number of groups

    for (size_t i = 0; i < logn; i++, tt *= 2, h /= 2)
    {
        for (size_t j = 0, kstart = 0; j < h; j++, kstart += (2 * tt))
        {
            fft_complex s;

#if defined(SE_IFFT_LOAD_FULL) || defined(SE_IFFT_ONE_SHOT)
            // If you have precomputed roots in bit-rev order, pick them:
            s = roots[root_idx++];
#elif defined(SE_IFFT_OTF)
            // On-the-fly root generation + conj
            s = conj_fft(calc_root_otf(bitrev(h + j, logn), m));
#else
            printf("Error! IFFT option not found!\n");
            exit(1);
#endif

            for (size_t k = kstart; k < kstart + tt; k++)
            {
                fft_complex u = vec[k];
                fft_complex v = vec[k + tt];
                // u + v
                vec[k]       = add_fft(u, v);
                // (u - v)*s
                fft_complex tmp = sub_fft(u, v);
                vec[k + tt] = mul_fft(tmp, s);
            }
        }
    }
}

void fft_inpl(fft_complex *vec, size_t n, size_t logn, const fft_complex *roots)
{
    // Similar logic to ifft_inpl, but reversed butterfly steps
    se_assert(n >= 4);

    size_t m = (size_t)(n << 1); // 2n for angle
#ifdef SE_FFT_OTF
    (void)roots;                // not used if OTF
#else
    se_assert(roots);
#endif

    size_t h  = 1;
    size_t tt = n / 2;

#if defined(SE_FFT_LOAD_FULL) || defined(SE_FFT_ONE_SHOT)
    size_t root_idx = 1;  // again, depends on how you stored the roots
#endif

    for (size_t i = 0; i < logn; i++, h *= 2, tt /= 2)
    {
        for (size_t j = 0, kstart = 0; j < h; j++, kstart += (2 * tt))
        {
            fft_complex s;

#if defined(SE_FFT_LOAD_FULL) || defined(SE_FFT_ONE_SHOT)
            s = roots[root_idx++];
#elif defined(SE_FFT_OTF)
            s = calc_root_otf(bitrev(h + j, logn), m);
#else
            printf("Error! FFT option not found!\n");
            exit(1);
#endif

            for (size_t k = kstart; k < kstart + tt; k++)
            {
                fft_complex u = vec[k];
                // v = vec[k+tt]*s
                fft_complex v = mul_fft(vec[k + tt], s);
                // (u+v), (u-v)
                vec[k]       = add_fft(u, v);
                vec[k + tt]  = sub_fft(u, v);
            }
        }
    }
}

#ifdef __cplusplus
} // extern "C"
#endif
