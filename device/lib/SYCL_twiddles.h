#pragma once

#include "SYCL_common.h"
#include "SYCL_data_types.h"
#include <sycl/sycl.hpp>

namespace sycl_ckks {

/**
 * IFFT Twiddle Factor Utilities
 * 
 * Optimization: Precompute twiddle factors once at kernel start instead of
 * computing cos/sin inside the butterfly loop.
 * 
 * Before: O(n log n) cos/sin calls (expensive on FPGA)
 * After:  O(n) cos/sin calls (initialization only)
 * 
 * For n=4096, log n=12, this is ~12x reduction in transcendental function calls.
 * 
 * The IFFT twiddle factor for index k is:
 *   conj(exp(2πi * k / (2N))) = exp(-2πi * k / (2N)) = cos(2πk/(2N)) - i*sin(2πk/(2N))
 * 
 * where N = POLY_N = 4096, so 2N = 8192.
 */

/**
 * Bit-reverse a value for IFFT butterfly indexing.
 * For 12-bit indices (n=4096), reverses the bit pattern.
 */
inline size_t bitrev(size_t input, size_t numbits)
{
    size_t t = (((input & 0xaaaa) >> 1) | ((input & 0x5555) << 1));
    t = (((t & 0xcccc) >> 2) | ((t & 0x3333) << 2));
    t = (((t & 0xf0f0) >> 4) | ((t & 0x0f0f) << 4));
    t = (((t & 0xff00) >> 8) | ((t & 0x00ff) << 8));
    return (numbits == 0) ? 0 : (t >> (16 - numbits));
}

/**
 * Initialize IFFT twiddle factor lookup table.
 * 
 * Computes: twiddles[k] = exp(-2πi * k / (2*POLY_N)) for k = 0 to POLY_N-1
 * 
 * This function should be called once at the start of the IFFT kernel.
 * The [[intel::initiation_interval(1)]] pragma ensures efficient pipelining on FPGA.
 * 
 * @param twiddles Output array of size POLY_N
 */
inline void init_ifft_twiddles(complex_double twiddles[POLY_N])
{
    constexpr double two_pi_over_2n = -2.0 * M_PI / static_cast<double>(POLY_N << 1);
    
    [[intel::initiation_interval(1)]]
    for (size_t k = 0; k < POLY_N; ++k) {
        double angle = two_pi_over_2n * static_cast<double>(k);
        twiddles[k] = complex_double(sycl::cos(angle), sycl::sin(angle));
    }
}

/**
 * Perform in-place IFFT using precomputed twiddle factors.
 * 
 * This is the optimized IFFT butterfly computation that uses LUT lookups
 * instead of computing cos/sin on-the-fly.
 * 
 * @param data Input/output array of size POLY_N
 * @param twiddles Precomputed twiddle factor LUT
 */
inline void ifft_butterfly_with_lut(complex_double data[POLY_N], 
                                    const complex_double twiddles[POLY_N])
{
    size_t tt = 1, hh = POLY_N / 2;
    
    #pragma unroll 4
    for (size_t i = 0; i < POLY_LOGN; i++, tt *= 2, hh /= 2) {
        for (size_t j = 0, kstart = 0; j < hh; j++, kstart += 2 * tt) {
            size_t br = bitrev(hh + j, POLY_LOGN);
            complex_double s = twiddles[br];
            
            for (size_t k = kstart; k < kstart + tt; k++) {
                complex_double u = data[k];
                complex_double v = data[k + tt];
                data[k] = u + v;
                data[k + tt] = (u - v) * s;
            }
        }
    }
}

}
