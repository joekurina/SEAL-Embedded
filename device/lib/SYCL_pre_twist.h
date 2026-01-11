#pragma once

#include "SYCL_common.h"
#include "SYCL_pipes.h"
#include "SYCL_data_types.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sycl_ckks {

// =============================================================================
// NWC (Negacyclic) Pre-Twist Kernel
// =============================================================================
//
// BACKGROUND:
// CKKS encoding requires a negacyclic IFFT on ring Z[X]/(X^N + 1), using
// 2N-th roots of unity: ψ = exp(πi/N). Our RTL IFFT uses standard N-th roots:
// ω = exp(2πi/N), corresponding to ring Z[X]/(X^N - 1).
//
// SOLUTION:
// Apply a "pre-twist" to input data before the standard IFFT. Multiplying
// each frequency-domain sample X[k] by ψ^k = exp(πik/N) converts the standard
// IFFT result into the negacyclic IFFT result.
//
// Mathematically: IFFT_nwc(X) = IFFT_std(twist(X))
// where twist(X)[k] = X[k] * exp(πik/N)
//
// INDEX MAPPING:
// The encoding buffer stores values at BIT-REVERSED indices. A value that
// logically belongs at frequency index k is stored at buffer position bitrev(k).
// Therefore, for buffer position j, the corresponding frequency index is
// k = bitrev(j), and we twist by exp(πi * bitrev(j) / N).
//
// IMPLEMENTATION:
// For each buffer position j, the twist factor is:
//   twist[j] = exp(πi * bitrev(j) / N) = cos(π * bitrev(j) / N) + i*sin(π * bitrev(j) / N)
//
// Complex multiplication:
//   (a + bi) * (cos(θ) + i*sin(θ)) = (a*cos(θ) - b*sin(θ)) + i*(a*sin(θ) + b*cos(θ))
// =============================================================================

// 12-bit bit-reversal for N=4096 (POLY_LOGN=12)
inline size_t bitrev12(size_t x) {
    x = ((x & 0xAAA) >> 1) | ((x & 0x555) << 1);
    x = ((x & 0xCCC) >> 2) | ((x & 0x333) << 2);
    x = ((x & 0xF0F) >> 4) | ((x & 0x0F0) << 4);
    x = ((x >> 8) | (x << 8)) & 0xFFF;
    return x;
}

class PreTwistKernelTask;

class PreTwistKernel {
public:
    PreTwistKernel() {}

    void operator()(sycl::handler& h) const {
        h.single_task<PreTwistKernelTask>([=]() [[intel::kernel_args_restrict]] {

            // Precompute: π/N for NWC twist angle computation
            constexpr double pi_over_n = M_PI / static_cast<double>(POLY_N);

            [[intel::initiation_interval(1)]]
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                encoding_block enc = SharedToPreTwistPipe::read();

                // BYPASS MODE: Pass through without twist to diagnose RTL behavior
                PreTwistToIFFTPipe::write(enc);
            }
        });
    }
};

}
