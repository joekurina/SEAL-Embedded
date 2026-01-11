#pragma once

#include "SYCL_common.h"
#include "SYCL_pipes.h"
#include "SYCL_data_types.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sycl_ckks {

// =============================================================================
// Scale and Reduce Kernel
// =============================================================================
//
// This kernel takes the real output from the negacyclic IFFT (after pre-twist
// has been applied to the input), scales it, adds error, and reduces mod q.
//
// The pre-twist is applied before the IFFT in a separate kernel, so this
// kernel only needs to:
// 1. Scale by (scale / N) to normalize the IFFT and apply CKKS scale
// 2. Add error samples for noise flooding
// 3. Barrett reduce to get result mod q
//
// After a correct negacyclic IFFT, the output should be purely real
// (imaginary parts ≈ 0), so we only use the real component.
// =============================================================================

template <int P>
class ScaleAndReduceKernelTask;

template <int P>
class ScaleAndReduceKernel {
private:
    double scale;
    uint32_t mod_value;
    uint32_t const_ratio[2];

public:
    ScaleAndReduceKernel(double s, uint32_t mod, const uint32_t* cr)
        : scale(s), mod_value(mod) {
        const_ratio[0] = cr[0];
        const_ratio[1] = cr[1];
    }

    void operator()(sycl::handler& h) const {
        double kernel_scale = scale;
        uint32_t kernel_mod = mod_value;
        uint32_t kernel_cr0 = const_ratio[0];
        uint32_t kernel_cr1 = const_ratio[1];

        h.single_task<ScaleAndReduceKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            using Pipes = PipeSet<P>;

            // RTL IFFT outputs unnormalized sum; scale by 1/N to normalize
            double n_inv = kernel_scale / static_cast<double>(POLY_N);

            [[intel::initiation_interval(1)]]
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                encoding_block enc = IFFTToScaleReducePipes::PipeAt<P>::read();
                i8x4 err = ErrorToScaleReducePipes::PipeAt<P>::read();

                // Scale and round to integer (use real part only)
                double scaled0 = sycl::round(enc.element0.real() * n_inv);
                double scaled1 = sycl::round(enc.element1.real() * n_inv);
                double scaled2 = sycl::round(enc.element2.real() * n_inv);
                double scaled3 = sycl::round(enc.element3.real() * n_inv);

                // Add error term for noise flooding
                int64_t int_val0 = static_cast<int64_t>(scaled0) + err.element0;
                int64_t int_val1 = static_cast<int64_t>(scaled1) + err.element1;
                int64_t int_val2 = static_cast<int64_t>(scaled2) + err.element2;
                int64_t int_val3 = static_cast<int64_t>(scaled3) + err.element3;

                // Barrett reduction to mod q
                u32x4 out;
                out.element0 = barrett_reduce_64_core(int_val0, kernel_mod, kernel_cr0, kernel_cr1, false);
                out.element1 = barrett_reduce_64_core(int_val1, kernel_mod, kernel_cr0, kernel_cr1, false);
                out.element2 = barrett_reduce_64_core(int_val2, kernel_mod, kernel_cr0, kernel_cr1, false);
                out.element3 = barrett_reduce_64_core(int_val3, kernel_mod, kernel_cr0, kernel_cr1, false);

                Pipes::ScaleReduceToNTTBPipe::write(out);
            }
        });
    }
};

}
