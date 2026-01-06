#pragma once

#include "SYCL_common.h"
#include "SYCL_pipes.h"
#include "SYCL_data_types.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sycl_ckks {

class IFFTKernelTask;

class IFFTKernel {
public:
    IFFTKernel() {}

    void operator()(sycl::handler& h) const {
        h.single_task<IFFTKernelTask>([=]() [[intel::kernel_args_restrict]] {

            complex_double data[POLY_N];

            [[intel::initiation_interval(1)]]
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                encoding_block block = SharedToIFFTPipe::read();
                size_t base = blk * LANES;
                data[base + 0] = block.element0;
                data[base + 1] = block.element1;
                data[base + 2] = block.element2;
                data[base + 3] = block.element3;
            }

            constexpr double neg_two_pi_over_2n = -2.0 * M_PI / static_cast<double>(POLY_N << 1);
            size_t tt = 1;
            size_t hh = POLY_N >> 1;

            for (size_t i = 0; i < POLY_LOGN; ++i) {
                size_t j = 0;
                size_t kstart = 0;
                for (; j < hh; ++j, kstart += (tt << 1)) {
                    size_t br = bitrev(hh + j, POLY_LOGN);
                    double angle = neg_two_pi_over_2n * static_cast<double>(br);
                    complex_double s(sycl::cos(angle), sycl::sin(angle));

                    for (size_t k = kstart; k < kstart + tt; ++k) {
                        complex_double u = data[k];
                        complex_double v = data[k + tt];
                        data[k] = u + v;
                        data[k + tt] = (u - v) * s;
                    }
                }
                tt <<= 1;
                hh >>= 1;
            }

            [[intel::initiation_interval(1)]]
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                size_t base = blk * LANES;
                encoding_block out;
                out.element0 = data[base + 0];
                out.element1 = data[base + 1];
                out.element2 = data[base + 2];
                out.element3 = data[base + 3];
                IFFTToScaleReducePipes::write(out);
            }
        });
    }

private:
    static size_t bitrev(size_t input, size_t numbits) {
        size_t t = (((input & 0xaaaa) >> 1) | ((input & 0x5555) << 1));
        t = (((t & 0xcccc) >> 2) | ((t & 0x3333) << 2));
        t = (((t & 0xf0f0) >> 4) | ((t & 0x0f0f) << 4));
        t = (((t & 0xff00) >> 8) | ((t & 0x00ff) << 8));
        return (numbits == 0) ? 0 : (t >> (16 - numbits));
    }
};

}
