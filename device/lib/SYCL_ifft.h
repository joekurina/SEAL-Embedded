#pragma once

#include "SYCL_common.h"
#include "SYCL_pipes.h"
#include "SYCL_data_types.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sycl_ckks {

template <int P>
class IFFTKernelTask;

template <int P>
class IFFTKernel {
public:
    IFFTKernel() {}

    void operator()(sycl::handler& h) const {
        h.single_task<IFFTKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            using Pipes = PipeSet<P>;

            auto bitrev = [](size_t input, size_t numbits) -> size_t {
                size_t t = (((input & 0xaaaa) >> 1) | ((input & 0x5555) << 1));
                t = (((t & 0xcccc) >> 2) | ((t & 0x3333) << 2));
                t = (((t & 0xf0f0) >> 4) | ((t & 0x0f0f) << 4));
                t = (((t & 0xff00) >> 8) | ((t & 0x00ff) << 8));
                return (numbits == 0) ? 0 : (t >> (16 - numbits));
            };

            auto calc_root = [](size_t k, size_t m) -> complex_double {
                double angle = 2.0 * M_PI * static_cast<double>(k) / static_cast<double>(m);
                return complex_double(sycl::cos(angle), sycl::sin(angle));
            };

            complex_double data[POLY_N];

            [[intel::initiation_interval(1)]]
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                encoding_block block = Pipes::EntryToIFFTPipe::read();
                size_t base = blk * LANES;
                data[base + 0] = block.element0;
                data[base + 1] = block.element1;
                data[base + 2] = block.element2;
                data[base + 3] = block.element3;
            }

            size_t tt = 1, hh = POLY_N / 2;
            #pragma unroll
            for (size_t i = 0; i < POLY_LOGN; i++, tt *= 2, hh /= 2) {
                for (size_t j = 0, kstart = 0; j < hh; j++, kstart += 2 * tt) {
                    size_t br = bitrev(hh + j, POLY_LOGN);
                    complex_double s = std::conj(calc_root(br, POLY_N << 1));

                    for (size_t k = kstart; k < kstart + tt; k++) {
                        complex_double u = data[k];
                        complex_double v = data[k + tt];
                        data[k] = u + v;
                        data[k + tt] = (u - v) * s;
                    }
                }
            }

            [[intel::initiation_interval(1)]]
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                size_t base = blk * LANES;
                encoding_block out;
                out.element0 = data[base + 0];
                out.element1 = data[base + 1];
                out.element2 = data[base + 2];
                out.element3 = data[base + 3];
                Pipes::IFFTToScaleReducePipe::write(out);
            }
        });
    }
};

}
