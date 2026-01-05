#pragma once

#include "SYCL_common.h"
#include "SYCL_pipes.h"
#include "SYCL_data_types.h"
#include "SYCL_twiddles.h"
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
            complex_double twiddles[POLY_N];

            init_ifft_twiddles(twiddles);

            [[intel::initiation_interval(1)]]
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                encoding_block block = SharedToIFFTPipe::read();
                size_t base = blk * LANES;
                data[base + 0] = block.element0;
                data[base + 1] = block.element1;
                data[base + 2] = block.element2;
                data[base + 3] = block.element3;
            }

            ifft_butterfly_with_lut(data, twiddles);

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
};

}
