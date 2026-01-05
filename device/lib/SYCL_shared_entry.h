#pragma once

#include "SYCL_common.h"
#include "SYCL_pipes.h"
#include "SYCL_data_types.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sycl_ckks {

class SharedEntryKernelTask;

class SharedEntryKernel {
private:
    mutable sycl::buffer<SharedInputBlock, 1> input_buf;

public:
    SharedEntryKernel(sycl::buffer<SharedInputBlock, 1>& buf) : input_buf(buf) {}

    void operator()(sycl::handler& h) const {
        auto input = input_buf.template get_access<sycl::access::mode::read>(h);

        h.single_task<SharedEntryKernelTask>([=]() [[intel::kernel_args_restrict]] {
            [[intel::initiation_interval(1)]]
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                SharedInputBlock block = input[blk];

                SharedToIFFTPipe::write(block.encoding);

                ErrorToScaleReducePipes::write(block.error);
            }
        });
    }
};

}
