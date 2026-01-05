#pragma once

#include "SYCL_common.h"
#include "SYCL_pipes.h"
#include "SYCL_data_types.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sycl_ckks {

template <int P>
class PerModulusEntryKernelTask;

template <int P>
class PerModulusEntryKernel {
private:
    mutable sycl::buffer<PerModulusInputBlock, 1> input_buf;

public:
    PerModulusEntryKernel(sycl::buffer<PerModulusInputBlock, 1>& buf) : input_buf(buf) {}

    void operator()(sycl::handler& h) const {
        auto input = input_buf.template get_access<sycl::access::mode::read>(h);

        h.single_task<PerModulusEntryKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            using Pipes = PipeSet<P>;

            [[intel::initiation_interval(1)]]
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                PerModulusInputBlock block = input[blk];

                Pipes::EntryToNTTAPipe::write(block.secret_key);
                Pipes::EntryToPolyMultNegPipe::write(block.uniform_poly);
            }
        });
    }
};

}
