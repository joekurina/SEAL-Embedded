#pragma once

#include "SYCL_common.h"
#include "SYCL_pipes.h"
#include "SYCL_data_types.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sycl_ckks {

template <int P>
class EntryKernelTask;

template <int P>
class EntryKernel {
private:
    mutable sycl::buffer<PipelineInputBlock, 1> input_buf;

public:
    EntryKernel(sycl::buffer<PipelineInputBlock, 1>& buf) : input_buf(buf) {}

    void operator()(sycl::handler& h) const {
        auto input = input_buf.template get_access<sycl::access::mode::read>(h);

        h.single_task<EntryKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            using Pipes = PipeSet<P>;

            [[intel::initiation_interval(1)]]
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                PipelineInputBlock block = input[blk];

                Pipes::EntryToIFFTPipe::write(block.encoding);
                Pipes::EntryToScaleReducePipe::write(block.error);
                Pipes::EntryToNTTAPipe::write(block.secret_key);
                Pipes::EntryToPolyMultNegPipe::write(block.uniform_poly);
            }
        });
    }
};

}
