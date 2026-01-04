#pragma once

#include "SYCL_common.h"
#include "SYCL_pipes.h"
#include "SYCL_data_types.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sycl_ckks {

template <int P>
class PolyAddKernelTask;

template <int P>
class PolyAddKernel {
private:
    uint32_t mod_value;

public:
    PolyAddKernel(uint32_t mod) : mod_value(mod) {}

    void operator()(sycl::handler& h) const {
        uint32_t kernel_mod = mod_value;

        h.single_task<PolyAddKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            using Pipes = PipeSet<P>;

            [[intel::initiation_interval(1)]]
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                u32x4 neg_as = Pipes::PolyMultNegToPolyAddPipe::read();
                u32x4 ntt_pte = Pipes::NTTBToPolyAddPipe::read();

                u32x4 out;
                out.element0 = mod_add(neg_as.element0, ntt_pte.element0, kernel_mod);
                out.element1 = mod_add(neg_as.element1, ntt_pte.element1, kernel_mod);
                out.element2 = mod_add(neg_as.element2, ntt_pte.element2, kernel_mod);
                out.element3 = mod_add(neg_as.element3, ntt_pte.element3, kernel_mod);

                Pipes::PolyAddToExitPipe::write(out);
            }
        });
    }
};

}
