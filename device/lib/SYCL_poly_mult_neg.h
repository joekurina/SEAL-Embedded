#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_pipes.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

template <int P>
class PolyMultNegNTTKernelTask;

// Kernel: multiply pointwise in NTT domain and negate, operating on packed 4-lane blocks.
template <int P>
class PolyMultNegNTTKernelT {
private:
    size_t n;
    uint32_t mod_value;
    const uint32_t* const_ratio;              // Barrett reduction constants
    mutable sycl::buffer<u32x4_input, 1> b_acc; // Input buffer packed

public:
    PolyMultNegNTTKernelT(size_t n_val, uint32_t mod_val, const uint32_t* const_ratio_val,
                          sycl::buffer<u32x4_input, 1>& b_buf)
        : n(n_val), mod_value(mod_val), const_ratio(const_ratio_val), b_acc(b_buf) {}

    void operator()(sycl::handler& h) const {
        auto b_blocks = b_acc.get_access<sycl::access::mode::read>(h);

        size_t kernel_n = n;
        uint32_t kernel_mod_val = mod_value;
        const uint32_t* kernel_const_ratio = const_ratio;

        h.single_task<PolyMultNegNTTKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            for (size_t blk = 0; blk < kernel_n / 4; ++blk) {
            using PipeSet = CKKS_PIPE_SET<P>;
            u32x4_input a_block = PipeSet::NTTToPolyMultNegPipe::read();
                u32x4_input b_block = b_blocks[blk];
                u32x4_input out_block{};

                for (size_t lane = 0; lane < 4; ++lane) {
                    uint32_t a_val = reinterpret_cast<const uint32_t*>(&a_block)[lane];
                    uint32_t b_val = reinterpret_cast<const uint32_t*>(&b_block)[lane];

                    uint64_t wide = static_cast<uint64_t>(a_val) * static_cast<uint64_t>(b_val);
                    uint32_t product[2];
                    product[0] = static_cast<uint32_t>(wide & 0xFFFFFFFFu);
                    product[1] = static_cast<uint32_t>((wide >> 32) & 0xFFFFFFFFu);

                    uint32_t right_hw;
                    {
                        uint64_t rt_temp = static_cast<uint64_t>(product[0]) * static_cast<uint64_t>(kernel_const_ratio[0]);
                        right_hw = static_cast<uint32_t>((rt_temp >> 32) & 0xFFFFFFFFu);
                    }

                    uint32_t middle_temp[2];
                    {
                        uint64_t mt_temp = static_cast<uint64_t>(product[0]) * static_cast<uint64_t>(kernel_const_ratio[1]);
                        middle_temp[0] = static_cast<uint32_t>(mt_temp & 0xFFFFFFFFu);
                        middle_temp[1] = static_cast<uint32_t>((mt_temp >> 32) & 0xFFFFFFFFu);
                    }

                    uint32_t middle_lw = right_hw + middle_temp[0];
                    uint32_t middle_lw_carry = static_cast<uint8_t>(middle_lw < right_hw);
                    uint32_t middle_hw = middle_temp[1] + middle_lw_carry;

                    uint32_t middle2_temp[2];
                    {
                        uint64_t mt2_temp = static_cast<uint64_t>(product[1]) * static_cast<uint64_t>(kernel_const_ratio[0]);
                        middle2_temp[0] = static_cast<uint32_t>(mt2_temp & 0xFFFFFFFFu);
                        middle2_temp[1] = static_cast<uint32_t>((mt2_temp >> 32) & 0xFFFFFFFFu);
                    }

                    uint32_t middle2_lw = middle_lw + middle2_temp[0];
                    uint32_t middle2_lw_carry = static_cast<uint8_t>(middle2_lw < middle_lw);
                    uint32_t middle2_hw = middle2_temp[1] + middle2_lw_carry;

                    uint32_t tmp = product[1] * kernel_const_ratio[1] + middle_hw + middle2_hw;
                    tmp = product[0] - tmp * kernel_mod_val;

                    int32_t is_ge_q = static_cast<int32_t>(tmp >= kernel_mod_val);
                    uint32_t mask_red = static_cast<uint32_t>(-is_ge_q);
                    uint32_t mult_result = tmp - (kernel_mod_val & mask_red);

                    int32_t non_zero = static_cast<int32_t>(mult_result != 0);
                    uint32_t mask_neg = static_cast<uint32_t>(-non_zero);
                    uint32_t neg_result = (kernel_mod_val - mult_result) & mask_neg;

                    reinterpret_cast<uint32_t*>(&out_block)[lane] = neg_result;
                }

                PipeSet::PolyMultNegToPolyAddModPipe::write(out_block);
            }
        });
    }
};

// Backwards-compatible alias for pipeline P = 0.
using PolyMultNegNTTKernel = PolyMultNegNTTKernelT<0>;
