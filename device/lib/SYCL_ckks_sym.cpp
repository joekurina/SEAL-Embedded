#include "SYCL_ckks_sym.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "SYCL_common.h"
#include "SYCL_data_types.h"
#include "SYCL_pipes.h"
#include "SYCL_pipeline_entry.h"
#include "SYCL_pipeline_exit.h"
#include "SYCL_ntt.h"
#include "SYCL_ifft.h"
#include "SYCL_scale_and_reduce.h"
#include "SYCL_poly_mult_neg.h"
#include "SYCL_poly_add.h"

#include <iostream>
#include <vector>

using namespace sycl;
using namespace sycl_ckks;

static void pack_pipeline_input(
    size_t n,
    const complex_double* encoding_buffer,
    const int8_t* error_samples,
    const uint32_t* expanded_s,
    const uint32_t* uniform_poly,
    std::vector<PipelineInputBlock>& input_blocks)
{
    size_t num_blocks = n / LANES;
    input_blocks.resize(num_blocks);

    for (size_t blk = 0; blk < num_blocks; ++blk) {
        PipelineInputBlock& block = input_blocks[blk];
        pack_encoding_to_block(encoding_buffer, blk, block.encoding);
        pack_error_to_block(error_samples, blk, block.error);
        pack_scalar_to_block(expanded_s, blk, block.secret_key);
        pack_scalar_to_block(uniform_poly, blk, block.uniform_poly);
    }
}

static void unpack_pipeline_output(
    size_t n,
    const std::vector<PipelineOutputBlock>& output_blocks,
    uint32_t* c0_out,
    uint32_t* ntt_s_out,
    uint32_t* ntt_pte_out,
    bool has_ntt_s,
    bool has_ntt_pte)
{
    size_t num_blocks = n / LANES;

    for (size_t blk = 0; blk < num_blocks; ++blk) {
        const PipelineOutputBlock& block = output_blocks[blk];
        unpack_block_to_scalar(block.c0, blk, c0_out);

        if (has_ntt_s && ntt_s_out) {
            unpack_block_to_scalar(block.ntt_s, blk, ntt_s_out);
        }
        if (has_ntt_pte && ntt_pte_out) {
            unpack_block_to_scalar(block.ntt_pte, blk, ntt_pte_out);
        }
    }
}

template <int P>
std::vector<event> pipeline(
    queue& q,
    size_t n,
    double scale,
    uint32_t mod_value,
    const uint32_t* const_ratio,
    buffer<PipelineInputBlock, 1>& input_buf,
    buffer<PipelineOutputBlock, 1>& output_buf,
    bool save_ntt_s,
    bool save_ntt_pte
);

template <int P>
static void SYCL_combined_encrypt_impl(
    size_t n,
    size_t logn,
    double scale,
    uint32_t mod_value,
    const uint32_t* const_ratio,
    complex_double* encoding_buffer,
    uint32_t* expanded_s,
    uint32_t* uniform_poly,
    int8_t* error_samples,
    int64_t* pt_with_error,
    uint32_t* ntt_pte,
    uint32_t* c0_s,
    uint32_t* c1,
    uint32_t* s_save,
    uint32_t* c1_save)
{
    (void)logn;
    (void)pt_with_error;

    if (n % LANES != 0) {
        std::cerr << "[SYCL_combined_encrypt] polynomial degree must be divisible by "
                  << LANES << " for lane normalization\n";
        std::exit(1);
    }

    if (n != POLY_N) {
        std::cerr << "[SYCL_combined_encrypt] polynomial degree " << n
                  << " does not match compiled POLY_N=" << POLY_N << "\n";
        std::exit(1);
    }

    bool save_ntt_s = (s_save != nullptr);
    bool save_ntt_pte = (ntt_pte != nullptr);

    if (c1_save != nullptr) {
        std::memcpy(c1_save, uniform_poly, n * sizeof(uint32_t));
    }

    std::vector<PipelineInputBlock> input_blocks;
    pack_pipeline_input(n, encoding_buffer, error_samples, expanded_s, uniform_poly, input_blocks);

    std::vector<PipelineOutputBlock> output_blocks(n / LANES);

    buffer<PipelineInputBlock, 1> input_buf(input_blocks.data(), range(n / LANES));
    buffer<PipelineOutputBlock, 1> output_buf(output_blocks.data(), range(n / LANES));

#if FPGA_HARDWARE
    auto selector = ext::intel::fpga_selector_v;
#else
    auto selector = ext::intel::fpga_emulator_selector_v;
#endif
    queue q{selector, property::queue::enable_profiling()};

    auto events = pipeline<P>(
        q, n, scale, mod_value, const_ratio,
        input_buf, output_buf,
        save_ntt_s, save_ntt_pte);

    for (auto& ev : events) {
        ev.wait();
    }

    unpack_pipeline_output(n, output_blocks, c0_s, s_save, ntt_pte, save_ntt_s, save_ntt_pte);

    std::memcpy(c1, uniform_poly, n * sizeof(uint32_t));
}

extern "C" void SYCL_combined_encrypt(
    size_t n,
    size_t logn,
    double scale,
    uint32_t mod_value,
    const uint32_t* const_ratio,
    complex_double* encoding_buffer,
    uint32_t* expanded_s,
    uint32_t* uniform_poly,
    int8_t* error_samples,
    int64_t* pt_with_error,
    uint32_t* ntt_pte,
    uint32_t* c0_s,
    uint32_t* c1,
    uint32_t* s_save,
    uint32_t* c1_save)
{
    SYCL_combined_encrypt_impl<0>(n, logn, scale, mod_value, const_ratio, encoding_buffer, expanded_s,
                                  uniform_poly, error_samples, pt_with_error, ntt_pte, c0_s, c1,
                                  s_save, c1_save);
}

extern "C" void SYCL_combined_encrypt_pipeline(
    int pipeline_index,
    size_t n,
    size_t logn,
    double scale,
    uint32_t mod_value,
    const uint32_t* const_ratio,
    complex_double* encoding_buffer,
    uint32_t* expanded_s,
    uint32_t* uniform_poly,
    int8_t* error_samples,
    int64_t* pt_with_error,
    uint32_t* ntt_pte,
    uint32_t* c0_s,
    uint32_t* c1,
    uint32_t* s_save,
    uint32_t* c1_save)
{
    switch (pipeline_index) {
        case 0:
            SYCL_combined_encrypt_impl<0>(n, logn, scale, mod_value, const_ratio, encoding_buffer, expanded_s,
                                          uniform_poly, error_samples, pt_with_error, ntt_pte, c0_s, c1,
                                          s_save, c1_save);
            break;
        case 1:
            SYCL_combined_encrypt_impl<1>(n, logn, scale, mod_value, const_ratio, encoding_buffer, expanded_s,
                                          uniform_poly, error_samples, pt_with_error, ntt_pte, c0_s, c1,
                                          s_save, c1_save);
            break;
        case 2:
            SYCL_combined_encrypt_impl<2>(n, logn, scale, mod_value, const_ratio, encoding_buffer, expanded_s,
                                          uniform_poly, error_samples, pt_with_error, ntt_pte, c0_s, c1,
                                          s_save, c1_save);
            break;
        default:
            std::cerr << "[SYCL_combined_encrypt_pipeline] invalid pipeline index " << pipeline_index
                      << ", defaulting to 0\n";
            SYCL_combined_encrypt_impl<0>(n, logn, scale, mod_value, const_ratio, encoding_buffer, expanded_s,
                                          uniform_poly, error_samples, pt_with_error, ntt_pte, c0_s, c1,
                                          s_save, c1_save);
            break;
    }
}

// FPGA pipe consumers must be submitted before producers to prevent deadlocks
template <int P>
std::vector<event> pipeline(
    queue& q,
    size_t n,
    double scale,
    uint32_t mod_value,
    const uint32_t* const_ratio,
    buffer<PipelineInputBlock, 1>& input_buf,
    buffer<PipelineOutputBlock, 1>& output_buf,
    bool save_ntt_s,
    bool save_ntt_pte)
{
    (void)n;

    uint8_t modulus_selector = get_modulus_selector(mod_value);

    try {
        std::vector<event> events;

        events.push_back(q.submit([&](handler& h) {
            ExitKernel<P> kernel(output_buf, save_ntt_s, save_ntt_pte);
            kernel(h);
        }));

        events.push_back(q.submit([&](handler& h) {
            PolyAddKernel<P> kernel(mod_value);
            kernel(h);
        }));

        events.push_back(q.submit([&](handler& h) {
            PolyMultNegKernel<P> kernel(mod_value, const_ratio);
            kernel(h);
        }));

        events.push_back(q.submit([&](handler& h) {
            NTTKernelA<P> kernel(modulus_selector, save_ntt_s);
            kernel(h);
        }));

        events.push_back(q.submit([&](handler& h) {
            NTTKernelB<P> kernel(modulus_selector, save_ntt_pte);
            kernel(h);
        }));

        events.push_back(q.submit([&](handler& h) {
            ScaleAndReduceKernel<P> kernel(scale, mod_value, const_ratio);
            kernel(h);
        }));

        events.push_back(q.submit([&](handler& h) {
            IFFTKernel<P> kernel;
            kernel(h);
        }));

        events.push_back(q.submit([&](handler& h) {
            EntryKernel<P> kernel(input_buf);
            kernel(h);
        }));

        return events;

    } catch (std::exception const& e) {
        std::cout << "[Pipeline] EXCEPTION CAUGHT!" << std::endl;
        std::cerr << "Caught exception in pipeline: " << e.what() << std::endl;
        std::exit(1);
    }
    return {};
}
