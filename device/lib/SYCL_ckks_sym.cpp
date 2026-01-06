#include "SYCL_ckks_sym.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "SYCL_common.h"
#include "SYCL_data_types.h"
#include "SYCL_pipes.h"
#include "SYCL_entry.h"
#include "SYCL_pipeline_exit.h"
#include "SYCL_ntt.h"
#include "SYCL_ifft.h"
#include "SYCL_scale_and_reduce.h"
#include "SYCL_poly_mult_neg_add.h"

#include <cstring>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>

using namespace sycl;
using namespace sycl_ckks;

// =============================================================================
// NWC (Negacyclic) Twist for RTL IFFT Compatibility
// =============================================================================
//
// BACKGROUND:
// The CKKS encoding requires a negacyclic IFFT operating on the polynomial
// ring Z[X]/(X^N + 1). This uses 2N-th roots of unity: ψ = exp(-πi/N).
//
// However, our DSP Builder-generated RTL IFFT uses standard N-th roots of
// unity: ω = exp(-2πi/N), which corresponds to the ring Z[X]/(X^N - 1).
//
// SOLUTION:
// Apply a "twist" to the input data before the standard IFFT. Multiplying
// each input sample x[k] by ψ^k = exp(πik/N) converts the standard IFFT
// result into the negacyclic IFFT result we need.
//
// Mathematically: IFFT_nwc(x) = IFFT_std(twist(x))
// where twist(x)[k] = x[k] * exp(πik/N)
//
// FUTURE REFACTORING:
// When regenerating RTL with correct NWC twiddle factors built into the
// hardware, remove this twist function and the call to apply_nwc_twist()
// in SYCL_encrypt(). The RTL will then directly compute the negacyclic
// IFFT without host-side preprocessing.
//
// To generate NWC-native RTL, modify the DSP Builder twiddle ROM .hex files
// to use ψ^k = exp(-πik/N) instead of ω^k = exp(-2πik/N).
// =============================================================================

/**
 * @brief Apply NWC twist to convert standard IFFT input to negacyclic IFFT input.
 *
 * Multiplies each element x[k] by exp(πik/N) where N is the polynomial degree.
 * This pre-processing step allows a standard IFFT to produce negacyclic results.
 *
 * @param n     Polynomial degree (number of complex samples)
 * @param data  Complex encoding buffer to twist (modified in-place)
 *
 * @note Remove this function when RTL IFFT has native NWC twiddle factors.
 */
static void apply_nwc_twist(size_t n, complex_double* data)
{
    const double pi_over_n = M_PI / static_cast<double>(n);

    for (size_t k = 0; k < n; ++k) {
        double angle = pi_over_n * static_cast<double>(k);
        complex_double twist(std::cos(angle), std::sin(angle));
        data[k] *= twist;
    }
}

static void pack_input(
    size_t n,
    const complex_double* encoding_buffer,
    const int8_t* error_samples,
    const uint32_t* const* secret_keys,
    const uint32_t* const* c1_polys,
    std::vector<PipelineInputBlock>& input_blocks)
{
    size_t num_blocks = n / LANES;
    input_blocks.resize(num_blocks);

    for (size_t blk = 0; blk < num_blocks; ++blk) {
        PipelineInputBlock& block = input_blocks[blk];
        pack_encoding_to_block(encoding_buffer, blk, block.encoding);
        pack_error_to_block(error_samples, blk, block.error);

        for (size_t p = 0; p < NUM_MODULI; ++p) {
            pack_scalar_to_block(secret_keys[p], blk, block.secret_key[p]);
            pack_scalar_to_block(c1_polys[p], blk, block.c1[p]);
        }
    }
}

static void unpack_output(
    size_t n,
    const std::vector<PerModulusOutputBlock>& output_blocks,
    uint32_t* c0_out,
    uint32_t* ntt_s_out,
    uint32_t* ntt_pte_out,
    bool has_ntt_s,
    bool has_ntt_pte)
{
    size_t num_blocks = n / LANES;

    for (size_t blk = 0; blk < num_blocks; ++blk) {
        const PerModulusOutputBlock& block = output_blocks[blk];
        unpack_block_to_scalar(block.c0, blk, c0_out);

        if (has_ntt_s && ntt_s_out) {
            unpack_block_to_scalar(block.ntt_s, blk, ntt_s_out);
        }
        if (has_ntt_pte && ntt_pte_out) {
            unpack_block_to_scalar(block.ntt_pte, blk, ntt_pte_out);
        }
    }
}

struct ModulusParams {
    double scale;
    uint32_t mod_value;
    uint32_t const_ratio[2];
    uint8_t modulus_selector;
    bool save_ntt_s;
    bool save_ntt_pte;
};

static std::vector<event> run_pipeline(
    queue& q,
    buffer<PipelineInputBlock, 1>& input_buf,
    std::array<buffer<PerModulusOutputBlock, 1>, NUM_MODULI>& output_bufs,
    const std::array<ModulusParams, NUM_MODULI>& mod_params)
{
    try {
        std::vector<event> events;

        events.push_back(q.submit([&](handler& h) {
            ExitKernel<0> kernel(output_bufs[0], mod_params[0].save_ntt_s, mod_params[0].save_ntt_pte);
            kernel(h);
        }));
        events.push_back(q.submit([&](handler& h) {
            ExitKernel<1> kernel(output_bufs[1], mod_params[1].save_ntt_s, mod_params[1].save_ntt_pte);
            kernel(h);
        }));
        events.push_back(q.submit([&](handler& h) {
            ExitKernel<2> kernel(output_bufs[2], mod_params[2].save_ntt_s, mod_params[2].save_ntt_pte);
            kernel(h);
        }));

        events.push_back(q.submit([&](handler& h) {
            PolyMultNegAddKernel<0> kernel(mod_params[0].mod_value, mod_params[0].const_ratio);
            kernel(h);
        }));
        events.push_back(q.submit([&](handler& h) {
            PolyMultNegAddKernel<1> kernel(mod_params[1].mod_value, mod_params[1].const_ratio);
            kernel(h);
        }));
        events.push_back(q.submit([&](handler& h) {
            PolyMultNegAddKernel<2> kernel(mod_params[2].mod_value, mod_params[2].const_ratio);
            kernel(h);
        }));

        events.push_back(q.submit([&](handler& h) {
            NTTKernelA<0> kernel(mod_params[0].modulus_selector, mod_params[0].save_ntt_s);
            kernel(h);
        }));
        events.push_back(q.submit([&](handler& h) {
            NTTKernelA<1> kernel(mod_params[1].modulus_selector, mod_params[1].save_ntt_s);
            kernel(h);
        }));
        events.push_back(q.submit([&](handler& h) {
            NTTKernelA<2> kernel(mod_params[2].modulus_selector, mod_params[2].save_ntt_s);
            kernel(h);
        }));

        events.push_back(q.submit([&](handler& h) {
            NTTKernelB<0> kernel(mod_params[0].modulus_selector, mod_params[0].save_ntt_pte);
            kernel(h);
        }));
        events.push_back(q.submit([&](handler& h) {
            NTTKernelB<1> kernel(mod_params[1].modulus_selector, mod_params[1].save_ntt_pte);
            kernel(h);
        }));
        events.push_back(q.submit([&](handler& h) {
            NTTKernelB<2> kernel(mod_params[2].modulus_selector, mod_params[2].save_ntt_pte);
            kernel(h);
        }));

        events.push_back(q.submit([&](handler& h) {
            ScaleAndReduceKernel<0> kernel(mod_params[0].scale, mod_params[0].mod_value, mod_params[0].const_ratio);
            kernel(h);
        }));
        events.push_back(q.submit([&](handler& h) {
            ScaleAndReduceKernel<1> kernel(mod_params[1].scale, mod_params[1].mod_value, mod_params[1].const_ratio);
            kernel(h);
        }));
        events.push_back(q.submit([&](handler& h) {
            ScaleAndReduceKernel<2> kernel(mod_params[2].scale, mod_params[2].mod_value, mod_params[2].const_ratio);
            kernel(h);
        }));

        events.push_back(q.submit([&](handler& h) {
            IFFTKernel kernel;
            kernel(h);
        }));

        events.push_back(q.submit([&](handler& h) {
            EntryKernel kernel(input_buf);
            kernel(h);
        }));

        return events;

    } catch (std::exception const& e) {
        std::cerr << "[SYCL_encrypt] Pipeline exception: " << e.what() << std::endl;
        std::exit(1);
    }
    return {};
}

extern "C" void SYCL_encrypt(
    size_t n,
    const double* scales,
    const uint32_t* mod_values,
    const uint32_t* const_ratios,
    const complex_double* encoding_buffer,
    const int8_t* error_samples,
    const uint32_t* const* secret_keys,
    const uint32_t* const* uniform_polys,
    uint32_t** c0_outputs,
    uint32_t** c1_outputs,
    uint32_t** s_save,
    uint32_t** c1_save,
    uint32_t** ntt_pte_outputs)
{
    if (n % LANES != 0) {
        std::cerr << "[SYCL_encrypt] polynomial degree must be divisible by "
                  << LANES << " for lane normalization\n";
        std::exit(1);
    }

    if (n != POLY_N) {
        std::cerr << "[SYCL_encrypt] polynomial degree " << n
                  << " does not match compiled POLY_N=" << POLY_N << "\n";
        std::exit(1);
    }

    for (size_t p = 0; p < NUM_MODULI; ++p) {
        if (c1_save && c1_save[p]) {
            std::memcpy(c1_save[p], uniform_polys[p], n * sizeof(uint32_t));
        }
    }

    bool save_ntt_s_flags[NUM_MODULI];
    bool save_ntt_pte_flags[NUM_MODULI];
    for (size_t p = 0; p < NUM_MODULI; ++p) {
        save_ntt_s_flags[p] = (s_save && s_save[p] != nullptr);
        save_ntt_pte_flags[p] = (ntt_pte_outputs && ntt_pte_outputs[p] != nullptr);
    }

    size_t num_blocks = n / LANES;

    std::vector<PipelineInputBlock> input_blocks;
    pack_input(n, encoding_buffer, error_samples, secret_keys, uniform_polys, input_blocks);

    std::array<std::vector<PerModulusOutputBlock>, NUM_MODULI> output_blocks;
    std::array<ModulusParams, NUM_MODULI> mod_params;

    for (size_t p = 0; p < NUM_MODULI; ++p) {
        output_blocks[p].resize(num_blocks);
        mod_params[p].scale = scales[p];
        mod_params[p].mod_value = mod_values[p];
        mod_params[p].const_ratio[0] = const_ratios[p * 2];
        mod_params[p].const_ratio[1] = const_ratios[p * 2 + 1];
        mod_params[p].modulus_selector = get_modulus_selector(mod_values[p]);
        mod_params[p].save_ntt_s = save_ntt_s_flags[p];
        mod_params[p].save_ntt_pte = save_ntt_pte_flags[p];
    }

    buffer<PipelineInputBlock, 1> input_buf(input_blocks.data(), range(num_blocks));

    std::array<buffer<PerModulusOutputBlock, 1>, NUM_MODULI> output_bufs = {
        buffer<PerModulusOutputBlock, 1>(output_blocks[0].data(), range(num_blocks)),
        buffer<PerModulusOutputBlock, 1>(output_blocks[1].data(), range(num_blocks)),
        buffer<PerModulusOutputBlock, 1>(output_blocks[2].data(), range(num_blocks))
    };

#if FPGA_HARDWARE
    auto selector = ext::intel::fpga_selector_v;
#else
    auto selector = ext::intel::fpga_emulator_selector_v;
#endif
    queue q{selector, property::queue::enable_profiling()};

    auto events = run_pipeline(q, input_buf, output_bufs, mod_params);

    for (auto& ev : events) {
        ev.wait();
    }

    for (size_t p = 0; p < NUM_MODULI; ++p) {
        unpack_output(
            n, output_blocks[p],
            c0_outputs[p],
            s_save ? s_save[p] : nullptr,
            ntt_pte_outputs ? ntt_pte_outputs[p] : nullptr,
            mod_params[p].save_ntt_s,
            mod_params[p].save_ntt_pte);
    }

    for (size_t p = 0; p < NUM_MODULI; ++p) {
        if (c1_outputs && c1_outputs[p]) {
            std::memcpy(c1_outputs[p], uniform_polys[p], n * sizeof(uint32_t));
        }
    }
}
