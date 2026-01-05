#include "SYCL_ckks_sym.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "SYCL_common.h"
#include "SYCL_data_types.h"
#include "SYCL_pipes.h"
#include "SYCL_shared_entry.h"
#include "SYCL_per_mod_entry.h"
#include "SYCL_pipeline_exit.h"
#include "SYCL_ntt.h"
#include "SYCL_ifft.h"
#include "SYCL_scale_and_reduce.h"
#include "SYCL_poly_mult_neg_add.h"

#include <cstring>
#include <iostream>
#include <vector>
#include <array>

using namespace sycl;
using namespace sycl_ckks;

static void pack_shared_input(
    size_t n,
    const complex_double* encoding_buffer,
    const int8_t* error_samples,
    std::vector<SharedInputBlock>& shared_blocks)
{
    size_t num_blocks = n / LANES;
    shared_blocks.resize(num_blocks);

    for (size_t blk = 0; blk < num_blocks; ++blk) {
        SharedInputBlock& block = shared_blocks[blk];
        pack_encoding_to_block(encoding_buffer, blk, block.encoding);
        pack_error_to_block(error_samples, blk, block.error);
    }
}

static void pack_per_modulus_input(
    size_t n,
    const uint32_t* secret_key,
    const uint32_t* uniform_poly,
    std::vector<PerModulusInputBlock>& per_mod_blocks)
{
    size_t num_blocks = n / LANES;
    per_mod_blocks.resize(num_blocks);

    for (size_t blk = 0; blk < num_blocks; ++blk) {
        PerModulusInputBlock& block = per_mod_blocks[blk];
        pack_scalar_to_block(secret_key, blk, block.secret_key);
        pack_scalar_to_block(uniform_poly, blk, block.uniform_poly);
    }
}

static void unpack_per_modulus_output(
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

std::vector<event> run_pipeline(
    queue& q,
    size_t n,
    buffer<SharedInputBlock, 1>& shared_input_buf,
    std::array<buffer<PerModulusInputBlock, 1>, NUM_MODULI>& per_mod_input_bufs,
    std::array<buffer<PerModulusOutputBlock, 1>, NUM_MODULI>& per_mod_output_bufs,
    const std::array<ModulusParams, NUM_MODULI>& mod_params)
{
    (void)n;

    try {
        std::vector<event> events;

        events.push_back(q.submit([&](handler& h) {
            ExitKernel<0> kernel(per_mod_output_bufs[0], mod_params[0].save_ntt_s, mod_params[0].save_ntt_pte);
            kernel(h);
        }));
        events.push_back(q.submit([&](handler& h) {
            ExitKernel<1> kernel(per_mod_output_bufs[1], mod_params[1].save_ntt_s, mod_params[1].save_ntt_pte);
            kernel(h);
        }));
        events.push_back(q.submit([&](handler& h) {
            ExitKernel<2> kernel(per_mod_output_bufs[2], mod_params[2].save_ntt_s, mod_params[2].save_ntt_pte);
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
            PerModulusEntryKernel<0> kernel(per_mod_input_bufs[0]);
            kernel(h);
        }));
        events.push_back(q.submit([&](handler& h) {
            PerModulusEntryKernel<1> kernel(per_mod_input_bufs[1]);
            kernel(h);
        }));
        events.push_back(q.submit([&](handler& h) {
            PerModulusEntryKernel<2> kernel(per_mod_input_bufs[2]);
            kernel(h);
        }));

        events.push_back(q.submit([&](handler& h) {
            SharedEntryKernel kernel(shared_input_buf);
            kernel(h);
        }));

        return events;

    } catch (std::exception const& e) {
        std::cout << "[run_pipeline] EXCEPTION CAUGHT!" << std::endl;
        std::cerr << "Caught exception in pipeline: " << e.what() << std::endl;
        std::exit(1);
    }
    return {};
}

extern "C" void SYCL_encrypt_all_moduli(
    size_t n,
    size_t logn,
    const complex_double* encoding_buffer,
    const int8_t* error_samples,
    const double* scales,
    const uint32_t* mod_values,
    const uint32_t* const_ratios,
    const uint32_t* const* secret_keys,
    const uint32_t* const* uniform_polys,
    uint32_t** c0_outputs,
    uint32_t** ntt_s_outputs,
    uint32_t** ntt_pte_outputs,
    const bool* save_ntt_s_flags,
    const bool* save_ntt_pte_flags)
{
    (void)logn;

    if (n % LANES != 0) {
        std::cerr << "[SYCL_encrypt_all_moduli] polynomial degree must be divisible by "
                  << LANES << " for lane normalization\n";
        std::exit(1);
    }

    if (n != POLY_N) {
        std::cerr << "[SYCL_encrypt_all_moduli] polynomial degree " << n
                  << " does not match compiled POLY_N=" << POLY_N << "\n";
        std::exit(1);
    }

    size_t num_blocks = n / LANES;

    std::vector<SharedInputBlock> shared_blocks;
    pack_shared_input(n, encoding_buffer, error_samples, shared_blocks);

    std::array<std::vector<PerModulusInputBlock>, NUM_MODULI> per_mod_input_blocks;
    std::array<std::vector<PerModulusOutputBlock>, NUM_MODULI> per_mod_output_blocks;
    std::array<ModulusParams, NUM_MODULI> mod_params;

    for (size_t p = 0; p < NUM_MODULI; ++p) {
        pack_per_modulus_input(n, secret_keys[p], uniform_polys[p], per_mod_input_blocks[p]);
        per_mod_output_blocks[p].resize(num_blocks);

        mod_params[p].scale = scales[p];
        mod_params[p].mod_value = mod_values[p];
        mod_params[p].const_ratio[0] = const_ratios[p * 2];
        mod_params[p].const_ratio[1] = const_ratios[p * 2 + 1];
        mod_params[p].modulus_selector = get_modulus_selector(mod_values[p]);
        mod_params[p].save_ntt_s = save_ntt_s_flags ? save_ntt_s_flags[p] : false;
        mod_params[p].save_ntt_pte = save_ntt_pte_flags ? save_ntt_pte_flags[p] : false;
    }

    buffer<SharedInputBlock, 1> shared_input_buf(shared_blocks.data(), range(num_blocks));

    std::array<buffer<PerModulusInputBlock, 1>, NUM_MODULI> per_mod_input_bufs = {
        buffer<PerModulusInputBlock, 1>(per_mod_input_blocks[0].data(), range(num_blocks)),
        buffer<PerModulusInputBlock, 1>(per_mod_input_blocks[1].data(), range(num_blocks)),
        buffer<PerModulusInputBlock, 1>(per_mod_input_blocks[2].data(), range(num_blocks))
    };

    std::array<buffer<PerModulusOutputBlock, 1>, NUM_MODULI> per_mod_output_bufs = {
        buffer<PerModulusOutputBlock, 1>(per_mod_output_blocks[0].data(), range(num_blocks)),
        buffer<PerModulusOutputBlock, 1>(per_mod_output_blocks[1].data(), range(num_blocks)),
        buffer<PerModulusOutputBlock, 1>(per_mod_output_blocks[2].data(), range(num_blocks))
    };

#if FPGA_HARDWARE
    auto selector = ext::intel::fpga_selector_v;
#else
    auto selector = ext::intel::fpga_emulator_selector_v;
#endif
    queue q{selector, property::queue::enable_profiling()};

    auto events = run_pipeline(
        q, n, shared_input_buf, per_mod_input_bufs, per_mod_output_bufs, mod_params);

    for (auto& ev : events) {
        ev.wait();
    }

    for (size_t p = 0; p < NUM_MODULI; ++p) {
        unpack_per_modulus_output(
            n, per_mod_output_blocks[p],
            c0_outputs[p],
            ntt_s_outputs ? ntt_s_outputs[p] : nullptr,
            ntt_pte_outputs ? ntt_pte_outputs[p] : nullptr,
            mod_params[p].save_ntt_s,
            mod_params[p].save_ntt_pte);
    }
}

extern "C" void SYCL_encrypt(
    size_t n,
    size_t logn,
    const double* scales,
    const uint32_t* mod_values,
    const uint32_t* const_ratios,
    complex_double* encoding_buffer,
    int8_t* error_samples,
    uint32_t* const* expanded_s,
    uint32_t* const* uniform_polys,
    uint32_t** c0_outputs,
    uint32_t** c1_outputs,
    uint32_t** s_save,
    uint32_t** c1_save,
    uint32_t** ntt_pte_outputs)
{
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

    SYCL_encrypt_all_moduli(
        n, logn, encoding_buffer, error_samples,
        scales, mod_values, const_ratios,
        (const uint32_t* const*)expanded_s, (const uint32_t* const*)uniform_polys,
        c0_outputs, s_save, ntt_pte_outputs,
        save_ntt_s_flags, save_ntt_pte_flags);

    for (size_t p = 0; p < NUM_MODULI; ++p) {
        if (c1_outputs && c1_outputs[p]) {
            std::memcpy(c1_outputs[p], uniform_polys[p], n * sizeof(uint32_t));
        }
    }
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
    (void)pt_with_error;

    double scales[NUM_MODULI] = {scale, scale, scale};
    uint32_t mod_vals[NUM_MODULI] = {mod_value, mod_value, mod_value};
    uint32_t cr[NUM_MODULI * 2] = {
        const_ratio[0], const_ratio[1],
        const_ratio[0], const_ratio[1],
        const_ratio[0], const_ratio[1]
    };
    uint32_t* exp_s[NUM_MODULI] = {expanded_s, expanded_s, expanded_s};
    uint32_t* uni_p[NUM_MODULI] = {uniform_poly, uniform_poly, uniform_poly};
    uint32_t* c0_out[NUM_MODULI] = {c0_s, c0_s, c0_s};
    uint32_t* c1_out[NUM_MODULI] = {c1, c1, c1};
    uint32_t* s_sv[NUM_MODULI] = {s_save, nullptr, nullptr};
    uint32_t* c1_sv[NUM_MODULI] = {c1_save, nullptr, nullptr};
    uint32_t* pte_out[NUM_MODULI] = {ntt_pte, nullptr, nullptr};

    SYCL_encrypt(
        n, logn, scales, mod_vals, cr,
        encoding_buffer, error_samples,
        exp_s, uni_p, c0_out, c1_out, s_sv, c1_sv, pte_out);
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
    (void)pipeline_index;
    SYCL_combined_encrypt(n, logn, scale, mod_value, const_ratio, encoding_buffer, expanded_s,
                          uniform_poly, error_samples, pt_with_error, ntt_pte, c0_s, c1,
                          s_save, c1_save);
}
