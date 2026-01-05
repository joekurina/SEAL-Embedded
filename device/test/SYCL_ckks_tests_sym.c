/**
@file SYCL_ckks_test_sym.c
*/

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckks_common.h"
#include "ckks_sym.h"
#include "ckks_tests_common.h"
#include "defines.h"
#include "fft.h"
#include "fileops.h"
#include "ntt.h"
#include "polymodarith.h"
#include "polymodmult.h"
#include "sample.h"
#include "test_common.h"
#include "util_print.h"
#include "SYCL_ckks_sym.h"

static void prepare_parms_for_prime(const Parms *base, Parms *dst, size_t prime_idx, ZZ *s_small)
{
    *dst = *base;
    ckks_reset_primes(dst);
    for (size_t step = 0; step < prime_idx; ++step) {
        ckks_next_prime_sym(dst, s_small);
    }
}

void SYCL_test_ckks_sym_base(size_t n, size_t nprimes, bool test_message)
{
    const size_t moduli_count = SYCL_NUM_MODULI;
    Parms parms;
    parms.sample_s      = false; 
    parms.is_asymmetric = false;
    parms.small_s       = true;
    bool encode_only    = false; 

    if (!parms.sample_s) se_assert(parms.small_s);

    complex_double *encoding_buffer = NULL;
    int8_t *error_samples = NULL;
    ZZ *c0[SYCL_NUM_MODULI] = {0};
    ZZ *c1[SYCL_NUM_MODULI] = {0};
    ZZ *expanded_s[SYCL_NUM_MODULI] = {0};
    ZZ *uniform_poly[SYCL_NUM_MODULI] = {0};
    uint16_t *index_map = NULL;
    ZZ *ntt_pte[SYCL_NUM_MODULI] = {0};
    ZZ *s_small = NULL;
    flpt *v = NULL;
    size_t vlen = n / 2;

    ZZ *s_test_save[SYCL_NUM_MODULI] = {0};
    ZZ *c1_test_save[SYCL_NUM_MODULI] = {0};
    ZZ *temp_test_mem = NULL;

    SE_PRNG prng;
    SE_PRNG shareable_prng;
    SE_PRNG error_prng;

    bool allocation_success = true;

    encoding_buffer = (complex_double *)calloc(n, sizeof(complex_double));
    if (!encoding_buffer) { allocation_success = false; goto cleanup; }

    error_samples = (int8_t *)calloc(n, sizeof(int8_t));
    if (!error_samples) { allocation_success = false; goto cleanup; }

    for (size_t i = 0; i < moduli_count; ++i) {
        c0[i] = (ZZ *)calloc(n, sizeof(ZZ));
        if (!c0[i]) { allocation_success = false; goto cleanup; }

        c1[i] = (ZZ *)calloc(n, sizeof(ZZ));
        if (!c1[i]) { allocation_success = false; goto cleanup; }

        expanded_s[i] = (ZZ *)calloc(n, sizeof(ZZ));
        if (!expanded_s[i]) { allocation_success = false; goto cleanup; }

        uniform_poly[i] = (ZZ *)calloc(n, sizeof(ZZ));
        if (!uniform_poly[i]) { allocation_success = false; goto cleanup; }

        ntt_pte[i] = (ZZ *)calloc(n, sizeof(ZZ));
        if (!ntt_pte[i]) { allocation_success = false; goto cleanup; }

        s_test_save[i] = (ZZ *)calloc(n, sizeof(ZZ));
        if (!s_test_save[i]) { allocation_success = false; goto cleanup; }

        c1_test_save[i] = (ZZ *)calloc(n, sizeof(ZZ));
        if (!c1_test_save[i]) { allocation_success = false; goto cleanup; }
    }

    size_t s_buf_size = parms.small_s ? (n / 16 + (n % 16 != 0)) : n;
    s_small = (ZZ *)calloc(s_buf_size, sizeof(ZZ));
    if (!s_small) { allocation_success = false; goto cleanup; }

    if (test_message) {
        v = (flpt *)calloc(vlen, sizeof(flpt));
        if (!v) { allocation_success = false; goto cleanup; }
    }

#if defined(SE_INDEX_MAP_PERSIST) || defined(SE_INDEX_MAP_LOAD) || \
    defined(SE_INDEX_MAP_LOAD_PERSIST) || defined(SE_INDEX_MAP_LOAD_PERSIST_SYM_LOAD_ASYM)
    index_map = (uint16_t *)calloc(n, sizeof(uint16_t));
    if (!index_map) { allocation_success = false; goto cleanup; }
#endif

    temp_test_mem = (ZZ *)calloc(4 * n, sizeof(ZZ));
    if (!temp_test_mem) { allocation_success = false; goto cleanup; }

    ckks_setup(n, nprimes, index_map, &parms);
    print_test_banner("Symmetric Encryption", &parms);

    ckks_setup_s(&parms, NULL, &prng, s_small);
    if (encode_only) clear(s_small, s_buf_size);

    for (size_t testnum = 0; testnum < 9; testnum++)
    {
        printf("-------------------- Test %zu -----------------------\n", testnum);
        ckks_reset_primes(&parms);

        if (test_message) {
            set_encode_encrypt_test(testnum, vlen, v);
            print_poly_flpt("v        ", v, vlen);
        } else if (v) {
            memset(v, 0, vlen * sizeof(flpt));
        }

        prng_randomize_reset(&shareable_prng, NULL);
        prng_randomize_reset(&prng, NULL);
        prng_randomize_reset(&error_prng, NULL);

        bool test_failed = false;

        se_assert(parms.nprimes == moduli_count);

        Parms mod_parms[SYCL_NUM_MODULI];
        double scales[SYCL_NUM_MODULI];
        uint32_t mod_values[SYCL_NUM_MODULI];
        uint32_t const_ratios[SYCL_NUM_MODULI * 2];

        for (size_t p = 0; p < moduli_count; ++p) {
            prepare_parms_for_prime(&parms, &mod_parms[p], p, s_small);
            scales[p] = mod_parms[p].scale;
            mod_values[p] = (uint32_t)mod_parms[p].curr_modulus->value;
            const_ratios[p * 2] = (uint32_t)mod_parms[p].curr_modulus->const_ratio[0];
            const_ratios[p * 2 + 1] = (uint32_t)mod_parms[p].curr_modulus->const_ratio[1];
        }

        memset(encoding_buffer, 0, n * sizeof(complex_double));
        if (test_message && v != NULL && index_map != NULL) {
            size_t slot_count = n / 2;
            for (size_t i = 0; i < vlen; i++) {
                uint16_t index1_rev = index_map[i];
                uint16_t index2_rev = index_map[i + slot_count];
                double val_real = (double)(v[i]);
                ((double*)encoding_buffer)[2*index1_rev] = val_real;
                ((double*)encoding_buffer)[2*index1_rev+1] = 0.0;
                ((double*)encoding_buffer)[2*index2_rev] = val_real;
                ((double*)encoding_buffer)[2*index2_rev+1] = 0.0;
            }
        }

        for (size_t i = 0; i < n; i++) {
            int8_t temp_buffer[1] = {0};
            sample_poly_cbd_generic(1, &error_prng, temp_buffer);
            error_samples[i] = temp_buffer[0];
        }

        for (size_t p = 0; p < moduli_count; ++p) {
            expand_poly_ternary(s_small, &mod_parms[p], expanded_s[p]);
            sample_poly_uniform(&mod_parms[p], &shareable_prng, uniform_poly[p]);
            prng_randomize_reset(&shareable_prng, NULL);
        }

        SYCL_encrypt(
            n, parms.logn, scales, mod_values, const_ratios,
            encoding_buffer, error_samples,
            (uint32_t* const*)expanded_s, (uint32_t* const*)uniform_poly,
            (uint32_t**)c0, (uint32_t**)c1,
            (uint32_t**)s_test_save, (uint32_t**)c1_test_save,
            (uint32_t**)ntt_pte);

        for (size_t p = 0; p < moduli_count; ++p)
        {
            bool s_test_save_small = false;
            bool success = check_decode_decrypt_inpl(
                c0[p], c1_test_save[p], v, vlen,
                s_test_save[p], s_test_save_small,
                ntt_pte[p], index_map, &mod_parms[p], temp_test_mem);
            if (!success) {
                test_failed = true;
            }
        }

        if (test_failed) {
            printf("TEST FAILED\n");
        }

        if (!test_message) break;
    }

cleanup:
    if (!allocation_success) {
        printf("Aborting test due to memory allocation failure.\n");
    }

    free(encoding_buffer);
    free(error_samples);
    for (size_t i = 0; i < moduli_count; ++i) {
        free(c0[i]);
        free(c1[i]);
        free(expanded_s[i]);
        free(uniform_poly[i]);
        free(ntt_pte[i]);
        free(s_test_save[i]);
        free(c1_test_save[i]);
    }
    free(s_small);
    free(v);
    free(index_map);
    free(temp_test_mem);

    delete_parameters(&parms);
}

/**
Full encode + symmetric encrypt test
@param[in] n        Polynomial ring degree
@param[in] nprimes  # of modulus primes
*/
void SYCL_test_ckks_encode_encrypt_sym(size_t n, size_t nprimes)
{
    printf("Beginning tests for ckks encode + symmetric encrypt (SYCL Test)...\n");
    bool test_message = 1;
    SYCL_test_ckks_sym_base(n, nprimes, test_message);
}

/**
Symmetric rlwe test only (message is the all-zeros vector)
@param[in] n        Polynomial ring degree
@param[in] nprimes  # of modulus primes
*/
void SYCL_test_enc_zero_sym(size_t n, size_t nprimes)
{
    printf("Beginning tests for rlwe symmetric encryption of 0 (SYCL Test)...\n");
    bool test_message = 0;
    SYCL_test_ckks_sym_base(n, nprimes, test_message);
}