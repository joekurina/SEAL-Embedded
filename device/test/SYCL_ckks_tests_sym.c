/**
@file SYCL_ckks_test_sym.c
*/

#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h> // Needed for calloc, free
#include <string.h> // Needed for memset

#include "ckks_common.h"
#include "ckks_sym.h"
#include "ckks_tests_common.h"
#include "defines.h"
#include "fft.h" // Might need complex type definition if not included elsewhere
#include "fileops.h"
#include "ntt.h"
#include "polymodarith.h"
#include "polymodmult.h"
#include "sample.h"
#include "test_common.h"
#include "util_print.h"

typedef struct PipelineTask
{
    int pipeline_index;
    Parms parms;
    const flpt *values;
    size_t values_len;
    SE_PRNG shareable_prng;
    SE_PRNG error_prng;
    ZZ *s_small;
    ZZ *ntt_pte;
    ZZ *ntt_roots;
    ZZ *c0;
    ZZ *c1;
    ZZ *s_save;
    ZZ *c1_save;
    complex_double *encoding_buffer;
    uint16_t *index_map;
} PipelineTask;

static void prepare_parms_for_prime(const Parms *base, Parms *dst, size_t prime_idx, ZZ *s_small)
{
    *dst = *base;
    ckks_reset_primes(dst);
    for (size_t step = 0; step < prime_idx; ++step) {
        ckks_next_prime_sym(dst, s_small);
    }
}

static void *run_pipeline_thread(void *arg)
{
    PipelineTask *task = (PipelineTask *)arg;
    ckks_combined_encode_encrypt_sym_pipeline(
        task->pipeline_index,
        &task->parms,
        task->values,
        task->values_len,
        &task->shareable_prng,
        &task->error_prng,
        task->s_small,
        task->ntt_pte,
        task->ntt_roots,
        task->c0,
        task->c1,
        task->s_save,
        task->c1_save,
        task->encoding_buffer,
        task->index_map);

    return NULL;
}

// Newer Combined ENCODE + ENCRYPT test using individual allocations
void SYCL_test_ckks_sym_base(size_t n, size_t nprimes, bool test_message)
{
    const size_t pipeline_count = 3;
    Parms parms;
    parms.sample_s      = false; 
    parms.is_asymmetric = false;
    parms.small_s       = true;
    bool encode_only    = false; 

    if (!parms.sample_s) se_assert(parms.small_s);

    // Pointers for individually allocated buffers (per pipeline)
    complex_double *conj_vals[3] = {0}; // Used as encoding buffer
    ZZ *c0[3]                    = {0};
    ZZ *c1[3]                    = {0};
    uint16_t *index_map          = NULL; // Shared
    ZZ *ntt_roots[3]             = {0};
    ZZ *ntt_pte[3]               = {0};
    ZZ *s[3]                     = {0}; // Secret key buffer (small form copy per pipeline)
    flpt *v                      = NULL; // Message buffer (shared)
    size_t vlen                 = n / 2;

    // -- Additional pointers required for testing (allocated separately as before)
    ZZ *s_test_save[3]  = {0};
    ZZ *c1_test_save[3] = {0};
    ZZ *temp_test_mem = NULL;

    SE_PRNG prng;
    SE_PRNG shareable_prng;
    SE_PRNG pipeline_prng[3];
    SE_PRNG pipeline_shareable_prng[3];

    // --- Allocate Buffers Individually ---
    bool allocation_success = true;

    for (size_t i = 0; i < pipeline_count; ++i) {
        conj_vals[i] = (complex_double *)calloc(n, sizeof(complex_double));
        if (!conj_vals[i]) { allocation_success = false; goto cleanup; }

        c0[i] = (ZZ *)calloc(n, sizeof(ZZ));
        if (!c0[i]) { allocation_success = false; goto cleanup; }

        c1[i] = (ZZ *)calloc(n, sizeof(ZZ));
        if (!c1[i]) { allocation_success = false; goto cleanup; }

        ntt_pte[i] = (ZZ *)calloc(n, sizeof(ZZ));
        if (!ntt_pte[i]) { allocation_success = false; goto cleanup; }
    }

    // Determine secret key buffer size (assuming ZZ type for buffer)
    size_t s_buf_size = parms.small_s ? (n / 16 + (n % 16 != 0)) : n; // Rough size for small_s, use n for expanded
    // Allocate slightly larger buffer for s if small form, or full ZZ * n if expanded needed
    for (size_t i = 0; i < pipeline_count; ++i) {
        s[i] = (ZZ *)calloc(s_buf_size, sizeof(ZZ)); // Adjust size accurately if needed
        if (!s[i]) { allocation_success = false; goto cleanup; }
    }

    if (test_message) {
        v = (flpt *)calloc(vlen, sizeof(flpt));
        if (!v) { allocation_success = false; goto cleanup; }
    }

    // Allocate conditional buffers based on defines (mirror logic from ckks_set_ptrs_sym)
#if defined(SE_INDEX_MAP_PERSIST) || defined(SE_INDEX_MAP_LOAD) || \
    defined(SE_INDEX_MAP_LOAD_PERSIST) || \
    defined(SE_INDEX_MAP_LOAD_PERSIST_SYM_LOAD_ASYM)
    index_map = (uint16_t *)calloc(n, sizeof(uint16_t));
    if (!index_map) { allocation_success = false; goto cleanup; }
#endif

#if defined(SE_NTT_ONE_SHOT) || defined(SE_NTT_REG)
    size_t ntt_roots_size = n;
#elif defined(SE_NTT_FAST)
    size_t ntt_roots_size = 2 * n;
#else
    size_t ntt_roots_size = 0; // NTT OTF
#endif
    if (ntt_roots_size > 0) {
        for (size_t i = 0; i < pipeline_count; ++i) {
            ntt_roots[i] = (ZZ *)calloc(ntt_roots_size, sizeof(ZZ));
            if (!ntt_roots[i]) { allocation_success = false; goto cleanup; }
        }
    }

    // Allocate test-specific buffers
    for (size_t i = 0; i < pipeline_count; ++i) {
        s_test_save[i] = calloc(n, sizeof(ZZ));
        if (!s_test_save[i]) { allocation_success = false; goto cleanup; }
        c1_test_save[i] = calloc(n, sizeof(ZZ));
        if (!c1_test_save[i]) { allocation_success = false; goto cleanup; }
    }
    temp_test_mem = calloc(4 * n, sizeof(ZZ)); // Used by check_decode_decrypt_inpl
    if (!temp_test_mem) { allocation_success = false; goto cleanup; }

    // --- Setup Parameters ---
    // Ensure index_map is allocated above if needed by ckks_setup config.
    ckks_setup(n, nprimes, index_map, &parms);
    print_test_banner("Symmetric Encryption", &parms);

    // Setup secret key 's' (populates the first buffer, then copy to others)
    ckks_setup_s(&parms, NULL, &prng, s[0]);
    for (size_t i = 1; i < pipeline_count; ++i) {
        memcpy(s[i], s[0], s_buf_size * sizeof(ZZ));
    }
    // size_t s_size = parms.small_s ? n / 16 : n; 
    // // This was from original, use s_buf_size
    if (encode_only) {
        for (size_t i = 0; i < pipeline_count; ++i) {
            clear(s[i], s_buf_size);
        }
    }

    // --- Run Tests ---
    for (size_t testnum = 0; testnum < 9; testnum++)
    {
        printf("-------------------- Test %zu -----------------------\n", testnum);
        ckks_reset_primes(&parms);

        // -- Set test values in 'v' buffer
        if (test_message)
        {
            set_encode_encrypt_test(testnum, vlen, v);
            print_poly_flpt("v        ", v, vlen);
        }
        else if (v) // Check if v was allocated (i.e., test_message was true at allocation time)
        {
             // If not testing with a message, use zeros (only if v exists)
             memset(v, 0, vlen * sizeof(flpt));
        }


        // -- Initialize PRNGs for this test run
        prng_randomize_reset(&shareable_prng, NULL);
        prng_randomize_reset(&prng, NULL);

        bool test_failed = false;
        size_t prime_idx = 0;
        while (prime_idx < parms.nprimes)
        {
            size_t batch = (parms.nprimes - prime_idx > pipeline_count) ? pipeline_count : (parms.nprimes - prime_idx);

            pthread_t threads[3];
            PipelineTask tasks[3];

            // Prepare and launch batch in parallel
            for (size_t p = 0; p < batch; ++p)
            {
                size_t pipeline_id = p; // 0,1,2 for this batch
                tasks[p].pipeline_index = (int)pipeline_id;
                prepare_parms_for_prime(&parms, &tasks[p].parms, prime_idx + p, s[pipeline_id]);

                prng_randomize_reset(&pipeline_shareable_prng[p], NULL);
                prng_randomize_reset(&pipeline_prng[p], NULL);

                tasks[p].values      = test_message ? v : NULL;
                tasks[p].values_len  = vlen;
                tasks[p].shareable_prng = pipeline_shareable_prng[p];
                tasks[p].error_prng     = pipeline_prng[p];
                tasks[p].s_small     = s[pipeline_id];
                tasks[p].ntt_pte     = ntt_pte[pipeline_id];
                tasks[p].ntt_roots   = ntt_roots[pipeline_id];
                tasks[p].c0          = c0[pipeline_id];
                tasks[p].c1          = c1[pipeline_id];
                tasks[p].s_save      = s_test_save[pipeline_id];
                tasks[p].c1_save     = c1_test_save[pipeline_id];
                tasks[p].encoding_buffer = conj_vals[pipeline_id];
                tasks[p].index_map   = index_map;

                int rc = pthread_create(&threads[p], NULL, run_pipeline_thread, &tasks[p]);
                se_assert(rc == 0);
            }

            // Join batch
            for (size_t p = 0; p < batch; ++p)
            {
                pthread_join(threads[p], NULL);
            }

            // Decode/decrypt sequentially for each modulus in the batch
            for (size_t p = 0; p < batch; ++p)
            {
                bool s_test_save_small = false;
                bool success = check_decode_decrypt_inpl(
                    tasks[p].c0,
                    tasks[p].c1_save,
                    v,
                    vlen,
                    tasks[p].s_save,
                    s_test_save_small,
                    tasks[p].ntt_pte,
                    index_map,
                    &tasks[p].parms,
                    temp_test_mem);
                if (!success) {
                    test_failed = true;
                } else {
                    printf("OK!\n");
                }
            }

            prime_idx += batch;
        }

        if (test_failed) {
            printf("TEST FAILED\n");
        }

        // -- Can exit now if rlwe testing only
        if (!test_message) break;
    }

cleanup:
    // --- Cleanup ---
    if (!allocation_success) {
        printf("Aborting test due to memory allocation failure.\n");
    }

    for (size_t i = 0; i < pipeline_count; ++i) {
        free(conj_vals[i]);
        free(c0[i]);
        free(c1[i]);
        free(ntt_pte[i]);
        free(s[i]);
        free(s_test_save[i]);
        free(c1_test_save[i]);
        free(ntt_roots[i]);
    }
    free(v); // Free v if it was allocated
    free(index_map); // Free index_map if it was allocated
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