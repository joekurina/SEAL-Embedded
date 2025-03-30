/**
@file ckks_test_sym.c
*/

#include <math.h>
#include <stdbool.h>
#include <stdio.h>

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

/**
Symmetric test core. If debugging is enabled, throws an error if a test fails.

@param[in] n             Polynomial ring degree (ignored if SE_USE_MALLOC is defined)
@param[in] nprimes       # of modulus primes    (ignored if SE_USE_MALLOC is defined)
@param[in] test_message  If 0, sets the message 0
*/
void test_ckks_sym_base(size_t n, size_t nprimes, bool test_message)
{
    Parms parms;
    parms.sample_s      = false;
    parms.is_asymmetric = false;
    parms.small_s       = true;
    bool encode_only    = false;  // this only works if s is not not persistent
    // -- Make sure we didn't set this accidentally
    if (!parms.sample_s) se_assert(parms.small_s);
    print_ckks_mempool_size(n, true);
    ZZ *mempool = ckks_mempool_setup_sym(n);
    // -- Get pointers
    SE_PTRS se_ptrs_local;
    ckks_set_ptrs_sym(n, mempool, &se_ptrs_local);
    double complex *conj_vals  = se_ptrs_local.conj_vals;
    int64_t *conj_vals_int     = se_ptrs_local.conj_vals_int_ptr;
    double complex *ifft_roots = se_ptrs_local.ifft_roots;
    ZZ *c0                     = se_ptrs_local.c0_ptr;
    ZZ *c1                     = se_ptrs_local.c1_ptr;
    uint16_t *index_map        = se_ptrs_local.index_map_ptr;
    ZZ *ntt_roots              = se_ptrs_local.ntt_roots_ptr;
    ZZ *ntt_pte                = se_ptrs_local.ntt_pte_ptr;
    ZZ *s                      = se_ptrs_local.ternary;
    flpt *v                    = se_ptrs_local.values;
    size_t vlen                = n / 2;
    if (!test_message)
    {
        se_assert(v);
        if (v) memset(v, 0, vlen * sizeof(flpt));
        // v = 0;
    }
    // -- Additional pointers required for testing.
    ZZ *s_test_save   = calloc(n, sizeof(ZZ));  // ntt(expanded(s)) or expanded(s)
    ZZ *c1_test_save  = calloc(n, sizeof(ZZ));
    ZZ *temp_test_mem = calloc(4 * n, sizeof(ZZ));
    SE_PRNG prng;
    SE_PRNG shareable_prng;
    // -- Set up parameters and index_map if applicable
    ckks_setup(n, nprimes, index_map, &parms);
    print_test_banner("Symmetric Encryption", &parms);
    // -- If s is allocated space ahead of time, can load ahead of time too
    // -- If we are testing and sample s is set, this will also sample s
    ckks_setup_s(&parms, NULL, &prng, s);
    size_t s_size = parms.small_s ? n / 16 : n;
    if (encode_only) clear(s, s_size);
    for (size_t testnum = 0; testnum < 9; testnum++)
    {
        printf("-------------------- Test %zu -----------------------\n", testnum);
        ckks_reset_primes(&parms);
        // -- Set test values
        if (test_message)
        {
            set_encode_encrypt_test(testnum, vlen, v);
            print_poly_flpt("v        ", v, vlen);
        }
        // -- Begin encode-encrypt sequence
        // -- First, calculate m + e (not fully reduced, not in ntt form)
        if (test_message)
        {
            bool ret = ckks_encode_base(&parms, v, vlen, index_map, ifft_roots, conj_vals);
            se_assert(ret);
        }
        else
            memset(conj_vals_int, 0, n * sizeof(conj_vals_int[0]));
        se_assert(v);
        // -- Sample error e. While sampling e, add it in place to the base message.
        if (!encode_only)
        { 
            ckks_sym_init(&parms, NULL, NULL, &shareable_prng, &prng, conj_vals_int); 
        }
        for (size_t i = 0; i < parms.nprimes; i++)
        {
            print_zz("\n ***** Modulus", parms.curr_modulus->value);

            // -- Per prime Encode + Encrypt
            ckks_encode_encrypt_sym(&parms, conj_vals_int, NULL, &shareable_prng, s, ntt_pte,
                                    ntt_roots, c0, c1, s_test_save, c1_test_save);
            // -- Check that decrypt gives back the pt+err and decode gives back v.
            // -- Note: This will only decode if values is non-zero. Otherwise, will
            //    just decrypt.
            // -- Note: sizeof(max(ntt_roots, ifft_roots)) must be passed as temp memory
            //    to undo ifft
            bool s_test_save_small = false;
            check_decode_decrypt_inpl(c0, c1_test_save, v, vlen, s_test_save, s_test_save_small,
                                      ntt_pte, index_map, &parms, temp_test_mem);
            // -- Done checking this prime. Now try next prime if requested
            bool ret = ckks_next_prime_sym(&parms, s);
            se_assert(ret || (!ret && i + 1 == parms.nprimes));
        }
        // -- Can exit now if rlwe testing only
        if (!test_message) break;
    }

    //clang-format off
    if (mempool)
    {
        free(mempool);
        mempool = 0;
    }
    if (s_test_save)
    {
        free(s_test_save);
        s_test_save = 0;
    }
    if (c1_test_save)
    {
        free(c1_test_save);
        c1_test_save = 0;
    }
    if (temp_test_mem)
    {
        free(temp_test_mem);
        temp_test_mem = 0;
    }
    delete_parameters(&parms);
}

// Newer Combined ENCODE + ENCRYPT test
void test_ckks_combined_sym_base(size_t n, size_t nprimes, bool test_message)
{
    Parms parms;
    parms.sample_s      = false;
    parms.is_asymmetric = false;
    parms.small_s       = true;
    bool encode_only    = false;  // this only works if s is not not persistent
    
    // -- Make sure we didn't set this accidentally
    if (!parms.sample_s) se_assert(parms.small_s);
    
    print_ckks_mempool_size(n, true);
    ZZ *mempool = ckks_mempool_setup_sym(n);
    
    // -- Get pointers
    SE_PTRS se_ptrs_local;
    
    ckks_set_ptrs_sym(n, mempool, &se_ptrs_local);
    double complex *conj_vals  = se_ptrs_local.conj_vals;
    int64_t *conj_vals_int     = se_ptrs_local.conj_vals_int_ptr;
    double complex *ifft_roots = se_ptrs_local.ifft_roots;
    ZZ *c0                     = se_ptrs_local.c0_ptr;
    ZZ *c1                     = se_ptrs_local.c1_ptr;
    uint16_t *index_map        = se_ptrs_local.index_map_ptr;
    ZZ *ntt_roots              = se_ptrs_local.ntt_roots_ptr;
    ZZ *ntt_pte                = se_ptrs_local.ntt_pte_ptr;
    ZZ *s                      = se_ptrs_local.ternary;
    flpt *v                    = se_ptrs_local.values;
    size_t vlen                = n / 2;
    
    if (!test_message)
    {
        se_assert(v);
        if (v) memset(v, 0, vlen * sizeof(flpt));
    }
    
    // -- Additional pointers required for testing.
    ZZ *s_test_save   = calloc(n, sizeof(ZZ));  // ntt(expanded(s)) or expanded(s)
    ZZ *c1_test_save  = calloc(n, sizeof(ZZ));
    ZZ *temp_test_mem = calloc(4 * n, sizeof(ZZ));
    
    SE_PRNG prng;
    SE_PRNG shareable_prng;
    
    // -- Set up parameters and index_map if applicable
    ckks_setup(n, nprimes, index_map, &parms);
    print_test_banner("Symmetric Encryption", &parms);
    
    // -- If s is allocated space ahead of time, can load ahead of time too
    // -- If we are testing and sample s is set, this will also sample s
    ckks_setup_s(&parms, NULL, &prng, s);
    size_t s_size = parms.small_s ? n / 16 : n;
    if (encode_only) clear(s, s_size);
    
    for (size_t testnum = 0; testnum < 9; testnum++)
    {
        printf("-------------------- Test %zu -----------------------\n", testnum);
        ckks_reset_primes(&parms);
        
        // -- Set test values
        if (test_message)
        {
            set_encode_encrypt_test(testnum, vlen, v);
            print_poly_flpt("v        ", v, vlen);
        }
        else
        {
            // If not testing with a message, use zeros
            memset(v, 0, vlen * sizeof(flpt));
        }
        
        // -- Initialize PRNGs for this test
        prng_randomize_reset(&shareable_prng, NULL);
        prng_randomize_reset(&prng, NULL);
        
        for (size_t i = 0; i < parms.nprimes; i++)
        {
            print_zz("\n ***** Modulus", parms.curr_modulus->value);

            // -- Encode + Encrypt in one step
            // -- Note: We're now directly passing v instead of conj_vals_int
            ckks_combined_encode_encrypt_sym(
                &parms,
                test_message ? v : NULL,  // Only pass values if we're testing with a message
                vlen,
                &shareable_prng,
                &prng,
                s,
                ntt_pte,
                ntt_roots,
                c0,
                c1,
                s_test_save,
                c1_test_save,
                conj_vals,  // Use existing buffer for encoding
                index_map
            );
            
            // -- Check that decrypt gives back the pt+err and decode gives back v.
            bool s_test_save_small = false;
            check_decode_decrypt_inpl(c0, c1_test_save, v, vlen, s_test_save, s_test_save_small,
                                      ntt_pte, index_map, &parms, temp_test_mem);
            
            // -- Done checking this prime. Now try next prime if requested
            bool ret = ckks_next_prime_sym(&parms, s);
            se_assert(ret || (!ret && i + 1 == parms.nprimes));
        }
        
        // -- Can exit now if rlwe testing only
        if (!test_message) break;
    }

    // Cleanup
    if (mempool)
    {
        free(mempool);
        mempool = 0;
    }
    if (s_test_save)
    {
        free(s_test_save);
        s_test_save = 0;
    }
    if (c1_test_save)
    {
        free(c1_test_save);
        c1_test_save = 0;
    }
    if (temp_test_mem)
    {
        free(temp_test_mem);
        temp_test_mem = 0;
    }
    delete_parameters(&parms);
}

/**
Full encode + symmetric encrypt test
@param[in] n        Polynomial ring degree (ignored if SE_USE_MALLOC is defined)
@param[in] nprimes  # of modulus primes    (ignored if SE_USE_MALLOC is defined)
*/
void test_ckks_encode_encrypt_sym(size_t n, size_t nprimes)
{
    printf("Beginning tests for ckks encode + symmetric encrypt...\n");
    bool test_message = 1;
    //test_ckks_sym_base(n, nprimes, test_message);

    // Newer combined encode + encrypt test with encoding combined into one function with encryption
    test_ckks_combined_sym_base(n, nprimes, test_message);
}

/**
Symmetric rlwe test only (message is the all-zeros vector)
@param[in] n        Polynomial ring degree (ignored if SE_USE_MALLOC is defined)
@param[in] nprimes  # of modulus primes    (ignored if SE_USE_MALLOC is defined)
*/
void test_enc_zero_sym(size_t n, size_t nprimes)
{
    printf("Beginning tests for rlwe symmetric encryption of 0...\n");
    bool test_message = 0;
    test_ckks_sym_base(n, nprimes, test_message);
}