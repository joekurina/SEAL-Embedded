/**
 * @file ckks_tests_asym.cpp
 * C++ implementation of asymmetric CKKS encryption testing for SEAL-Embedded
 */

 #include <cmath>      // for pow
 #include <cstdio>     // for printf
 #include <cstdlib>    // for malloc, free
 #include <cstring>    // for memset, memcpy
 #include <memory>     // for std::unique_ptr
 #include <complex>    // for std::complex
 
 #include "ckks_asym.h"
 #include "ckks_common.h"
 #include "ckks_sym.h"
 #include "ckks_tests_common.h"
 #include "defines.h"
 #include "ntt.h"
 #include "polymodarith.h"
 #include "sample.h"
 #include "test_common.h"
 #include "util_print.h"
 
 /**
  * Asymmetric test core. If debugging is enabled, throws an error if a test fails.
  *
  * @param[in] n             Polynomial ring degree (ignored if SE_USE_MALLOC is defined)
  * @param[in] nprimes       # of modulus primes    (ignored if SE_USE_MALLOC is defined)
  * @param[in] test_message  If 0, sets the message 0
  */
 void test_ckks_asym_base(size_t n, size_t nprimes, bool test_message)
 {
     Parms parms;
     parms.sample_s      = false;
     parms.is_asymmetric = true;
     parms.pk_from_file  = false;  // must be false to check on device
     parms.small_s       = true;   // try this first
     parms.small_u       = true;   // try this first
     bool encode_only    = false;
 
     // -- Make sure we didn't set this accidentally
     if (!parms.sample_s) se_assert(parms.small_s);
 
     // Allocate memory pool
     ZZ* mempool = nullptr;

     print_ckks_mempool_size(n, 0);
     mempool = ckks_mempool_setup_asym(n);
 
     // -- Get pointers
     SE_PTRS se_ptrs_local;
     ckks_set_ptrs_asym(n, mempool, &se_ptrs_local);
     
     // Create pointers to the various components
     auto conj_vals = se_ptrs_local.conj_vals;
     auto conj_vals_int = se_ptrs_local.conj_vals_int_ptr;
     auto ifft_roots = se_ptrs_local.ifft_roots;
     auto pk_c0 = se_ptrs_local.c0_ptr;
     auto pk_c1 = se_ptrs_local.c1_ptr;
     auto index_map = se_ptrs_local.index_map_ptr;
     auto ntt_roots = se_ptrs_local.ntt_roots_ptr;
     auto ntt_u_e1_pte = se_ptrs_local.ntt_pte_ptr;
     auto u = se_ptrs_local.ternary;
     auto v = se_ptrs_local.values;
     auto e1 = se_ptrs_local.e1_ptr;
     size_t vlen = n / 2;
 
     if (!test_message) {
         std::memset(v, 0, vlen * sizeof(flpt));
     }
 
     // -- Additional pointers required for testing.
     // Use smart pointers or manage memory carefully
     ZZ* s = static_cast<ZZ*>(std::calloc(n / 16, sizeof(ZZ)));
     int8_t* ep_small = static_cast<int8_t*>(std::calloc(n, sizeof(int8_t)));
     ZZ* ntt_s_save = static_cast<ZZ*>(std::calloc(n, sizeof(ZZ)));  // ntt(expanded(s)) or expanded(s)
     
     printf("            s addr: %p\n", s);
     printf("     ep_small addr: %p\n", ep_small);
     printf("   ntt_s_save addr: %p\n", ntt_s_save);
 
     // -- This is too much memory for the NRF5, so just allocate as needed later in that case
     ZZ* ntt_ep_save = static_cast<ZZ*>(std::calloc(n, sizeof(ZZ)));  // ntt(reduced(ep)) or reduced(ep)
     ZZ* ntt_e1_save = static_cast<ZZ*>(std::calloc(n, sizeof(ZZ)));
     ZZ* ntt_u_save = static_cast<ZZ*>(std::calloc(n, sizeof(ZZ)));
     ZZ* temp_test_mem = static_cast<ZZ*>(std::calloc(4 * n, sizeof(ZZ)));
     
     printf("  ntt_ep_save addr: %p\n", ntt_ep_save);
     printf("  ntt_e1_save addr: %p\n", ntt_e1_save);
     printf("   ntt_u_save addr: %p\n", ntt_u_save);
     printf("temp_test_mem addr: %p\n", temp_test_mem);
 
     SE_PRNG prng;
     SE_PRNG shareable_prng;
 
     // -- Set up parameters and index_map if applicable
     ckks_setup(n, nprimes, index_map, &parms);
 
     print_test_banner("Asymmetric Encryption", &parms);
 
     // -- If s is allocated space ahead of time, can load ahead of time too.
     // -- (If we are testing and sample s is set, this will also sample s)
     ckks_setup_s(&parms, nullptr, &prng, s);
     size_t s_size = parms.small_s ? n / 16 : n;
     if (encode_only) clear(s, s_size);
 
     for (size_t testnum = 0; testnum < 9; testnum++)
     {
         printf("-------------------- Test %zu -----------------------\n", testnum);
         ckks_reset_primes(&parms);
 
         if (test_message)
         {
             // -- Set test values
             set_encode_encrypt_test(testnum, vlen, v);
             print_poly_flpt("v        ", v, vlen);
         }
 
         // ------------------------------------------
         // ----- Begin encode-encrypt sequence ------
         // ------------------------------------------
 
         // -- First, encode
         if (test_message)
         {
             bool ret = ckks_encode_base(&parms, v, vlen, index_map, ifft_roots, conj_vals);
             se_assert(ret);
         }
         else {
             std::memset(conj_vals_int, 0, n * sizeof(conj_vals_int[0]));
         }
         // print_poly_int64("conj_vals_int      ", conj_vals_int, n);
         se_assert(v);
 
         // -- Sample u, ep, e0, e1. While sampling e0, add in-place to base message.
         if (!encode_only)
         {
             // -- Sample ep here for generating the public key later
             sample_poly_cbd_generic_prng_16(n, &prng, ep_small);

             // print_poly_int8_full("ep_small", ep_small, n);
             // printf("About to init\n");
             ckks_asym_init(&parms, nullptr, &prng, conj_vals_int, u, e1);
             // printf("Back from init\n");
         }
 
         // -- Debugging
         // size_t u_size = parms.small_u ? n/16 : n;
         // memset(u, 0, u_size * sizeof(ZZ));
         // if (parms.small_u) set_small_poly_idx(0, 1, u);
         // else u[0] = 1;
 
         print_poly_ternary("u   ", u, n, true);
         print_poly_ternary("s   ", s, n, true);
         // print_poly_int8_full("e1  ", e1, n);
 
         for (size_t i = 0; i < parms.nprimes; i++)
         {
             const Modulus* mod = parms.curr_modulus;
             print_zz(" ***** Modulus", mod->value);
 
             // -- We can't just load pk if we are testing because we need ep to
             //    check the "decryption". Need to generate pk here instead to keep
             //    track of the secret key error term.
             se_assert(!parms.pk_from_file);
             printf("generating pk...\n");
             gen_pk(&parms, s, ntt_roots, nullptr, &shareable_prng, ntt_s_save, ep_small, ntt_ep_save,
                    pk_c0, pk_c1);
             printf("...done generating pk.\n");
 
             // -- Debugging
             print_poly("pk0 ", pk_c0, n);
             print_poly("pk1 ", pk_c1, n);
             print_poly_ternary("u   ", u, n, 1);
             // print_poly_int8_full("e1  ", e1, n);
             // if (ntt_e1_ptr) print_poly_full("ntt e1  ", ntt_e1_ptr, n);
 
             // -- Per prime Encode + Encrypt
             ckks_encode_encrypt_asym(&parms, conj_vals_int, u, e1, ntt_roots, ntt_u_e1_pte,
                                      ntt_u_save, ntt_e1_save, pk_c0, pk_c1);
             print_poly_int64("conj_vals_int      ", conj_vals_int, n);
 
             // -- Debugging
             // print_poly_int8_full("e1  ", e1, n);
             // if (ntt_e1_ptr) print_poly_full("ntt e1  ", ntt_e1, n);
             print_poly_ternary("u   ", u, n, 1);
             // if (ntt_u_save) print_poly("ntt_u_save   ", ntt_u_save, n);
 
             // -- Decryption does the following:
             //    (c1 = pk1*u + e1)*s + (c0 = pk0*u + e0 + m)  -->
             //    ((pk1 = c1 = a)*u + e1)*s + ((pk0 = c1 = -a*s+ep)*u + e0 + m) -->
             //    a*u*s + e1*s -a*s*u + ep*u + e0 ===> e1*s + ep*u + e0 + m
             print_poly("c0      ", pk_c0, n);
             print_poly("c1      ", pk_c1, n);
             print_poly("ntt(u)  ", ntt_u_save, n);
             print_poly("ntt(ep) ", ntt_ep_save, n);
             poly_mult_mod_ntt_form_inpl(ntt_u_save, ntt_ep_save, n, mod);
             print_poly("ntt(u) * ntt(ep)", ntt_u_save, n);
 
             print_poly("ntt(s)  ", ntt_s_save, n);
             print_poly("ntt(e1) ", ntt_e1_save, n);
             poly_mult_mod_ntt_form_inpl(ntt_e1_save, ntt_s_save, n, mod);
             print_poly("ntt(s) * ntt(e1)", ntt_e1_save, n);
 
             print_poly("ntt(u) * ntt(ep)", ntt_u_save, n);
             poly_add_mod_inpl(ntt_u_save, ntt_e1_save, n, mod);
             print_poly("ntt(u) * ntt(ep) + ntt(s) * ntt(e1)", ntt_u_save, n);
 
             print_poly("ntt(m + e0)", ntt_u_e1_pte, n);
             poly_add_mod_inpl(ntt_u_e1_pte, ntt_u_save, n, mod);
             print_poly("ntt(u) * ntt(ep) + ntt(s) * ntt(e1) + ntt(m + e0)", ntt_u_e1_pte, n);
             ZZ* pterr = ntt_u_e1_pte;
 
             // -- Check that decrypt gives back the pt+err and decode gives back v.
             // -- Note: This will only decode if values is non-zero. Otherwise, will just decrypt.
             // -- Note: sizeof(max(ntt_roots, ifft_roots)) must be passed as temp memory to undo
             //    ifft.
             bool s_test_save_small = false;
             check_decode_decrypt_inpl(pk_c0, pk_c1, v, vlen, ntt_s_save, s_test_save_small, pterr,
                                       index_map, &parms, temp_test_mem);
 
             // -- Done checking this prime, now try next prime if requested
             // -- Note: This does nothing to u if u is in small form
             bool ret = ckks_next_prime_asym(&parms, u);
             se_assert(ret || (!ret && i + 1 == parms.nprimes));
         }
 
         // -- Can exit now if rlwe testing only
         if (!test_message) break;
     }
 
     // Clean up memory
     if (mempool) {
         std::free(mempool);
         mempool = nullptr;
     }
     if (s) {
         std::free(s);
         s = nullptr;
     }
     if (ep_small) {
         std::free(ep_small);
         ep_small = nullptr;
     }
     if (ntt_s_save) {
         std::free(ntt_s_save);
         ntt_s_save = nullptr;
     }

     if (ntt_ep_save) {
         std::free(ntt_ep_save);
         ntt_ep_save = nullptr;
     }
     if (ntt_e1_save) {
         std::free(ntt_e1_save);
         ntt_e1_save = nullptr;
     }
     if (ntt_u_save) {
         std::free(ntt_u_save);
         ntt_u_save = nullptr;
     }
     if (temp_test_mem) {
         std::free(temp_test_mem);
         temp_test_mem = nullptr;
     }

     delete_parameters(&parms);
 }
 
 /**
  * Full encode + asymmetric encrypt test
  *
  * @param[in] n        Polynomial ring degree (ignored if SE_USE_MALLOC is defined)
  * @param[in] nprimes  # of modulus primes    (ignored if SE_USE_MALLOC is defined)
  */
 extern "C" void test_ckks_encode_encrypt_asym(size_t n, size_t nprimes)
 {
     printf("Beginning tests for ckks encode + asymmetric encrypt...\n");
     bool test_message = true;
     test_ckks_asym_base(n, nprimes, test_message);
 }
 
 /**
  * Asymmetric rlwe test only (message is the all-zeros vector)
  *
  * @param[in] n        Polynomial ring degree (ignored if SE_USE_MALLOC is defined)
  * @param[in] nprimes  # of modulus primes    (ignored if SE_USE_MALLOC is defined)
  */
 extern "C" void test_enc_zero_asym(size_t n, size_t nprimes)
 {
     printf("Beginning tests for rlwe asymmetric encryption of 0...\n");
     bool test_message = false;
     test_ckks_asym_base(n, nprimes, test_message);

 }