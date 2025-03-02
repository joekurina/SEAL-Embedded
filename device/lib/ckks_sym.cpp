/**
 * @file ckks_sym.cpp
 * C++ implementation of CKKS symmetric encryption for SEAL-Embedded
 */

 #include "ckks_sym.h"

 #include <complex>
 #include <cstdio>
 #include <cstring>
 #include <algorithm>
 #include <memory>
 
 #include "ckks_common.h"
 #include "defines.h"
 #include "fft.h"
 #include "fileops.h"
 #include "modulo.h"
 #include "ntt.h"
 #include "parameters.h"
 #include "polymodarith.h"
 #include "polymodmult.h"
 #include "sample.h"
 #include "uintmodarith.h"
 #include "util_print.h"
 
 extern "C" size_t ckks_get_mempool_size_sym(size_t degree)
 {
     se_assert(degree >= 16);
     
     // If we're using the predefined value, return the constant defined in defines.h
     if (degree == SE_DEGREE_N) {
         return MEMPOOL_SIZE_sym;
     }
     
     // Otherwise, calculate dynamically
     size_t n = degree;
     size_t mempool_size = 4 * n;  // minimum base size
     
     // Add space for values
     mempool_size += n / 2;
     
     // Add space for secret key
     mempool_size += n / 16;
     
     // Add space for index map
     mempool_size += n / 2;
     
     se_assert(mempool_size > 0);
     return mempool_size;
 }
 
 extern "C" ZZ* ckks_mempool_setup_sym(size_t degree)
 {
     size_t mempool_size = ckks_get_mempool_size_sym(degree);
     
     // Use std::calloc for initializing memory to zero
     ZZ* mempool = static_cast<ZZ*>(std::calloc(mempool_size, sizeof(ZZ)));
     
     if (!mempool) {
         std::printf("Error! Allocation failed. Exiting...\n");
         std::exit(1);
     }
     
     se_assert(mempool_size > 0 && mempool != nullptr);
     return mempool;
 }
 
 extern "C" void ckks_set_ptrs_sym(size_t degree, ZZ* mempool, SE_PTRS* se_ptrs)
 {
     se_assert(mempool != nullptr && se_ptrs != nullptr);
     const size_t n = degree;
 
     // First, set everything to set size or nullptr
     se_ptrs->conj_vals = reinterpret_cast<double complex*>(mempool);
     se_ptrs->conj_vals_int_ptr = reinterpret_cast<int64_t*>(mempool);
     se_ptrs->c1_ptr = &(mempool[2 * n]);
     se_ptrs->c0_ptr = &(mempool[3 * n]);
     se_ptrs->ntt_pte_ptr = &(mempool[2 * n]);
 
     // Default values for optional components
     se_ptrs->ternary = &(mempool[3 * n]);  // default: SE_SK_NOT_PERSISTENT
     se_ptrs->ifft_roots = nullptr;         // default: SE_IFFT_OTF
     se_ptrs->index_map_ptr = nullptr;      // default: SE_INDEX_MAP_OTF
     se_ptrs->ntt_roots_ptr = nullptr;      // default: SE_NTT_OTF
     se_ptrs->values = nullptr;
 
     // Sizes for various parts
     size_t ifft_roots_size = 0;
     size_t ntt_roots_size = 0;
     size_t index_map_persist_size = 0;
     size_t s_persist_size = 0;
 
     // Calculate the total size for block 2 (either ifft or ntt roots)
     size_t total_block2_size = ifft_roots_size ? ifft_roots_size : ntt_roots_size;
 
     // Set the index map pointer based on the configuration
     se_ptrs->index_map_ptr = reinterpret_cast<uint16_t*>(&(mempool[4 * n + total_block2_size]));
     index_map_persist_size = n / 2;
 
     // Set the ternary (secret key) pointer
     s_persist_size = n / 16;
     se_ptrs->ternary = &(mempool[4 * n + total_block2_size + index_map_persist_size]);
 
     // Set the values pointer
     se_ptrs->values = reinterpret_cast<flpt*>(
         &(mempool[4 * n + total_block2_size + index_map_persist_size + s_persist_size])
     );
 
     // Validate pointer alignments
     size_t address_size = 4;
     se_assert(reinterpret_cast<ZZ*>(se_ptrs->conj_vals) == 
               reinterpret_cast<ZZ*>(se_ptrs->conj_vals_int_ptr));
     se_assert(se_ptrs->c1_ptr ==
               reinterpret_cast<ZZ*>(se_ptrs->conj_vals_int_ptr) + 2 * n * sizeof(ZZ) / address_size);
     se_assert(se_ptrs->c1_ptr + n * sizeof(ZZ) / address_size == se_ptrs->c0_ptr);
 
     // Debug: print all addresses
     se_print_addresses(mempool, se_ptrs, n, true);
     se_print_relative_positions(mempool, se_ptrs, n, true);
 }
 
 extern "C" void ckks_setup_s(const Parms* parms, uint8_t* seed, SE_PRNG* prng, ZZ* s)
 {
     // Keep s in small form until a later point, so we can store in
     // separate memory in compressed form
     if (parms->sample_s) {
         se_assert(prng != nullptr);
         prng_randomize_reset(prng, seed);
         sample_small_poly_ternary_prng_96(parms->coeff_count, prng, s);
         // TODO: Does not work to sample s for multi prime for now
         // if s and index map share memory
     } else {
         SE_UNUSED(prng);
         load_sk(parms, s);
     }
 }
 
 extern "C" void ckks_sym_init(const Parms* parms, uint8_t* share_seed, uint8_t* seed, 
                     SE_PRNG* shareable_prng, SE_PRNG* prng, int64_t* conj_vals_int)
 {
     // Each prng must be reset & re-randomized once per encode-encrypt sequence.
     // 'prng_randomize_reset' will set the prng seed to a random value and the prng counter to 0
     // (If seeds are not nullptr, seeds will be used to seed prng instead of a random value.)
     // The seed associated with the prng used to sample 'a' can be shared
     // NOTE: The re-randomization is not strictly necessary if counter has not wrapped around
     // and we share both the seed and starting counter value with the server
     // for the shareable part.
     prng_randomize_reset(shareable_prng, share_seed);  // Used for 'a'
     prng_randomize_reset(prng, seed);                  // Used for error
 
     // Sample error polynomial (ep) and add it to the signed plaintext.
     // This prng's seed value should not be shared.
     sample_add_poly_cbd_generic_inpl_prng_16(conj_vals_int, parms->coeff_count, prng);
 }
 
 extern "C" void ckks_encode_encrypt_sym(const Parms* parms, const int64_t* conj_vals_int,
                              const int8_t* ep_small, SE_PRNG* shareable_prng, ZZ* s_small,
                              ZZ* ntt_pte, ZZ* ntt_roots, ZZ* c0_s, ZZ* c1, ZZ* s_save, ZZ* c1_save)
 {
     se_assert(parms != nullptr);
 
     // ==============================================================
     //   Generate ciphertext: (c[1], c[0]) = (a, [-a*s + m + e]_Rq)
     // ==============================================================
     const PolySizeType n = parms->coeff_count;
     const Modulus* mod = parms->curr_modulus;
 
     // ----------------------
     //     c1 = a <--- U
     // ----------------------
     sample_poly_uniform(parms, shareable_prng, c1);
 
     se_assert(conj_vals_int != nullptr || ep_small != nullptr);
     
     // At this point, it is safe to send c1 away. This will allow us to re-use c1's memory.
     // However, we may be debugging and need to store c1 somewhere for debugging later.
     // Note: This method provides very little memory savings overall, so isn't necessary to use.
     if (c1_save != nullptr) {
         std::memcpy(c1_save, c1, n * sizeof(ZZ));
     }
 
     // ----------------------------
     //    c0 = [-a*s + m + e]_Rq
     // ----------------------------
     // Load s (if not already loaded)
     // For now, we require s to be in small form.
     se_assert(s_small != nullptr);
 
     // Expand and store s in c0
     expand_poly_ternary(s_small, parms, c0_s);
 
     // Calculate [a*s]_Rq = [c1*s]_Rq. This will free up c1 space too.
     // First calculate ntt(s) and store in c0_s. Note that this will load
     // the ntt roots into ntt_roots memory as well (used later for
     // calculating ntt(pte))
 
     // Note: Calling ntt_roots_initialize will do nothing if SE_NTT_OTF is defined
     ntt_roots_initialize(parms, ntt_roots);
     ntt_inpl(parms, ntt_roots, c0_s);
 
     // Save ntt(reduced(s)) for later decryption
     if (s_save != nullptr) {
         std::memcpy(s_save, c0_s, n * sizeof(c0_s[0]));
     }
 
     poly_mult_mod_ntt_form_inpl(c0_s, c1, n, mod);
 
     // Negate [a*s]_Rq to get [-a*s]_Rq
     poly_neg_mod_inpl(c0_s, n, mod);
 
     // Calculate reduce(m + e) == reduce(conj_vals_int) ---> store in ntt_pte
     if (ep_small != nullptr) {
         reduce_set_e_small(parms, ep_small, ntt_pte);
     }
 
     if (conj_vals_int != nullptr) {
         reduce_set_pte(parms, conj_vals_int, ntt_pte);
     }
 
     // Calculate ntt(m + e) = ntt(reduce(conj_vals_int)) = ntt(ntt_pte)
     // and store result in ntt_pte. Note: ntt roots (if required) should already be
     // loaded from above
     ntt_inpl(parms, ntt_roots, ntt_pte);
 
     // Add the plaintext + error to the ciphertext
     poly_add_mod_inpl(c0_s, ntt_pte, n, mod);
 }
 
 extern "C" bool ckks_next_prime_sym(Parms* parms, ZZ* s)
 {
     se_assert(parms != nullptr && !parms->is_asymmetric);
 
     if (!parms->small_s) {
         convert_poly_ternary_inpl(s, parms);
     }
 
     // Update curr_modulus_idx to next index
     return next_modulus(parms);
 }