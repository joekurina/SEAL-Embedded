/**
 * @file ckks_sym.hpp
 * 
 * C++ header for symmetric CKKS encryption.
 * Converted from the C version to provide modern C++ interfaces while
 * maintaining compatibility with the existing codebase.
 */

 #pragma once

 #include <complex>
 #include <cstdint>
 #include <cstdbool>
 #include <memory>
 
 // Include our C++ ckks_common header
 #include "ckks_common.h"
 
 // Original C includes - maintain compatibility with C code
 extern "C" {
     #include "defines.h"
     #include "parameters.h"
     #include "rng.h"
 }
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 #ifdef SE_USE_MALLOC
 /**
  * Returns the required size of the memory pool in units of sizeof(ZZ).
  *
  * @param[in] degree  Desired polynomial ring degree
  * @returns           Required size of the memory pool in units of sizeof(ZZ)
  */
 size_t ckks_get_mempool_size_sym(size_t degree);
 
 /**
  * Sets up the memory pool for CKKS symmetric encryption.
  *
  * Note: This function calls calloc.
  *
  * @param[in] degree  Desired polynomial ring degree
  * @returns           A handle to the memory pool
  */
 ZZ* ckks_mempool_setup_sym(size_t degree);
 #endif
 
 /**
  * Sets addresses of objects according to parameter settings for symmetric CKKS encryption.
  * Should only need to be called once during initial memory allocation.
  *
  * @param[in]  degree   Polynomial ring degree
  * @param[in]  mempool  Handle to memory pool
  * @param[out] se_ptrs  Pointers object. Pointers will be updated to mempool locations.
  */
 void ckks_set_ptrs_sym(size_t degree, ZZ* mempool, SE_PTRS* se_ptrs);
 
 /**
  * Sets up a CKKS secret key. For symmetric encryption, this should either be called once at the start
  * during memory allocation (if storing s persistently in working memory) or once per encode-encrypt
  * sequence (between base and remaining steps, if not storing s persistently in working memory).
  *
  * If parms.sample_s == 1, will internally generate the secret key (from the uniform ternary
  * distribution). This is mainly useful for testing. If parms.sample_s == 0, will read in secret key
  * from file.
  *
  * Size req: If seed is !NULL, seed must be SE_PRNG_SEED_BYTE_COUNT long.
  *
  * @param[in]     parms  Parameters set by ckks_setup
  * @param[in]     seed   [Optional]. Seed to seed 'prng' with, if prng is used.
  * @param[in,out] prng   [Optional]. PRNG instance needed to generate randomness for secret key
  *                      polynomial. Should not be shared.
  * @param[out]    s      Secret key polynomial. Must have space for n coefficients. If 'small' s is
  *                      used, this must be 2 bits per coefficient. Otherwise, this must be sizeof(ZZ)
  *                      per coefficient.
  */
 void ckks_setup_s(const Parms* parms, uint8_t* seed, SE_PRNG* prng, ZZ* s);
 
 /**
  * Initializes values for a single full symmetric CKKS encryption. Samples the error (w/o respect to
  * any prime). Should be called once per encode-encrypt sequence (just after ckks_encode_base).
  *
  * Note: This function modifies (i.e. resets and re-randomizes) the prng instance.
  *
  * Size req: If seeds are !NULL, seeds must be SE_PRNG_SEED_BYTE_COUNT long.
  *
  * @param[in]     parms           Parameters set by ckks_setup
  * @param[in]     share_seed_in   [Optional]. Seed to seed 'shareable_prng' with.
  * @param[in]     seed_in         [Optional]. Seed to seed 'prng' with.
  * @param[in,out] shareable_prng  PRNG instance needed to generate first component of ciphertexts. Is
  *                               safe to share.
  * @param[in,out] prng            PRNG instance needed to generate error polynomial. Should not be
  *                               shared.
  * @param[in,out] conj_vals_int   As pointed to by conj_vals_int_ptr (n int64 values).
  *                               In: ckks pt; Out: pt + error (non-reduced)
  */
 void ckks_sym_init(const Parms* parms, uint8_t* share_seed_in, uint8_t* seed_in,
                   SE_PRNG* shareable_prng, SE_PRNG* prng, int64_t* conj_vals_int);
 
 /**
  * Encodes and symmetrically encrypts a vector of values using CKKS for the current modulus prime.
  *
  * Internally converts various objects to NTT form. If debugging encryption-only (assuming a message of
  * 0), or if calling to generate a public key, can set conj_vals_int to zero. In this case, must set
  * ep_small to the compessed form of the error.
  *
  * @param[in]     parms           Parameters set by ckks_setup
  * @param[in]     conj_vals_int   [Optional]. See description.
  * @param[in]     ep_small        [Optional]. See description. For debugging only.
  * @param[in,out] shareable_prng  PRNG instance needed to generate first component of ciphertexts. Is
  *                               safe to share.
  * @param         s_small         Secret key in small form
  * @param         ntt_pte         Scratch space. Will be used to store pt + e (in NTT form)
  * @param         ntt_roots       Scratch space. May be used to load NTT roots.
  * @param[out]    c0_s            1st component of the ciphertext. Stores n coeffs of size ZZ.
  * @param[out]    c1              2nd component of the ciphertext. Stores n coeffs of size ZZ.
  * @param[out]    s_save          [Optional]. Useful for testing.
  * @param[out]    c1_save         [Optional]. Useful for testing.
  */
 void ckks_encode_encrypt_sym(const Parms* parms, const int64_t* conj_vals_int,
                             const int8_t* ep_small, SE_PRNG* shareable_prng, ZZ* s_small,
                             ZZ* ntt_pte, ZZ* ntt_roots, ZZ* c0_s, ZZ* c1, ZZ* s_save, ZZ* c1_save);
 
 /**
  * Updates parameters to next prime in modulus switching chain for symmetric CKKS encryption. Also
  * converts secret key polynomial to next prime modulus if used in expanded form (compressed form s
  * will be reduced later).
  *
  * @param[in,out] parms  Parameters set by ckks_setup. curr_modulus_idx will be advanced by 1
  * @param[in,out] s      [Optional]. Secret key to convert to next modulus prime
  * @returns              1 on success, 0 on failure (reached end of modulus chain)
  */
 bool ckks_next_prime_sym(Parms* parms, ZZ* s);
 
 #ifdef __cplusplus
 }
 #endif
 
 namespace seal_embedded {
 
 /**
  * @class SymmetricEncryptor
  * @brief C++ wrapper for symmetric CKKS encryption
  * 
  * This class provides a modern C++ interface for the symmetric CKKS encryption functions.
  * It handles memory management and provides a more type-safe interface.
  */
 class SymmetricEncryptor {
 public:
     /**
      * @brief Construct a new Symmetric Encryptor object
      * 
      * @param degree The polynomial ring degree
      * @param nprimes Number of prime moduli
      */
     SymmetricEncryptor(size_t degree, size_t nprimes) 
         : degree_(degree), nprimes_(nprimes) {
         // Setup parameters
         parms_ = std::make_unique<Parms>();
         
         // Allocate memory pool
 #ifdef SE_USE_MALLOC
         mempool_ = ckks_mempool_setup_sym(degree);
 #else
         static_assert(degree == SE_DEGREE_N && nprimes == SE_NPRIMES, 
                      "Static configuration requires matching degree and primes");
         mempool_ = nullptr; // Will be set externally
 #endif
 
         // Setup pointers
         se_ptrs_ = std::make_unique<SE_PTRS>();
         ckks_set_ptrs_sym(degree, mempool_, se_ptrs_.get());
         
         // Initialize PRNG instances
         shareable_prng_ = std::make_unique<SE_PRNG>();
         error_prng_ = std::make_unique<SE_PRNG>();
         
         // Setup parameters and index map
         ckks_setup(degree, nprimes, se_ptrs_->index_map_ptr, parms_.get());
         
         // Initialize the secret key
         ckks_setup_s(parms_.get(), nullptr, error_prng_.get(), se_ptrs_->ternary);
     }
     
     /**
      * @brief Destroy the Symmetric Encryptor object
      */
     ~SymmetricEncryptor() {
 #ifdef SE_USE_MALLOC
         if (mempool_) {
             free(mempool_);
             mempool_ = nullptr;
         }
 #endif
     }
     
     /**
      * @brief Encode and encrypt a vector of values
      * 
      * @param values The values to encode and encrypt
      * @param values_len The number of values
      * @return true on success, false on failure
      */
     bool encode_encrypt(const flpt* values, size_t values_len) {
         // Reset parameters to start of modulus chain
         ckks_reset_primes(parms_.get());
         
         // Encode the values
         bool success = ckks_encode_base(parms_.get(), values, values_len, 
                                       se_ptrs_->index_map_ptr, se_ptrs_->ifft_roots, 
                                       se_ptrs_->conj_vals);
         if (!success) return false;
         
         // Initialize encryption (sample error)
         ckks_sym_init(parms_.get(), nullptr, nullptr, 
                     shareable_prng_.get(), error_prng_.get(), 
                     se_ptrs_->conj_vals_int_ptr);
         
         // Perform encryption for each prime
         for (size_t i = 0; i < nprimes_; i++) {
             ckks_encode_encrypt_sym(parms_.get(), se_ptrs_->conj_vals_int_ptr, 
                                   nullptr, shareable_prng_.get(), 
                                   se_ptrs_->ternary, se_ptrs_->ntt_pte_ptr, 
                                   se_ptrs_->ntt_roots_ptr, se_ptrs_->c0_ptr, 
                                   se_ptrs_->c1_ptr, nullptr, nullptr);
             
             // Process result here (e.g., send over network, store, etc.)
             // ...
             
             // Go to next prime
             if (i < nprimes_ - 1) {
                 bool ret = ckks_next_prime_sym(parms_.get(), se_ptrs_->ternary);
                 if (!ret) break;
             }
         }
         
         return true;
     }
     
     /**
      * @brief Get the first component of the ciphertext
      * 
      * @return ZZ* pointer to the first component
      */
     ZZ* get_c0() const {
         return se_ptrs_->c0_ptr;
     }
     
     /**
      * @brief Get the second component of the ciphertext
      * 
      * @return ZZ* pointer to the second component
      */
     ZZ* get_c1() const {
         return se_ptrs_->c1_ptr;
     }
     
     /**
      * @brief Get the parameters
      * 
      * @return const Parms* pointer to the parameters
      */
     const Parms* get_parameters() const {
         return parms_.get();
     }
 
 private:
     size_t degree_;
     size_t nprimes_;
     std::unique_ptr<Parms> parms_;
     std::unique_ptr<SE_PTRS> se_ptrs_;
     std::unique_ptr<SE_PRNG> shareable_prng_;
     std::unique_ptr<SE_PRNG> error_prng_;
     ZZ* mempool_;
 };
 
 } // namespace seal_embedded