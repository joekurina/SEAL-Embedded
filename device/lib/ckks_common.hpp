/**
 * @file ckks_common.hpp
 * 
 * C++ header for CKKS common functionality.
 * Converted from the C version to provide modern C++ interfaces while
 * maintaining compatibility with the existing codebase.
 */

 #pragma once

 #include <complex>
 #include <cstdint>
 #include <cstddef>
 #include <cstdbool>
 #include <memory>
 
 // Original C includes - maintain compatibility with C code
 extern "C" {
     #include "defines.h"
     #include "modulo.h"
     #include "parameters.h"
     #include "rng.h"
 }
 
 /**
  * Object that stores pointers to various objects for CKKS encode/encryption.
  * For the following, n is the polynomial ring degree.
  */
 struct SE_PTRS {
     std::complex<double>* conj_vals;     // Storage for output of encode
     std::complex<double>* ifft_roots;    // Roots for inverse fft (n double complex values)
     flpt* values;                        // Floating point values to encode/encrypt
     ZZ* ternary;                         // Ternary polynomial ('s' for symmetric, 'u' for asymmetric)
 
     // The following point to sections of conj_vals and ifft_roots above
     int64_t* conj_vals_int_ptr;
     ZZ* c0_ptr;                         // 1st component of a ciphertext for a particular prime
     ZZ* c1_ptr;                         // 2nd component of a ciphertext for a particular prime
     uint16_t* index_map_ptr;            // Index map values
     ZZ* ntt_roots_ptr;                  // Storage for NTT roots
     ZZ* ntt_pte_ptr;                    // Used for adding the plaintext to the error
     int8_t* e1_ptr;                     // Second error polynomial (unused in symmetric case)
 };
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 /**
  * Computes the values for the index map. This corresponds to the "pi" inverse 
  * projection symbol in the CKKS paper, merged with the bit-reversal required for the ifft/fft.
  * 
  * @param parms      Parameters set by ckks_setup
  * @param index_map  Index map values
  */
 void ckks_calc_index_map(const Parms* parms, uint16_t* index_map);
 
 /**
  * Sets the parameters according to the request polynomial ring degree. Also sets the 
  * index map if required (based on index map option defined). This should be called 
  * once during memory allocation to setup the parameters.
  * 
  * @param degree     Desired polynomial ring degree
  * @param nprimes    Desired number of primes
  * @param index_map  [Optional] Pointer to index map values buffer
  * @param parms      Parameters instance
  */
 void ckks_setup(size_t degree, size_t nprimes, uint16_t* index_map, Parms* parms);
 
 /**
  * Sets the parameters according to request custom parameters. Also sets the index map if required
  * (based on index map option defined). This should be called once during memory allocation to setup
  * the parameters. If either 'modulus_vals' or 'ratios' is NULL, uses regular (non-custom) ckks_setup
  * instead.
  * 
  * @param degree        Desired polynomial ring degree
  * @param nprimes       Desired number of primes
  * @param modulus_vals  An array of nprimes type-ZZ modulus values
  * @param ratios        An array of const_ratio values for each custom modulus value
  * @param index_map     [Optional] Pointer to index map values buffer
  * @param parms         Parameters instance
  */
 void ckks_setup_custom(size_t degree, size_t nprimes, const ZZ* modulus_vals, const ZZ* ratios,
                         uint16_t* index_map, Parms* parms);
 
 /**
  * Resets the encryption parameters. Should be called once per encode-encrypt sequence to set
  * curr_modulus_idx back to the start of the modulus chain. Does not need to be called the very first
  * time after ckks_setup_parms is called, however.
  * 
  * @param parms  Parameters set by ckks_setup
  */
 void ckks_reset_primes(Parms* parms);
 
 /**
  * CKKS encoding base (w/o respect to a particular modulus). Should be called once per encode-encrypt
  * sequence. Encoding can fail for certain inputs, so returns a value indicating success or failure.
  * 
  * @param parms       Parameters set by ckks_setup
  * @param values      Initial message array with (up to) n/2 slots
  * @param values_len  Number of elements in values array. Must be <= n/2
  * @param index_map   [Optional] If passed in, can avoid 1 flash read
  * @param ifft_roots  Scratch space to load ifft roots
  * @param conj_vals   conj_vals_int in first n ZZ values
  * @returns           True on success, False on failure
  */
 bool ckks_encode_base(const Parms* parms, const flpt* values, size_t values_len,
                       uint16_t* index_map, std::complex<double>* ifft_roots, 
                       std::complex<double>* conj_vals);
 
 /**
  * Reduces all values in conj_vals_int modulo the current modulus and stores result in out.
  * 
  * @param parms          Parameters set by ckks_setup
  * @param conj_vals_int  Array of values to be reduced, as output by ckks_encode_base
  * @param out            Result array of reduced values
  */
 void reduce_set_pte(const Parms* parms, const int64_t* conj_vals_int, ZZ* out);
 
 /**
  * Reduces all values in conj_vals_int modulo the current modulus and adds result to out.
  * 
  * @param parms          Parameters set by ckks_setup
  * @param conj_vals_int  Array of values to be reduced
  * @param out            Updated array
  */
 void reduce_add_pte(const Parms* parms, const int64_t* conj_vals_int, ZZ* out);
 
 /**
  * Converts an error polynomial in small form to an error polynomial modulo the current
  * modulus.
  * 
  * @param parms  Parameters set by ckks_setup
  * @param e      Error polynomial in small form
  * @param out    Result error polynomial modulo the current modulus
  */
 void reduce_set_e_small(const Parms* parms, const int8_t* e, ZZ* out);
 
 /**
  * Converts an error polynomial in small form to an error polynomial modulo the current
  * modulus and adds it to the values stored in the 'out' array.
  * 
  * @param parms  Parameters set by ckks_setup
  * @param e      Error polynomial in small form
  * @param out    Result updated polynomial
  */
 void reduce_add_e_small(const Parms* parms, const int8_t* e, ZZ* out);
 
 #ifdef SE_USE_MALLOC
 /**
  * Prints the relative positions of various objects.
  * 
  * @param st       Starting address of memory pool
  * @param se_ptrs  SE_PTRS object contains pointers to objects
  * @param n        Polynomial ring degree
  * @param sym      Set to 1 if in symmetric mode
  */
 void se_print_relative_positions(const ZZ* st, const SE_PTRS* se_ptrs, size_t n, bool sym);
 
 /**
  * Prints the addresses of various objects.
  * 
  * @param mempool  Memory pool handle
  * @param se_ptrs  SE_PTRS object contains pointers to objects
  * @param n        Polynomial ring degree
  * @param sym      Set to 1 if in symmetric mode
  */
 void se_print_addresses(const ZZ* mempool, const SE_PTRS* se_ptrs, size_t n, bool sym);
 
 /**
  * Prints a banner for the size of the memory pool.
  * 
  * @param n        Polynomial ring degree
  * @param sym      Set to 1 if in symmetric mode
  */
 void print_ckks_mempool_size(size_t n, bool sym);
 #else
 /**
  * Prints the relative positions of various objects.
  * 
  * @param st       Starting address of memory pool
  * @param se_ptrs  SE_PTRS object contains pointers to objects
  */
 void se_print_relative_positions(const ZZ* st, const SE_PTRS* se_ptrs);
 
 /**
  * Prints the addresses of various objects.
  * 
  * @param mempool  Memory pool handle
  * @param se_ptrs  SE_PTRS object contains pointers to objects
  */
 void se_print_addresses(const ZZ* mempool, const SE_PTRS* se_ptrs);
 
 /**
  * Prints a banner for the size of the memory pool.
  */
 void print_ckks_mempool_size(void);
 #endif
 
 #ifdef __cplusplus
 }
 #endif
 
 // C++ only interfaces and classes would go here
 
 // Calculate mempool size for no-malloc case
 #ifdef SE_IFFT_OTF
 #if defined(SE_NTT_ONE_SHOT) || defined(SE_NTT_REG)
 #define MEMPOOL_SIZE_BASE 5 * SE_DEGREE_N
 #elif defined(SE_NTT_FAST)
 #define MEMPOOL_SIZE_BASE 7 * SE_DEGREE_N
 #else
 #define MEMPOOL_SIZE_BASE 4 * SE_DEGREE_N
 #endif
 #else
 #define MEMPOOL_SIZE_BASE 8 * SE_DEGREE_N
 #endif
 
 #if defined(SE_INDEX_MAP_PERSIST) || defined(SE_INDEX_MAP_LOAD_PERSIST) || \
     defined(SE_INDEX_MAP_LOAD_PERSIST_SYM_LOAD_ASYM) || defined(SE_SK_INDEX_MAP_SHARED)
 #define SE_INDEX_MAP_PERSIST_SIZE_sym SE_DEGREE_N / 2
 #else
 #define SE_INDEX_MAP_PERSIST_SIZE_sym 0
 #endif
 
 #ifdef SE_INDEX_MAP_LOAD_PERSIST_SYM_LOAD_ASYM
 #define SE_INDEX_MAP_PERSIST_SIZE_asym 0
 #else
 #define SE_INDEX_MAP_PERSIST_SIZE_asym SE_INDEX_MAP_PERSIST_SIZE_sym
 #endif
 
 #ifdef SE_SK_PERSISTENT
 #define SK_PERSIST_SIZE SE_DEGREE_N / 16
 #else
 #define SK_PERSIST_SIZE 0
 #endif
 
 #ifdef SE_MEMPOOL_ALLOC_VALUES
 #define VALUES_ALLOC_SIZE SE_DEGREE_N / 2
 #else
 #define VALUES_ALLOC_SIZE 0
 #endif
 
 #define MEMPOOL_SIZE_sym \
     MEMPOOL_SIZE_BASE + SE_INDEX_MAP_PERSIST_SIZE_sym + SK_PERSIST_SIZE + VALUES_ALLOC_SIZE
 
 #ifdef SE_IFFT_OTF
 #define MEMPOOL_SIZE_BASE_Asym MEMPOOL_SIZE_BASE + SE_DEGREE_N + SE_DEGREE_N / 4 + SE_DEGREE_N / 16
 // 4n + n + n + n/4 + n/16
 #else
 #define MEMPOOL_SIZE_BASE_Asym MEMPOOL_SIZE_BASE
 #endif
 
 #define MEMPOOL_SIZE_Asym \
     MEMPOOL_SIZE_BASE_Asym + SE_INDEX_MAP_PERSIST_SIZE_asym + VALUES_ALLOC_SIZE
 
 #ifdef SE_ENCRYPT_TYPE_SYMMETRIC
 #define MEMPOOL_SIZE MEMPOOL_SIZE_sym
 #else
 #define MEMPOOL_SIZE MEMPOOL_SIZE_Asym
 #endif