/**
 * @file ckks_common.cpp
 * C++ implementation of common CKKS functionality for SEAL-Embedded
 */

 #include "ckks_common.h"

 #include <cmath>
 #include <cstdio>
 #include <cstdlib>
 #include <algorithm>
 #include <memory>
 
 #include "ckks_asym.h"
 #include "ckks_sym.h"
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
 
 // Define this constant in C++ style
 static const double MAX_INT_64_DOUBLE = static_cast<double>(0x7FFFFFFFFFFFFFFFULL);
 
 // This function maintains C-compatibility since it's called from C code
 extern "C" void ckks_calc_index_map(const Parms *parms, uint16_t *index_map)
 {
     // Ensure valid inputs
     se_assert(parms);
     size_t n = parms->coeff_count;
     se_assert(n <= 16384);  // uint16_t can't handle larger sizes
 
     uint64_t m = static_cast<uint64_t>(n) * 2;  // m = 2n
     size_t slot_count = n / 2;                  // slot_count = n/2
     size_t logn = parms->logn;                  // number of bits to represent n
 
     // Generator for the multiplicative group (3 generates a mult. group mod 2^n w/ order n/2)
     uint64_t gen = 3;
     uint64_t pos = 1;
 
     for (size_t i = 0; i < n / 2; i++) {
         // We want index1 + index2 to equal n-1
         size_t index1 = (static_cast<size_t>(pos) - 1) / 2;
         size_t index2 = n - index1 - 1;
 
         // Merge index mapping step with bitrev step required for later application of ifft/fft
         index_map[i] = static_cast<uint16_t>(bitrev(index1, logn));
         index_map[i + slot_count] = static_cast<uint16_t>(bitrev(index2, logn));
 
         // Next root
         pos *= gen;
 
         // Since m is a power of 2, m-1 sets all bits less significant than the '1'
         // bit in the value m. A bit-wise 'and' with (m-1) is therefore essentially
         // a reduction modulo m. Ex: m = 4 = 0100, m-1 = 3 = 0011 --> if pos = 21
         // = 10101: 21 % 4 = 1 = 10101 & 0011
         pos &= (m - 1);
     }
 }
 
 extern "C" void ckks_setup(size_t degree, size_t nprimes, uint16_t *index_map, Parms *parms)
 {
     set_parms_ckks(degree, nprimes, parms);
 #ifdef SE_INDEX_MAP_PERSIST
     ckks_calc_index_map(parms, index_map);
 #elif defined(SE_INDEX_MAP_LOAD_PERSIST)
     load_index_map(parms, index_map);
 #elif defined(SE_INDEX_MAP_LOAD_PERSIST_SYM_LOAD_ASYM)
     if (!parms->is_asymmetric) load_index_map(parms, index_map);
 #endif
 }
 
 extern "C" void ckks_setup_custom(size_t degree, size_t nprimes, const ZZ *modulus_vals, const ZZ *ratios,
                        uint16_t *index_map, Parms *parms)
 {
     if (!modulus_vals || !ratios) {
         ckks_setup(degree, nprimes, index_map, parms);
         return;
     }
     
     set_custom_parms_ckks(degree, 0.0, nprimes, modulus_vals, ratios, parms);
     
 #ifdef SE_INDEX_MAP_PERSIST
     ckks_calc_index_map(parms, index_map);
 #elif defined(SE_INDEX_MAP_LOAD_PERSIST)
     load_index_map(parms, index_map);
 #elif defined(SE_INDEX_MAP_LOAD_PERSIST_SYM_LOAD_ASYM)
     if (!parms->is_asymmetric) load_index_map(parms, index_map);
 #endif
 }
 
 extern "C" void ckks_reset_primes(Parms *parms)
 {
     reset_primes(parms);
 }
 
 extern "C" bool ckks_encode_base(const Parms *parms, const flpt *values, size_t values_len,
                       uint16_t *index_map, complex_double *ifft_roots, complex_double *conj_vals)
 {
     se_assert(parms);
     size_t n = parms->coeff_count;
     size_t logn = parms->logn;
     double scale = parms->scale;
 
 #ifdef SE_INDEX_MAP_LOAD
     se_assert(index_map);
     load_index_map(parms, index_map);
 #elif defined(SE_INDEX_MAP_LOAD_PERSIST_SYM_LOAD_ASYM)
     if (parms->is_asymmetric) {
         se_assert(index_map);
         load_index_map(parms, index_map);
     }
 #endif
 
 #ifdef SE_INDEX_MAP_OTF
     SE_UNUSED(index_map);
     uint64_t gen = 3;
     uint64_t pos = 1;
     uint64_t m = static_cast<uint64_t>(n) * 2;  // m = 2n
 
     for (size_t i = 0; i < values_len; i++, pos = ((pos * gen) & (m - 1))) {
         size_t index1 = (static_cast<size_t>(pos) - 1) / 2;
         size_t index2 = n - index1 - 1;
         uint16_t index1_rev = static_cast<uint16_t>(bitrev(index1, logn));
         uint16_t index2_rev = static_cast<uint16_t>(bitrev(index2, logn));
 #else
     size_t slot_count = n / 2;
     for (size_t i = 0; i < values_len; i++) {
         se_assert(index_map);
         uint16_t index1_rev = index_map[i];
         uint16_t index2_rev = index_map[i + slot_count];
 #endif
         se_assert(index1_rev < n);
         se_assert(index2_rev < n);
         
         // For C++, handle complex types correctly
 #ifdef __cplusplus
         double val_real = static_cast<double>(values[i]);
         std::complex<double>* complex_ptr = reinterpret_cast<std::complex<double>*>(conj_vals);
         complex_ptr[index1_rev] = std::complex<double>(val_real, 0.0);
         complex_ptr[index2_rev] = std::complex<double>(val_real, 0.0);
 #else
         // For C code
         double val_real = (double)(values[i]);
         conj_vals[index1_rev] = val_real;
         conj_vals[index2_rev] = val_real;
 #endif
         
         // Note: conj_vals[index2_rev] should be set to conj(val), but since we
         // assume values[i] is non-complex, val == conj(val)
     }
 
 #ifdef SE_VERBOSE_TESTING
     print_poly_double_complex("conj_vals inside", (double complex*)conj_vals, n);
 #endif
 
     // Note: ifft_roots argument will be ignored if SE_IFFT_OTF is defined
     ifft_inpl((double complex*)conj_vals, n, logn, (double complex*)ifft_roots);
 
 #ifdef SE_VERBOSE_TESTING
     print_poly_double_complex("ifft(conj_vals)", (double complex*)conj_vals, n);
 
     // Don't combine ifft step of dividing by n with ckks step of scaling by "scale"
     double n_inv = 1.0 / static_cast<double>(n);
     se_assert(n_inv >= 0);
 
 #ifdef __cplusplus
     std::complex<double>* complex_ptr = reinterpret_cast<std::complex<double>*>(conj_vals);
     for (size_t i = 0; i < n; i++) {
         complex_ptr[i] *= n_inv;
     }
 #else
     for (size_t i = 0; i < n; i++) {
         conj_vals[i] *= n_inv;
     }
 #endif
 
     print_poly_double_complex("conj_vals", (double complex*)conj_vals, n);
 
     n_inv = scale;
 #else
     // Combine ifft step of dividing by n with ckks step of scaling by "scale"
     double n_inv = scale / static_cast<double>(n);
 #endif
 
     // We no longer need the imaginary part of conj_vals
     int64_t *conj_vals_int = (int64_t *)conj_vals;
     for (size_t i = 0; i < n; i++) {
         // Using C++ std::complex and std::round for C++ code
 #ifdef __cplusplus
         std::complex<double>* complex_ptr = reinterpret_cast<std::complex<double>*>(conj_vals);
         double coeff = std::round(complex_ptr[i].real() * n_inv);
 #else
         double coeff = round(creal(conj_vals[i]) * n_inv);
 #endif
 
         // Check to make sure value can fit in an int64_t
         if (std::fabs(coeff) > MAX_INT_64_DOUBLE) {
             printf("Error! Value at index %zu is possibly too large.\n", i);
 #ifdef __cplusplus
             printf("complex_val.real():     %0.6f\n", complex_ptr[i].real());
 #else
             printf("complex_val.real():     %0.6f\n", creal(conj_vals[i]));
 #endif
             printf("ninv:                   %0.6f\n", n_inv);
             printf("coeff:                  %0.6f\n", coeff);
             printf("fabs(coeff):            %0.6f\n", std::fabs(coeff));
             printf("MAX_INT_64_DOUBLE:      %0.6f\n", MAX_INT_64_DOUBLE);
             return false;
         }
 
         conj_vals_int[i] = static_cast<int64_t>(coeff);
     }
 
     return true;
 }
 
 /**
  * Core functionality for reduce_set_pte and reduce_add_pte functions.
  *
  * @param[in] conj_vals_int  Value to be reduced
  * @param[in] mod            Modulus to reduce value by
  * @returns                  Reduced value
  */
 static ZZ reduce_pte_core(int64_t conj_vals_int, const Modulus *mod)
 {
     uint64_t coeff_abs = static_cast<uint64_t>(llabs(conj_vals_int));
     ZZ mask = static_cast<ZZ>(conj_vals_int < 0);
 
     ZZ *coeff_abs_vec = (ZZ*)(&coeff_abs);
     ZZ coeff_crt = barrett_reduce_64input_32modulus(coeff_abs_vec, mod);
 
     // This is the same as the following, but in constant-time:
     // ZZ val = (conj_vals_int < 0) ? (mod->value - coeff_crt) : coeff_crt;
     ZZ val = ((mod->value - coeff_crt) & (-mask)) + (coeff_crt & (mask - 1));
 
     return val;
 }
 
 extern "C" void reduce_set_pte(const Parms *parms, const int64_t *conj_vals_int, ZZ *out)
 {
     PolySizeType n = parms->coeff_count;
     Modulus *mod = parms->curr_modulus;
 
     for (size_t i = 0; i < n; i++) {
         out[i] = reduce_pte_core(conj_vals_int[i], mod);
     }
 }
 
 extern "C" void reduce_add_pte(const Parms *parms, const int64_t *conj_vals_int, ZZ *out)
 {
     PolySizeType n = parms->coeff_count;
     Modulus *mod = parms->curr_modulus;
 
     for (size_t i = 0; i < n; i++) {
         ZZ val = reduce_pte_core(conj_vals_int[i], mod);
         add_mod_inpl(&(out[i]), val, mod);
     }
 }
 
 extern "C" void reduce_set_e_small(const Parms *parms, const int8_t *e, ZZ *out)
 {
     PolySizeType n = parms->coeff_count;
     Modulus *mod = parms->curr_modulus;
 
     for (size_t i = 0; i < n; i++) {
         out[i] = ((-(ZZ)(e[i] < 0)) & mod->value) + static_cast<ZZ>(e[i]);
     }
 }
 
 extern "C" void reduce_add_e_small(const Parms *parms, const int8_t *e, ZZ *out)
 {
     PolySizeType n = parms->coeff_count;
     Modulus *mod = parms->curr_modulus;
 
     for (size_t i = 0; i < n; i++) {
         add_mod_inpl(&(out[i]), ((-(ZZ)(e[i] < 0)) & mod->value) + static_cast<ZZ>(e[i]), mod);
     }
 }
 
 #ifdef SE_USE_MALLOC
 extern "C" void se_print_relative_positions(const ZZ *st, const SE_PTRS *se_ptrs, size_t n, int sym)
 {
     printf("\n\tPrinting relative positions (negative value == does not exist)...\n");
     printf("\t    conj_vals: %0.4f\n", ((const ZZ*)(se_ptrs->conj_vals) - st) / static_cast<double>(n));
     printf("\tconj_vals_int: %0.4f\n", ((const ZZ*)(se_ptrs->conj_vals_int_ptr) - st) / static_cast<double>(n));
     printf("\t           c1: %0.4f\n", (se_ptrs->c1_ptr - st) / static_cast<double>(n));
     printf("\t           c0: %0.4f\n", (se_ptrs->c0_ptr - st) / static_cast<double>(n));
     printf("\t      ntt_pte: %0.4f\n", (se_ptrs->ntt_pte_ptr - st) / static_cast<double>(n));
     printf("\t   ifft_roots: %0.4f\n", ((const ZZ*)(se_ptrs->ifft_roots) - st) / static_cast<double>(n));
     printf("\t    ntt_roots: %0.4f\n", (se_ptrs->ntt_roots_ptr - st) / static_cast<double>(n));
     printf("\t    index_map: %0.4f\n", ((const ZZ*)(se_ptrs->index_map_ptr) - st) / static_cast<double>(n));
     if (!sym) {
         printf("\t           e1: %0.4f\n", ((const ZZ*)(se_ptrs->e1_ptr) - st) / static_cast<double>(n));
     }
     printf("\t      ternary: %0.4f\n", (se_ptrs->ternary - st) / static_cast<double>(n));
     printf("\t       values: %0.4f\n", ((const ZZ*)(se_ptrs->values) - st) / static_cast<double>(n));
     printf("\n");
 }
 
 extern "C" void se_print_addresses(const ZZ *mempool, const SE_PTRS *se_ptrs, size_t n, int sym)
 {
     size_t mempool_size = sym ? ckks_get_mempool_size_sym(n) : ckks_get_mempool_size_asym(n);
 
     printf("\n\tPrinting addresses (nil == does not exist)...\n");
     printf("mempool begin address: %p\n", mempool);
     printf("mempool end   address: %p\n", &(mempool[mempool_size - 1]));
     printf("\t    conj_vals: %p\n", se_ptrs->conj_vals);
     printf("\tconj_vals_int: %p\n", se_ptrs->conj_vals_int_ptr);
     printf("\t           c1: %p\n", se_ptrs->c1_ptr);
     printf("\t           c0: %p\n", se_ptrs->c0_ptr);
     printf("\t      ntt_pte: %p\n", se_ptrs->ntt_pte_ptr);
     printf("\t   ifft_roots: %p\n", se_ptrs->ifft_roots);
     printf("\t    ntt_roots: %p\n", se_ptrs->ntt_roots_ptr);
     printf("\t    index_map: %p\n", se_ptrs->index_map_ptr);
     if (!sym) printf("\t           e1: %p\n", se_ptrs->e1_ptr);
     printf("\t      ternary: %p\n", se_ptrs->ternary);
     printf("\t       values: %p\n", se_ptrs->values);
     printf("\n");
 }
 
 extern "C" void print_ckks_mempool_size(size_t n, int sym)
 {
     se_assert(n >= 16);
     size_t mempool_size = sym ? ckks_get_mempool_size_sym(n) : ckks_get_mempool_size_asym(n);
     se_assert(mempool_size);
 
     size_t n_size_B = n * sizeof(ZZ);
     size_t n_size_KB = n * sizeof(ZZ) / 1024;
 
     bool alloc_values = 1;
 
     const char *print_str1 = "\nTotal memory requirement (incl. values buffer)  :";
     const char *print_str2 = "\nTotal memory requirement (without values buffer):";
     const char *print_str_curr = alloc_values ? print_str1 : print_str2;
 
     for (size_t i = 0; i < 1 + static_cast<size_t>(alloc_values); i++) {
         size_t mempool_size_B = mempool_size * sizeof(ZZ);
         size_t mempool_size_KB = mempool_size_B / 1024;
 
         if (mempool_size_KB)
             printf("%s %zu KB\n", print_str_curr, mempool_size_KB);
         else
             printf("%s %zu bytes\n", print_str_curr, mempool_size_B);
 
         printf("\t( i.e. [(degree = %zu) * (sizeof(ZZ) = %zu bytes) = ", n, sizeof(ZZ));
 
         if (n_size_KB)
             printf("%zu KB] * %0.4f )\n\n", n_size_KB, mempool_size / static_cast<double>(n));
         else
             printf("%zu bytes] * %0.4f )\n\n", n_size_B, mempool_size / static_cast<double>(n));
 
         mempool_size -= n / 2;
         print_str_curr = print_str2;
     }
 }
 #endif