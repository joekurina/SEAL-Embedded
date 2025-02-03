/**
 * @file main.cpp
 * C++ test harness maintaining C compatibility with SEAL-Embedded
 */

//––– Prevent inclusion of the problematic headers –––
// These macros (which should match the include guard names in the C headers)
// cause the headers to be skipped in this translation unit.
#define UTIL_PRINT_H
#define FIPS202_H

#ifndef SE_DISABLE_TESTING_CAPABILITY

// Now include the rest of your project headers. Because the include guards
// for util_print.h and fips202.h are already defined, their contents will not
// be re-included here.
extern "C" {
#include "defines.h"
// (Note: if defines.h itself includes util_print.h or fips202.h, those inclusions
// will be skipped because of the macros above.)
}
// Test function declarations in C linkage block
extern "C" {
    //extern void test_add_uint(void);
    //extern void test_mult_uint(void);
    //extern void test_add_mod(void);
    //extern void test_neg_mod(void);
    //extern void test_mul_mod(void);
    //extern void test_sample_poly_uniform(size_t n);
    //extern void test_sample_poly_ternary(size_t n);
    //extern void test_sample_poly_ternary_small(size_t n);
    //extern void test_barrett_reduce(void);
    //extern void test_barrett_reduce_wide(void);
    //extern void test_poly_mult_ntt(size_t n, size_t nprimes);
    extern void test_fft(size_t n);
    //extern void test_enc_zero_sym(size_t n, size_t nprimes);
    //extern void test_enc_zero_asym(size_t n, size_t nprimes);
    //extern void test_ckks_encode(size_t n);
    //extern void test_ckks_encode_encrypt_sym(size_t n, size_t nprimes);
    //extern void test_ckks_encode_encrypt_asym(size_t n, size_t nprimes);
    //extern void test_ckks_api_sym(void);
    //extern void test_ckks_api_asym(void);
    //extern int test_network_basic(void);
    //extern void test_network(void);
}

int main()
{
    printf("Beginning tests...\n");

#ifdef SE_USE_MALLOC
    const size_t n = 4096, nprimes = 3;
    // Alternate configurations:
    // const size_t n =  1024, nprimes = 1;
    // const size_t n =  2048, nprimes = 1;
    // const size_t n =  8192, nprimes = 6;
    // const size_t n = 16384, nprimes = 13;
#else
    const size_t n       = SE_DEGREE_N;
    const size_t nprimes = SE_NPRIMES;
#endif

    //test_sample_poly_uniform(n);
    //test_sample_poly_ternary(n);
    //test_sample_poly_ternary_small(n);  // Only useful when SE_USE_MALLOC is defined

    //test_add_uint();
    //test_mult_uint();
    //test_barrett_reduce();
    //test_barrett_reduce_wide();
    //test_add_mod();
    //test_neg_mod();
    //test_mul_mod();

    // Note: This test uses schoolbook multiplication (slow)
    // test_poly_mult_ntt(n, nprimes);

    test_fft(n);
    //test_enc_zero_sym(n, nprimes);
    //test_enc_zero_asym(n, nprimes);
    //test_ckks_encode(n);

    // Main tests
    //test_ckks_encode_encrypt_sym(n, nprimes);
    //test_ckks_encode_encrypt_asym(n, nprimes);

    // Optional API verification tests
    // test_ckks_api_sym();
    // test_ckks_api_asym();
    // test_network_basic();
    // test_network();

    printf("...done with all tests. All tests passed.\n");
    return 0;
}

#endif