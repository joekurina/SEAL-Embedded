/**
@file fft_tests.cpp
*/

//––– Prevent inclusion of the problematic headers –––
// These macros (which should match the include guard names in the C headers)
// cause the headers to be skipped in this translation unit.
#define UTIL_PRINT_H
#define FIPS202_H

#include <complex>       // std::complex
#include <cmath>        // std::log2
#include <cstring>      // std::memcpy
#include <cstdint>      // fixed-width integers
#include <memory>       // smart pointers
#include <vector>       // std::vector

extern "C" {
#include "defines.h"
#include "fft.h"
#include "fileops.h"
#include "test_common.h"
#include "util_print.h"
}

#ifndef SE_USE_MALLOC
#if defined(SE_IFFT_LOAD_FULL) || defined(SE_IFFT_ONE_SHOT)
#define IFFT_TEST_ROOTS_MEM SE_DEGREE_N
#else
#define IFFT_TEST_ROOTS_MEM 0
#endif
#if defined(SE_FFT_LOAD_FULL) || defined(SE_FFT_ONE_SHOT)
#define FFT_TEST_ROOTS_MEM SE_DEGREE_N
#else
#define FFT_TEST_ROOTS_MEM 0
#endif
#endif

/**
Multiplies two complex-valued polynomials using schoolbook multiplication.

Space req: 'res' must constain space for n double complex elements, where each input polynomial
consists of n double complex elements (each).

@param[in]  a    Input polynomial 1
@param[in]  b    Input polynomial 2
@param[in]  n    Number of coefficients to multiply
@param[out] res  Result of [a * b]
*/
// Forward declare C-compatible version
extern "C" void poly_mult_sb_complex(const double complex *a, 
                                   const double complex *b,
                                   PolySizeType n,
                                   double complex *res);

// C++ implementation
template<typename T = double>
void poly_mult_sb_complex_cpp(const std::complex<T>* a,
                            const std::complex<T>* b,
                            const PolySizeType n,
                            std::complex<T>* res) 
{
    for (PolySizeType i = 0; i < n; i++) {
        for (PolySizeType j = 0; j < n; j++) {
            res[i + j] += a[i] * b[j];
        }
    }
}

// C-compatible wrapper 
extern "C" void poly_mult_sb_complex(const double complex *a,
                                   const double complex *b,
                                   PolySizeType n, 
                                   double complex *res)
{
    // Convert C complex to C++ complex
    auto a_cpp = reinterpret_cast<const std::complex<double>*>(a);
    auto b_cpp = reinterpret_cast<const std::complex<double>*>(b);
    auto res_cpp = reinterpret_cast<std::complex<double>*>(res);

    // Call C++ implementation
    poly_mult_sb_complex_cpp(a_cpp, b_cpp, n, res_cpp);
}

/**
In-place divides each element of a polynomial by a divisor.

@param[in,out] poly     In: Input polynomial; Out: Result polynomial
@param[in]     n        Number of coefficients to multiply
@param[in]     divisor  Divisor value
*/
// C++ implementation
template<typename T = double>
static inline void poly_div_inpl_complex_cpp(
    std::complex<T>* poly,
    std::size_t n, 
    std::size_t divisor)
{
    if (divisor == 0) throw std::invalid_argument("Division by zero");
    for (std::size_t i = 0; i < n; i++) {
        poly[i] /= static_cast<T>(divisor);
    }
}

// C interface wrapper
static inline void poly_div_inpl_complex(
    double complex* poly,
    size_t n,
    size_t divisor) 
{
    auto poly_cpp = reinterpret_cast<std::complex<double>*>(poly);
    poly_div_inpl_complex_cpp(poly_cpp, n, divisor);
}

/**
Pointwise multiplies two complex-valued polynomials. 'a' and 'res' starting address may overlap for
in-place computation (see: pointwise_mult_inpl_complex)

Space req: 'res' must constain space for n double complex elements, where each input polynomial
consists of n double complex elements (each).

@param[in]  a    Input polynomial 1
@param[in]  b    Input polynomial 2
@param[in]  n    Number of coefficients to multiply
@param[out] res  Result of [a . b]
*/
// C++ implementation
template<typename T = double>
static inline void pointwise_mult_complex_cpp(
    const std::complex<T>* a,
    const std::complex<T>* b,
    const PolySizeType n,
    std::complex<T>* res)
{
    for (std::size_t i = 0; i < n; i++) {
        res[i] = a[i] * b[i];
    }
}

// C interface wrapper
static inline void pointwise_mult_complex(
    const double complex* a,
    const double complex* b,
    PolySizeType n,
    double complex* res)
{
    auto a_cpp = reinterpret_cast<const std::complex<double>*>(a);
    auto b_cpp = reinterpret_cast<const std::complex<double>*>(b);
    auto res_cpp = reinterpret_cast<std::complex<double>*>(res);
    
    pointwise_mult_complex_cpp(a_cpp, b_cpp, n, res_cpp);
}

/**
In-place pointwise multiplies two complex-valued polynomials.

@param[in]  a    Input polynomial 1
@param[in]  b    Input polynomial 2
@param[in]  n    Number of coefficients to multiply
@param[out] res  Result of [a . b]
*/
// C++ implementation
template<typename T = double>
static inline void pointwise_mult_inpl_complex_cpp(
    std::complex<T>* a,
    const std::complex<T>* b,
    std::size_t n)
{
    for (std::size_t i = 0; i < n; i++) {
        a[i] *= b[i];
    }
}

// C interface wrapper
static inline void pointwise_mult_inpl_complex(
    double complex* a,
    const double complex* b,
    PolySizeType n)
{
    auto a_cpp = reinterpret_cast<std::complex<double>*>(a);
    auto b_cpp = reinterpret_cast<const std::complex<double>*>(b);
    
    pointwise_mult_inpl_complex_cpp(a_cpp, b_cpp, n);
}

/**
Helper function for FFT test
*/
// C++ implementation
template<typename T = double>
void test_fft_mult_helper_cpp(
    std::size_t n, 
    std::complex<T>* v1,
    std::complex<T>* v2,
    std::complex<T>* v_exp,
    std::complex<T>* temp,
    std::complex<T>* roots)
{
    std::size_t logn = static_cast<std::size_t>(std::log2(n));

    // Correctness: ifft(fft(vec) .* fft(v2))*(1/n) = vec * vec2
    print_poly_double_complex("v1              ", 
        reinterpret_cast<double complex*>(v1), n);
    print_poly_double_complex("v2              ", 
        reinterpret_cast<double complex*>(v2), n);

    auto v_res = temp;
    std::memset(v_res, 0, n * sizeof(std::complex<T>));

    poly_mult_sb_complex_cpp(v1, v2, n / 2, v_res);
    print_poly_double_complex("vec_res (expected)", 
        reinterpret_cast<double complex*>(v_res), n);

#ifdef SE_FFT_LOAD_FULL
    se_assert(roots);
    load_fft_roots(n, reinterpret_cast<double complex*>(roots));
#elif defined(SE_FFT_ONE_SHOT)
    se_assert(roots);
    calc_fft_roots(n, logn, reinterpret_cast<double complex*>(roots));
#endif

    fft_inpl(reinterpret_cast<double complex*>(v1), n, logn, 
             reinterpret_cast<double complex*>(roots));
    fft_inpl(reinterpret_cast<double complex*>(v2), n, logn,
             reinterpret_cast<double complex*>(roots));

    pointwise_mult_inpl_complex_cpp(v1, v2, n);

#ifdef SE_IFFT_LOAD_FULL
    se_assert(roots);
    load_ifft_roots(n, reinterpret_cast<double complex*>(roots));
#elif defined(SE_IFFT_ONE_SHOT)
    se_assert(roots);
    calc_ifft_roots(n, logn, reinterpret_cast<double complex*>(roots));
#endif

    ifft_inpl(reinterpret_cast<double complex*>(v1), n, logn,
              reinterpret_cast<double complex*>(roots));

    poly_div_inpl_complex_cpp(v1, n, n);
    print_poly_double_complex("vec_res (actual)  ",
        reinterpret_cast<double complex*>(v1), n);

    constexpr double maxdiff = 0.0001;
    bool err = compare_poly_double_complex(
        reinterpret_cast<double complex*>(v1),
        reinterpret_cast<double complex*>(v_res),
        n, maxdiff);
    se_assert(!err);

    if (v_exp) {
        print_poly_double_complex("v_exp2          ",
            reinterpret_cast<double complex*>(v_exp), n);
        err = compare_poly_double_complex(
            reinterpret_cast<double complex*>(v_exp),
            reinterpret_cast<double complex*>(v_res),
            n, maxdiff);
        se_assert(!err);
    }
}

// C interface wrapper
extern "C" void test_fft_mult_helper(
    size_t n,
    double complex* v1,
    double complex* v2,
    double complex* v_exp,
    double complex* temp,
    double complex* roots)
{
    auto v1_cpp = reinterpret_cast<std::complex<double>*>(v1);
    auto v2_cpp = reinterpret_cast<std::complex<double>*>(v2);
    auto v_exp_cpp = reinterpret_cast<std::complex<double>*>(v_exp);
    auto temp_cpp = reinterpret_cast<std::complex<double>*>(temp);
    auto roots_cpp = reinterpret_cast<std::complex<double>*>(roots);

    test_fft_mult_helper_cpp(n, v1_cpp, v2_cpp, v_exp_cpp, temp_cpp, roots_cpp);
}

// C++ implementation
template<typename T = double>
void test_fft_helper_cpp(
    std::size_t degree, 
    const std::complex<T>* v,
    std::complex<T>* temp,
    std::complex<T>* roots)
{
    const std::size_t n = degree;
    const std::size_t logn = static_cast<std::size_t>(std::log2(degree));

    // Save vec for comparison later. Write to temp to apply fft/ifft in-place
    print_poly_double_complex("vec               ", 
        reinterpret_cast<const double complex*>(v), n);
    
    auto v_fft = temp;
    std::memcpy(v_fft, v, n * sizeof(std::complex<T>));
    
    print_poly_double_complex("vec               ",
        reinterpret_cast<const double complex*>(v_fft), n);

#ifdef SE_FFT_LOAD_FULL
    se_assert(roots);
    load_fft_roots(n, reinterpret_cast<double complex*>(roots));
#elif defined(SE_FFT_ONE_SHOT)
    se_assert(roots);
    calc_fft_roots(n, logn, reinterpret_cast<double complex*>(roots));
#endif

    fft_inpl(reinterpret_cast<double complex*>(v_fft), n, logn,
             reinterpret_cast<double complex*>(roots));
             
    print_poly_double_complex("vec (after fft)   ",
        reinterpret_cast<const double complex*>(v), n);

#ifdef SE_IFFT_LOAD_FULL
    se_assert(roots);
    load_ifft_roots(n, reinterpret_cast<double complex*>(roots));
    print_poly_double_complex("roots               ",
        reinterpret_cast<const double complex*>(roots), n);
#elif defined(SE_IFFT_ONE_SHOT)
    se_assert(roots);
    calc_ifft_roots(n, logn, reinterpret_cast<double complex*>(roots));
#endif

    ifft_inpl(reinterpret_cast<double complex*>(v_fft), n, logn,
              reinterpret_cast<double complex*>(roots));
              
    print_poly_double_complex("vec (after ifft)  ",
        reinterpret_cast<const double complex*>(v_fft), n);

    poly_div_inpl_complex_cpp(v_fft, n, n);
    
    print_poly_double_complex("vec (after *(1/n))",
        reinterpret_cast<const double complex*>(v_fft), n);

    constexpr double maxdiff = 0.0001;
    bool err = compare_poly_double_complex(
        reinterpret_cast<const double complex*>(v_fft),
        reinterpret_cast<const double complex*>(v),
        n, maxdiff);
    se_assert(!err);
}

// C interface wrapper
extern "C" void test_fft_helper(
    size_t degree,
    const double complex* v,
    double complex* temp,
    double complex* roots)
{
    auto v_cpp = reinterpret_cast<const std::complex<double>*>(v);
    auto temp_cpp = reinterpret_cast<std::complex<double>*>(temp);
    auto roots_cpp = reinterpret_cast<std::complex<double>*>(roots);
    
    test_fft_helper_cpp(degree, v_cpp, temp_cpp, roots_cpp);
}

/**
FFT test function

@param[in] n  Polynomial ring degree (ignored if SE_USE_MALLOC is defined)
*/
extern "C" void test_fft(size_t n)
{
#ifndef SE_USE_MALLOC
    se_assert(n == SE_DEGREE_N);
    if (n != SE_DEGREE_N) n = SE_DEGREE_N;
#endif

    // Calculate memory requirements
    std::size_t ifft_roots_size = 0;
    std::size_t fft_roots_size = 0;

#ifdef SE_USE_MALLOC
    #if defined(SE_IFFT_LOAD_FULL) || defined(SE_IFFT_ONE_SHOT)
        ifft_roots_size = n;
    #endif
    #if defined(SE_FFT_LOAD_FULL) || defined(SE_FFT_ONE_SHOT)
        fft_roots_size = n;
    #endif
#else
    ifft_roots_size = IFFT_TEST_ROOTS_MEM;
    fft_roots_size = FFT_TEST_ROOTS_MEM; 
#endif

    std::size_t roots_size = ifft_roots_size ? ifft_roots_size : fft_roots_size;
    std::size_t mempool_size = 4*n + roots_size;

    // Allocate memory pool using vector
    std::vector<std::complex<double>> mempool(mempool_size);
    std::fill(mempool.begin(), mempool.end(), std::complex<double>{0.0, 0.0});

    // Set up pointers into memory pool
    auto v1 = reinterpret_cast<double complex*>(&mempool[0]);
    auto v2 = reinterpret_cast<double complex*>(&mempool[n]);
    auto v_exp = reinterpret_cast<double complex*>(&mempool[2*n]);
    auto temp = reinterpret_cast<double complex*>(&mempool[3*n]);
    auto roots = roots_size ? reinterpret_cast<double complex*>(&mempool[4*n]) : nullptr;

    Parms parms;
    set_parms_ckks(n, 1, &parms);
    print_test_banner("fft/ifft", &parms);

    for (std::size_t testnum = 0; testnum < 15; ++testnum) {
        std::printf("\n--------------- Test: %zu -----------------\n", testnum);
        std::fill(mempool.begin(), mempool.end(), std::complex<double>{0.0, 0.0});

        switch (testnum) {
            case 0: set_double_complex(v1, n, 1); break;
            case 1: set_double_complex(v1, n, 2); break;
            case 2:
                for (std::size_t i = 0; i < n; ++i) {
                    v1[i] = static_cast<double complex>(_complex(static_cast<double>(i), 0.0));
                }
                break;
            case 3:
                for (size_t i = 0; i < n; i++)
                { v1[i] = (double complex)(gen_double_eighth(10), (double)0); }
                break;
            case 4:
                for (size_t i = 0; i < n; i++)
                { v1[i] = (double complex)(gen_double_quarter(100), (double)0); }
                break;
            case 5:
                for (size_t i = 0; i < n; i++)
                { v1[i] = (double complex)(gen_double_half(-100), (double)0); }
                break;
            case 6:
                for (size_t i = 0; i < n; i++)
                { v1[i] = (double complex)(gen_double(1000), (double)0); }
                break;
            case 7:  // {1, 0, 0, ...} * {2, 2, 2, ...} = {2, 2, 2, ..., 0, 0, ...}
                v1[0] = (double complex)_complex((double)1, (double)0);
                set_double_complex(v2, n / 2, 2);
                set_double_complex(v_exp, n / 2, 2);
                break;
            case 8:  // {-1, 0, 0, ...} * {2, 2, 2, ...} = {-2, -2, -2, ..., 0, 0,
                     // ...}
                v1[0] = (double complex)_complex((double)-1, (double)0);
                set_double_complex(v2, n / 2, (flpt)2);
                set_double_complex(v_exp, n / 2, (flpt)(-2));
                break;
            case 9:  // {1, 0, 0, ...} * {-2, -2, -2, ...} = {-2, -2, -2, ..., 0, 0,
                     // ...}
                v1[0] = (double complex)_complex((double)1, (double)0);
                set_double_complex(v2, n / 2, (flpt)(-2));
                set_double_complex(v_exp, n / 2, (flpt)(-2));
                break;
            case 10:  // {1, 1, 1, ...} * {2, 2, 2, ...} = {2, 4, 8, ..., 4, 2, 0}
                set_double_complex(v1, n / 2, (flpt)1);
                set_double_complex(v2, n / 2, (flpt)2);
                for (size_t i = 0; i < n / 2; i++)
                { v_exp[i] = (double complex)_complex((double)(2 * (i + 1)), (double)0); }
                for (size_t i = 0; i < (n / 2) - 1; i++)
                { v_exp[i + n / 2] = v_exp[n / 2 - (i + 2)]; }
                break;
            case 11:
                for (size_t i = 0; i < n / 2; i++)
                {
                    v1[i] = (double complex)_complex(gen_double_eighth(pow(10, 1)), (double)0);
                    v2[i] = (double complex)_complex(gen_double_eighth(pow(10, 1)), (double)0);
                }
                break;
            case 12:
                for (size_t i = 0; i < n / 2; i++)
                {
                    v1[i] = (double complex)_complex(gen_double_quarter(-pow(10, 2)), (double)0);
                    v2[i] = (double complex)_complex(gen_double_quarter(-pow(10, 2)), (double)0);
                }
                break;
            case 13:
                for (size_t i = 0; i < n / 2; i++)
                {
                    v1[i] = (double complex)(gen_double_half(pow(10, 3)), (double)0);
                    v2[i] = (double complex)(gen_double_half(pow(10, 3)), (double)0);
                }
                break;
            case 14:
                for (size_t i = 0; i < n / 2; i++)
                {
                    v1[i] = (double complex)(gen_double(pow(10, 6)), (double)0);
                    v2[i] = (double complex)(gen_double(pow(10, 6)), (double)0);
                }
                break;
        }
        if (testnum < 7)
            test_fft_helper(n, v1, temp, roots);
        else if (testnum < 11)
            test_fft_mult_helper(n, v1, v2, v_exp, temp, roots);
        else
            test_fft_mult_helper(n, v1, v2, nullptr, temp, roots);
    }
}

#ifndef SE_USE_MALLOC
#ifdef IFFT_TEST_ROOTS_MEM
#undef IFFT_TEST_ROOTS_MEM
#endif
#ifdef FFT_TEST_ROOTS_MEM
#undef FFT_TEST_ROOTS_MEM
#endif
#endif
