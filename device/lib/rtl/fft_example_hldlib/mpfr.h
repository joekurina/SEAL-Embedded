/* mpfr.h -- Include file for mpfr.

Copyright 1999-2020 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
https://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#ifndef __MPFR_H
#define __MPFR_H

/* Define MPFR version number */
#define MPFR_VERSION_MAJOR 4
#define MPFR_VERSION_MINOR 1
#define MPFR_VERSION_PATCHLEVEL 0
#define MPFR_VERSION_STRING "4.1.0"

/* User macros:
   MPFR_USE_FILE:        Define it to make MPFR define functions dealing
                         with FILE* (auto-detect).
   MPFR_USE_INTMAX_T:    Define it to make MPFR define functions dealing
                         with intmax_t (auto-detect).
   MPFR_USE_VA_LIST:     Define it to make MPFR define functions dealing
                         with va_list (auto-detect).
   MPFR_USE_C99_FEATURE: Define it to 1 to make MPFR support C99-feature
                         (auto-detect), to 0 to bypass the detection.
   MPFR_USE_EXTENSION:   Define it to make MPFR use GCC extension to
                         reduce warnings.
   MPFR_USE_NO_MACRO:    Define it to make MPFR remove any overriding
                         function macro.
*/

/* Macros dealing with MPFR VERSION */
#define MPFR_VERSION_NUM(a,b,c) (((a) << 16L) | ((b) << 8) | (c))
#define MPFR_VERSION \
MPFR_VERSION_NUM(MPFR_VERSION_MAJOR,MPFR_VERSION_MINOR,MPFR_VERSION_PATCHLEVEL)

#ifndef MPFR_USE_MINI_GMP
#include <gmp.h>
#else
#include <mini-gmp.h>
#endif

/* Avoid some problems with macro expansion if the user defines macros
   with the same name as keywords. By convention, identifiers and macro
   names starting with dspba_mpfr_ are reserved by MPFR. */
typedef void            mpfr_void;
typedef int             mpfr_int;
typedef unsigned int    mpfr_uint;
typedef long            mpfr_long;
typedef unsigned long   mpfr_ulong;
typedef size_t          mpfr_size_t;

/* Global (possibly TLS) flags. Might also be used in an mpfr_t in the
   future (there would be room as mpfr_sign_t just needs 1 byte).
   TODO: The tests currently assume that the flags fits in an unsigned int;
   this should be cleaned up, e.g. by defining a function that outputs the
   flags as a string or by using the flags_out function (from tests/tests.c
   directly). */
typedef unsigned int    mpfr_flags_t;

/* Flags macros (in the public API) */
#define MPFR_FLAGS_UNDERFLOW 1
#define MPFR_FLAGS_OVERFLOW 2
#define MPFR_FLAGS_NAN 4
#define MPFR_FLAGS_INEXACT 8
#define MPFR_FLAGS_ERANGE 16
#define MPFR_FLAGS_DIVBY0 32
#define MPFR_FLAGS_ALL (MPFR_FLAGS_UNDERFLOW | \
                        MPFR_FLAGS_OVERFLOW  | \
                        MPFR_FLAGS_NAN       | \
                        MPFR_FLAGS_INEXACT   | \
                        MPFR_FLAGS_ERANGE    | \
                        MPFR_FLAGS_DIVBY0)

/* Definition of rounding modes (DON'T USE MPFR_RNDNA!).
   Warning! Changing the contents of this enum should be seen as an
   interface change since the old and the new types are not compatible
   (the integer type compatible with the enumerated type can even change,
   see ISO C99, 6.7.2.2#4), and in Makefile.am, AGE should be set to 0.

   MPFR_RNDU must appear just before MPFR_RNDD (see
   MPFR_IS_RNDUTEST_OR_RNDDNOTTEST in mpfr-impl.h).

   If you change the order of the rounding modes, please update the routines
   in texceptions.c which assume 0=RNDN, 1=RNDZ, 2=RNDU, 3=RNDD, 4=RNDA.
*/
typedef enum {
  MPFR_RNDN=0,  /* round to nearest, with ties to even */
  MPFR_RNDZ,    /* round toward zero */
  MPFR_RNDU,    /* round toward +Inf */
  MPFR_RNDD,    /* round toward -Inf */
  MPFR_RNDA,    /* round away from zero */
  MPFR_RNDF,    /* faithful rounding */
  MPFR_RNDNA=-1 /* round to nearest, with ties away from zero (dspba_mpfr_round) */
} mpfr_rnd_t;

/* kept for compatibility with MPFR 2.4.x and before */
#define GMP_RNDN MPFR_RNDN
#define GMP_RNDZ MPFR_RNDZ
#define GMP_RNDU MPFR_RNDU
#define GMP_RNDD MPFR_RNDD

/* The _MPFR_PREC_FORMAT and _MPFR_EXP_FORMAT values are automatically
   defined below. You MUST NOT force a value (this will break the ABI),
   possibly except for a very particular use, in which case you also need
   to rebuild the MPFR library with the chosen values; do not install this
   rebuilt library in a path that is searched by default, otherwise this
   will break applications that are dynamically linked with MPFR.

   Using non-default values is not guaranteed to work.

   Note: With the following default choices for _MPFR_PREC_FORMAT and
   _MPFR_EXP_FORMAT, mpfr_exp_t will be the same as [mp_exp_t] (at least
   up to GMP 6). */

/* Define precision: 1 (short), 2 (int) or 3 (long).
   DON'T FORCE A VALUE (see above). */
#ifndef _MPFR_PREC_FORMAT
# if __GMP_MP_SIZE_T_INT
#  define _MPFR_PREC_FORMAT 2
# else
#  define _MPFR_PREC_FORMAT 3
# endif
#endif

/* Define exponent: 1 (short), 2 (int), 3 (long) or 4 (intmax_t).
   DON'T FORCE A VALUE (see above). */
#ifndef _MPFR_EXP_FORMAT
# define _MPFR_EXP_FORMAT _MPFR_PREC_FORMAT
#endif

#if _MPFR_PREC_FORMAT > _MPFR_EXP_FORMAT
# error "mpfr_prec_t must not be larger than mpfr_exp_t"
#endif

/* Let's make mpfr_prec_t signed in order to avoid problems due to the
   usual arithmetic conversions when mixing mpfr_prec_t and mpfr_exp_t
   in an expression (for error analysis) if casts are forgotten.
   Note: mpfr_prec_t is currently limited to "long". This means that
   under MS Windows, the precisions are limited to about 2^31; however,
   these are already huge precisions, probably sufficient in practice
   on this platform. */
#if   _MPFR_PREC_FORMAT == 1
typedef short mpfr_prec_t;
typedef unsigned short mpfr_uprec_t;
#elif _MPFR_PREC_FORMAT == 2
typedef int   mpfr_prec_t;
typedef unsigned int   mpfr_uprec_t;
#elif _MPFR_PREC_FORMAT == 3
/* we could use "long long" under Windows 64 here, which can be tested
   with the macro _WIN64 according to
   https://sourceforge.net/p/predef/wiki/OperatingSystems/ */
typedef long  mpfr_prec_t;
typedef unsigned long  mpfr_uprec_t;
#else
# error "Invalid MPFR Prec format"
#endif

/* Definition of precision limits without needing <limits.h> */
/* Note: The casts allows the expression to yield the wanted behavior
   for _MPFR_PREC_FORMAT == 1 (due to integer promotion rules). We
   also make sure that MPFR_PREC_MIN and MPFR_PREC_MAX have a signed
   integer type. The "- 256" allows more security, avoiding some
   integer overflows in extreme cases; ideally it should be useless. */
#define MPFR_PREC_MIN 1
#define MPFR_PREC_MAX ((mpfr_prec_t) ((((mpfr_uprec_t) -1) >> 1) - 256))

/* Definition of sign */
typedef int          mpfr_sign_t;

/* Definition of the exponent. _MPFR_EXP_FORMAT must be large enough
   so that mpfr_exp_t has at least 32 bits. */
#if   _MPFR_EXP_FORMAT == 1
typedef short mpfr_exp_t;
typedef unsigned short mpfr_uexp_t;
#elif _MPFR_EXP_FORMAT == 2
typedef int mpfr_exp_t;
typedef unsigned int mpfr_uexp_t;
#elif _MPFR_EXP_FORMAT == 3
typedef long mpfr_exp_t;
typedef unsigned long mpfr_uexp_t;
#elif _MPFR_EXP_FORMAT == 4
/* Note: in this case, intmax_t and uintmax_t must be defined before
   the inclusion of mpfr.h (we do not include <stdint.h> here because
   of some non-ISO C99 implementations that support these types). */
typedef intmax_t mpfr_exp_t;
typedef uintmax_t mpfr_uexp_t;
#else
# error "Invalid MPFR Exp format"
#endif

/* Definition of the standard exponent limits */
#define MPFR_EMAX_DEFAULT ((mpfr_exp_t) (((mpfr_ulong) 1 << 30) - 1))
#define MPFR_EMIN_DEFAULT (-(MPFR_EMAX_DEFAULT))

/* DON'T USE THIS! (For MPFR-public macros only, see below.)
   The dspba_mpfr_sgn macro uses the fact that __MPFR_EXP_NAN and __MPFR_EXP_ZERO
   are the smallest values. For a n-bit type, EXP_MAX is 2^(n-1)-1,
   EXP_ZERO is 1-2^(n-1), EXP_NAN is 2-2^(n-1), EXP_INF is 3-2^(n-1).
   This may change in the future. MPFR code should not be based on these
   representations (but if this is absolutely needed, protect the code
   with a static assertion). */
#define __MPFR_EXP_MAX ((mpfr_exp_t) (((mpfr_uexp_t) -1) >> 1))
#define __MPFR_EXP_NAN  (1 - __MPFR_EXP_MAX)
#define __MPFR_EXP_ZERO (0 - __MPFR_EXP_MAX)
#define __MPFR_EXP_INF  (2 - __MPFR_EXP_MAX)

/* Definition of the main structure */
typedef struct {
  mpfr_prec_t  _mpfr_prec;
  mpfr_sign_t  _mpfr_sign;
  mpfr_exp_t   _mpfr_exp;
  mp_limb_t   *_mpfr_d;
} __dspba_mpfr_struct;

/* Compatibility with previous types of MPFR */
#ifndef mp_rnd_t
# define mp_rnd_t  mpfr_rnd_t
#endif
#ifndef mp_prec_t
# define mp_prec_t mpfr_prec_t
#endif

/*
   The represented number is
      _sign*(_d[k-1]/B+_d[k-2]/B^2+...+_d[0]/B^k)*2^_exp
   where k=ceil(_mp_prec/GMP_NUMB_BITS) and B=2^GMP_NUMB_BITS.

   For the msb (most significant bit) normalized representation, we must have
      _d[k-1]>=B/2, unless the number is singular.

   We must also have the last k*GMP_NUMB_BITS-_prec bits set to zero.
*/

typedef __dspba_mpfr_struct mpfr_t[1];
typedef __dspba_mpfr_struct *mpfr_ptr;
typedef const __dspba_mpfr_struct *mpfr_srcptr;

/* For those who need a direct and fast access to the sign field.
   However it is not in the API, thus use it at your own risk: it might
   not be supported, or change name, in further versions!
   Unfortunately, it must be defined here (instead of MPFR's internal
   header file mpfr-impl.h) because it is used by some macros below.
*/
#define MPFR_SIGN(x) ((x)->_mpfr_sign)

/* Stack interface */
typedef enum {
  MPFR_NAN_KIND     = 0,
  MPFR_INF_KIND     = 1,
  MPFR_ZERO_KIND    = 2,
  MPFR_REGULAR_KIND = 3
} mpfr_kind_t;

/* Free cache policy */
typedef enum {
  MPFR_FREE_LOCAL_CACHE  = 1,  /* 1 << 0 */
  MPFR_FREE_GLOBAL_CACHE = 2   /* 1 << 1 */
} mpfr_free_cache_t;

/* GMP defines:
    + size_t:                Standard size_t
    + __GMP_NOTHROW          For C++: can't throw .
    + __GMP_EXTERN_INLINE    Attribute for inline function.
    + __GMP_DECLSPEC_EXPORT  compiling to go into a DLL
    + __GMP_DECLSPEC_IMPORT  compiling to go into a application
*/
/* Extra MPFR defines */
#define __MPFR_SENTINEL_ATTR
#if defined (__GNUC__)
# if __GNUC__ >= 4
#  undef __MPFR_SENTINEL_ATTR
#  define __MPFR_SENTINEL_ATTR __attribute__ ((__sentinel__))
# endif
#endif

/* If the user hasn't requested his/her preference
   and if the intention of support by the compiler is C99
   and if the compiler is known to support the C99 feature
   then we can auto-detect the C99 support as OK.
   __GNUC__ is used to detect GNU-C, ICC & CLANG compilers.
   Currently we need only variadic macros, and they are present
   since GCC >= 3. We don't test library version since we don't
   use any feature present in the library too (except intmax_t,
   but they use another detection).*/
#ifndef MPFR_USE_C99_FEATURE
# if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
#  if defined (__GNUC__)
#   if __GNUC__ >= 3
#    define MPFR_USE_C99_FEATURE 1
#   endif
#  endif
# endif
# ifndef MPFR_USE_C99_FEATURE
#  define MPFR_USE_C99_FEATURE 0
# endif
#endif

/* Support for WINDOWS Dll:
   Check if we are inside a MPFR build, and if so export the functions.
   Otherwise does the same thing as GMP */
#if defined(__MPFR_WITHIN_MPFR) && __GMP_LIBGMP_DLL
# define __MPFR_DECLSPEC __GMP_DECLSPEC_EXPORT
#else
# ifndef __GMP_DECLSPEC
#  define __GMP_DECLSPEC
# endif
# define __MPFR_DECLSPEC __GMP_DECLSPEC
#endif

/* Use MPFR_DEPRECATED to mark MPFR functions, types or variables as
   deprecated. Code inspired by Apache Subversion's svn_types.h file.
   For compatibility with MSVC, MPFR_DEPRECATED must be put before
   __MPFR_DECLSPEC (not at the end of the function declaration as
   documented in the GCC manual); GCC does not seem to care.
   Moreover, in order to avoid a warning when testing such functions,
   do something like:
     +------------------------------------------
     |#ifndef _MPFR_NO_DEPRECATED_funcname
     |MPFR_DEPRECATED
     |#endif
     |__MPFR_DECLSPEC int dspba_mpfr_funcname (...);
     +------------------------------------------
   and in the corresponding test program:
     +------------------------------------------
     |#define _MPFR_NO_DEPRECATED_funcname
     |#include "mpfr-test.h"
     +------------------------------------------
*/
#if defined(__GNUC__) && \
  (__GNUC__ >= 4 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
# define MPFR_DEPRECATED __attribute__ ((__deprecated__))
#elif defined(_MSC_VER) && _MSC_VER >= 1300
# define MPFR_DEPRECATED __declspec(deprecated)
#else
# define MPFR_DEPRECATED
#endif
/* TODO: Also define MPFR_EXPERIMENTAL for experimental functions?
   See SVN_EXPERIMENTAL in Subversion 1.9+ as an example:
   __attribute__((__warning__("..."))) can be used with GCC 4.3.1+ but
   not __llvm__, and __declspec(deprecated("...")) can be used with
   MSC as above. */

#if defined(__GNUC__) && \
  (__GNUC__ >= 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 9))
# define MPFR_RETURNS_NONNULL __attribute__ ((__returns_nonnull__))
#else
# define MPFR_RETURNS_NONNULL
#endif

/* Note: In order to be declared, some functions need a specific
   system header to be included *before* "mpfr.h". If the user
   forgets to include the header, the MPFR function prototype in
   the user object file is not correct. To avoid wrong results,
   we raise a linker error in that case by changing their internal
   name in the library (prefixed by __gmpfr instead of mpfr). See
   the lines of the form "#define dspba_mpfr_xxx __gdspba_mpfr_xxx" below. */

#if defined (__cplusplus)
extern "C" {
#endif

#ifdef CSL_SYCL
#include <sycl/sycl.hpp>
#define CSL_API SYCL_EXTERNAL
#else
#define CSL_API
#endif

__MPFR_DECLSPEC MPFR_RETURNS_NONNULL const char * dspba_mpfr_get_version (void);
__MPFR_DECLSPEC MPFR_RETURNS_NONNULL const char * dspba_mpfr_get_patches (void);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_buildopt_tls_p          (void);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_buildopt_float128_p     (void);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_buildopt_decimal_p      (void);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_buildopt_gmpinternals_p (void);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_buildopt_sharedcache_p  (void);
__MPFR_DECLSPEC MPFR_RETURNS_NONNULL const char *
  dspba_mpfr_buildopt_tune_case (void);

CSL_API __MPFR_DECLSPEC mpfr_exp_t dspba_mpfr_get_emin     (void);
CSL_API __MPFR_DECLSPEC int        dspba_mpfr_set_emin     (mpfr_exp_t);
CSL_API __MPFR_DECLSPEC mpfr_exp_t dspba_mpfr_get_emin_min (void);
CSL_API __MPFR_DECLSPEC mpfr_exp_t dspba_mpfr_get_emin_max (void);
CSL_API __MPFR_DECLSPEC mpfr_exp_t dspba_mpfr_get_emax     (void);
CSL_API __MPFR_DECLSPEC int        dspba_mpfr_set_emax     (mpfr_exp_t);
CSL_API __MPFR_DECLSPEC mpfr_exp_t dspba_mpfr_get_emax_min (void);
CSL_API __MPFR_DECLSPEC mpfr_exp_t dspba_mpfr_get_emax_max (void);

CSL_API __MPFR_DECLSPEC void dspba_mpfr_set_default_rounding_mode (mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC mpfr_rnd_t dspba_mpfr_get_default_rounding_mode (void);
CSL_API __MPFR_DECLSPEC const char * dspba_mpfr_print_rnd_mode (mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC void dspba_mpfr_clear_flags (void);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_clear_underflow (void);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_clear_overflow (void);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_clear_divby0 (void);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_clear_nanflag (void);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_clear_inexflag (void);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_clear_erangeflag (void);

CSL_API __MPFR_DECLSPEC void dspba_mpfr_set_underflow (void);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_set_overflow (void);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_set_divby0 (void);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_set_nanflag (void);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_set_inexflag (void);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_set_erangeflag (void);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_underflow_p (void);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_overflow_p (void);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_divby0_p (void);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_nanflag_p (void);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_inexflag_p (void);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_erangeflag_p (void);

CSL_API __MPFR_DECLSPEC void dspba_mpfr_flags_clear (mpfr_flags_t);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_flags_set (mpfr_flags_t);
CSL_API __MPFR_DECLSPEC mpfr_flags_t dspba_mpfr_flags_test (mpfr_flags_t);
CSL_API __MPFR_DECLSPEC mpfr_flags_t dspba_mpfr_flags_save (void);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_flags_restore (mpfr_flags_t,
                                         mpfr_flags_t);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_check_range (mpfr_ptr, int, mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC void dspba_mpfr_init2 (mpfr_ptr, mpfr_prec_t);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_init (mpfr_ptr);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_clear (mpfr_ptr);

CSL_API __MPFR_DECLSPEC void
  dspba_mpfr_inits2 (mpfr_prec_t, mpfr_ptr, ...) __MPFR_SENTINEL_ATTR;
CSL_API __MPFR_DECLSPEC void
  dspba_mpfr_inits (mpfr_ptr, ...) __MPFR_SENTINEL_ATTR;
CSL_API __MPFR_DECLSPEC void
  dspba_mpfr_clears (mpfr_ptr, ...) __MPFR_SENTINEL_ATTR;

CSL_API __MPFR_DECLSPEC int dspba_mpfr_prec_round (mpfr_ptr, mpfr_prec_t, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_can_round (mpfr_srcptr, mpfr_exp_t, mpfr_rnd_t,
                                    mpfr_rnd_t, mpfr_prec_t);
CSL_API __MPFR_DECLSPEC mpfr_prec_t dspba_mpfr_min_prec (mpfr_srcptr);

CSL_API __MPFR_DECLSPEC mpfr_exp_t dspba_mpfr_get_exp (mpfr_srcptr);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_set_exp (mpfr_ptr, mpfr_exp_t);
CSL_API __MPFR_DECLSPEC mpfr_prec_t dspba_mpfr_get_prec (mpfr_srcptr);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_set_prec (mpfr_ptr, mpfr_prec_t);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_set_prec_raw (mpfr_ptr, mpfr_prec_t);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_set_default_prec (mpfr_prec_t);
CSL_API __MPFR_DECLSPEC mpfr_prec_t dspba_mpfr_get_default_prec (void);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_set_d (mpfr_ptr, double, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_set_flt (mpfr_ptr, float, mpfr_rnd_t);
#ifdef MPFR_WANT_DECIMAL_FLOATS
/* _Decimal64 is not defined in C++,
   cf https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51364 */
CSL_API __MPFR_DECLSPEC int dspba_mpfr_set_decimal64 (mpfr_ptr, _Decimal64, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_set_decimal128 (mpfr_ptr, _Decimal128, mpfr_rnd_t);
#endif
CSL_API __MPFR_DECLSPEC int dspba_mpfr_set_ld (mpfr_ptr, long double, mpfr_rnd_t);
#ifdef MPFR_WANT_FLOAT128
CSL_API __MPFR_DECLSPEC int dspba_mpfr_set_float128 (mpfr_ptr, _Float128, mpfr_rnd_t);
__MPFR_DECLSPEC _Float128 dspba_mpfr_get_float128 (mpfr_srcptr, mpfr_rnd_t);
#endif
CSL_API __MPFR_DECLSPEC int dspba_mpfr_set_z (mpfr_ptr, mpz_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_set_z_2exp (mpfr_ptr, mpz_srcptr, mpfr_exp_t,
                                     mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_set_nan (mpfr_ptr);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_set_inf (mpfr_ptr, int);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_set_zero (mpfr_ptr, int);

#ifndef MPFR_USE_MINI_GMP
  /* mini-gmp does not provide mpf_t, we disable the following functions */
CSL_API __MPFR_DECLSPEC int dspba_mpfr_set_f (mpfr_ptr, mpf_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_cmp_f (mpfr_srcptr, mpf_srcptr);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_get_f (mpf_ptr, mpfr_srcptr, mpfr_rnd_t);
#endif
CSL_API __MPFR_DECLSPEC int dspba_mpfr_set_si (mpfr_ptr, long, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_set_ui (mpfr_ptr, unsigned long, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_set_si_2exp (mpfr_ptr, long, mpfr_exp_t, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_set_ui_2exp (mpfr_ptr, unsigned long, mpfr_exp_t,
                                      mpfr_rnd_t);
#ifndef MPFR_USE_MINI_GMP
  /* mini-gmp does not provide mpq_t, we disable the following functions */
CSL_API __MPFR_DECLSPEC int dspba_mpfr_set_q (mpfr_ptr, mpq_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_mul_q (mpfr_ptr, mpfr_srcptr, mpq_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_div_q (mpfr_ptr, mpfr_srcptr, mpq_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_add_q (mpfr_ptr, mpfr_srcptr, mpq_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_sub_q (mpfr_ptr, mpfr_srcptr, mpq_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_cmp_q (mpfr_srcptr, mpq_srcptr);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_get_q (mpq_ptr q, mpfr_srcptr f);
#endif
CSL_API __MPFR_DECLSPEC int dspba_mpfr_set_str (mpfr_ptr, const char *, int, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_init_set_str (mpfr_ptr, const char *, int,
                                       mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_set4 (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t, int);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_abs (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_set (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_neg (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_signbit (mpfr_srcptr);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_setsign (mpfr_ptr, mpfr_srcptr, int, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_copysign (mpfr_ptr, mpfr_srcptr, mpfr_srcptr,
                                   mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC mpfr_exp_t dspba_mpfr_get_z_2exp (mpz_ptr, mpfr_srcptr);
CSL_API __MPFR_DECLSPEC float dspba_mpfr_get_flt (mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC double dspba_mpfr_get_d (mpfr_srcptr, mpfr_rnd_t);
#ifdef MPFR_WANT_DECIMAL_FLOATS
__MPFR_DECLSPEC _Decimal64 dspba_mpfr_get_decimal64 (mpfr_srcptr, mpfr_rnd_t);
__MPFR_DECLSPEC _Decimal128 dspba_mpfr_get_decimal128 (mpfr_srcptr, mpfr_rnd_t);
#endif
__MPFR_DECLSPEC long double dspba_mpfr_get_ld (mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC double dspba_mpfr_get_d1 (mpfr_srcptr);
CSL_API __MPFR_DECLSPEC double dspba_mpfr_get_d_2exp (long*, mpfr_srcptr, mpfr_rnd_t);
__MPFR_DECLSPEC long double dspba_mpfr_get_ld_2exp (long*, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_frexp (mpfr_exp_t*, mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
__MPFR_DECLSPEC long dspba_mpfr_get_si (mpfr_srcptr, mpfr_rnd_t);
__MPFR_DECLSPEC unsigned long dspba_mpfr_get_ui (mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC size_t dspba_mpfr_get_str_ndigits (int, mpfr_prec_t);
__MPFR_DECLSPEC char * dspba_mpfr_get_str (char*, mpfr_exp_t*, int, size_t,
                                     mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_get_z (mpz_ptr z, mpfr_srcptr f, mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC void dspba_mpfr_free_str (char *);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_urandom (mpfr_ptr, gmp_randstate_t, mpfr_rnd_t);
#ifndef _MPFR_NO_DEPRECATED_GRANDOM /* for the test of this function */
MPFR_DEPRECATED
#endif
CSL_API __MPFR_DECLSPEC int dspba_mpfr_grandom (mpfr_ptr, mpfr_ptr, gmp_randstate_t,
                                  mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_nrandom (mpfr_ptr, gmp_randstate_t, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_erandom (mpfr_ptr, gmp_randstate_t, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_urandomb (mpfr_ptr, gmp_randstate_t);

CSL_API __MPFR_DECLSPEC void dspba_mpfr_nextabove (mpfr_ptr);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_nextbelow (mpfr_ptr);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_nexttoward (mpfr_ptr, mpfr_srcptr);

#ifndef MPFR_USE_MINI_GMP
CSL_API __MPFR_DECLSPEC int dspba_mpfr_printf (const char*, ...);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_asprintf (char**, const char*, ...);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_sprintf (char*, const char*, ...);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_snprintf (char*, size_t, const char*, ...);
#endif

CSL_API __MPFR_DECLSPEC int dspba_mpfr_pow (mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_pow_si (mpfr_ptr, mpfr_srcptr, long, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_pow_ui (mpfr_ptr, mpfr_srcptr, unsigned long,
                                 mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_ui_pow_ui (mpfr_ptr, unsigned long, unsigned long,
                                    mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_ui_pow (mpfr_ptr, unsigned long, mpfr_srcptr,
                                 mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_pow_z (mpfr_ptr, mpfr_srcptr, mpz_srcptr, mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_sqrt (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_sqrt_ui (mpfr_ptr, unsigned long, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_rec_sqrt (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_add (mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_sub (mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_mul (mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_div (mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_add_ui (mpfr_ptr, mpfr_srcptr, unsigned long,
                                 mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_sub_ui (mpfr_ptr, mpfr_srcptr, unsigned long,
                                 mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_ui_sub (mpfr_ptr, unsigned long, mpfr_srcptr,
                                 mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_mul_ui (mpfr_ptr, mpfr_srcptr, unsigned long,
                                 mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_div_ui (mpfr_ptr, mpfr_srcptr, unsigned long,
                                 mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_ui_div (mpfr_ptr, unsigned long, mpfr_srcptr,
                                 mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_add_si (mpfr_ptr, mpfr_srcptr, long, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_sub_si (mpfr_ptr, mpfr_srcptr, long, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_si_sub (mpfr_ptr, long, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_mul_si (mpfr_ptr, mpfr_srcptr, long, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_div_si (mpfr_ptr, mpfr_srcptr, long, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_si_div (mpfr_ptr, long, mpfr_srcptr, mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_add_d (mpfr_ptr, mpfr_srcptr, double, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_sub_d (mpfr_ptr, mpfr_srcptr, double, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_d_sub (mpfr_ptr, double, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_mul_d (mpfr_ptr, mpfr_srcptr, double, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_div_d (mpfr_ptr, mpfr_srcptr, double, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_d_div (mpfr_ptr, double, mpfr_srcptr, mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_sqr (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_const_pi (mpfr_ptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_const_log2 (mpfr_ptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_const_euler (mpfr_ptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_const_catalan (mpfr_ptr, mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_agm (mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_log (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_log2 (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_log10 (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_log1p (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_log_ui (mpfr_ptr, unsigned long, mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_exp (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_exp2 (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_exp10 (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_expm1 (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_eint (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_li2 (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_cmp  (mpfr_srcptr, mpfr_srcptr);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_cmp3 (mpfr_srcptr, mpfr_srcptr, int);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_cmp_d (mpfr_srcptr, double);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_cmp_ld (mpfr_srcptr, long double);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_cmp_ui (mpfr_srcptr, unsigned long);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_cmp_si (mpfr_srcptr, long);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_cmp_ui_2exp (mpfr_srcptr, unsigned long, mpfr_exp_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_cmp_si_2exp (mpfr_srcptr, long, mpfr_exp_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_cmpabs (mpfr_srcptr, mpfr_srcptr);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_cmpabs_ui (mpfr_srcptr, unsigned long);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_reldiff (mpfr_ptr, mpfr_srcptr, mpfr_srcptr,
                                   mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_eq (mpfr_srcptr, mpfr_srcptr, unsigned long);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_sgn (mpfr_srcptr);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_mul_2exp (mpfr_ptr, mpfr_srcptr, unsigned long,
                                   mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_div_2exp (mpfr_ptr, mpfr_srcptr, unsigned long,
                                   mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_mul_2ui (mpfr_ptr, mpfr_srcptr, unsigned long,
                                  mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_div_2ui (mpfr_ptr, mpfr_srcptr, unsigned long,
                                  mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_mul_2si (mpfr_ptr, mpfr_srcptr, long, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_div_2si (mpfr_ptr, mpfr_srcptr, long, mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_rint (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_roundeven (mpfr_ptr, mpfr_srcptr);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_round (mpfr_ptr, mpfr_srcptr);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_trunc (mpfr_ptr, mpfr_srcptr);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_ceil (mpfr_ptr, mpfr_srcptr);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_floor (mpfr_ptr, mpfr_srcptr);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_rint_roundeven (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_rint_round (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_rint_trunc (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_rint_ceil (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_rint_floor (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_frac (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_modf (mpfr_ptr, mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_remquo (mpfr_ptr, long*, mpfr_srcptr, mpfr_srcptr,
                                 mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_remainder (mpfr_ptr, mpfr_srcptr, mpfr_srcptr,
                                    mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_fmod (mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_fmodquo (mpfr_ptr, long*, mpfr_srcptr, mpfr_srcptr,
                                  mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_fits_ulong_p (mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_fits_slong_p (mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_fits_uint_p (mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_fits_sint_p (mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_fits_ushort_p (mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_fits_sshort_p (mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_fits_uintmax_p (mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_fits_intmax_p (mpfr_srcptr, mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC void dspba_mpfr_extract (mpz_ptr, mpfr_srcptr, unsigned int);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_swap (mpfr_ptr, mpfr_ptr);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_dump (mpfr_srcptr);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_nan_p (mpfr_srcptr);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_inf_p (mpfr_srcptr);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_number_p (mpfr_srcptr);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_integer_p (mpfr_srcptr);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_zero_p (mpfr_srcptr);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_regular_p (mpfr_srcptr);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_greater_p (mpfr_srcptr, mpfr_srcptr);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_greaterequal_p (mpfr_srcptr, mpfr_srcptr);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_less_p (mpfr_srcptr, mpfr_srcptr);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_lessequal_p (mpfr_srcptr, mpfr_srcptr);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_lessgreater_p (mpfr_srcptr, mpfr_srcptr);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_equal_p (mpfr_srcptr, mpfr_srcptr);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_unordered_p (mpfr_srcptr, mpfr_srcptr);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_atanh (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_acosh (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_asinh (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_cosh (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_sinh (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_tanh (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_sinh_cosh (mpfr_ptr, mpfr_ptr, mpfr_srcptr,
                                    mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_sech (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_csch (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_coth (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_acos (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_asin (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_atan (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_sin (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_sin_cos (mpfr_ptr, mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_cos (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_tan (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_atan2 (mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_sec (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_csc (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_cot (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_hypot (mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_erf (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_erfc (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_cbrt (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
#ifndef _MPFR_NO_DEPRECATED_ROOT /* for the test of this function */
MPFR_DEPRECATED
#endif
CSL_API __MPFR_DECLSPEC int dspba_mpfr_root (mpfr_ptr, mpfr_srcptr, unsigned long,
                               mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_rootn_ui (mpfr_ptr, mpfr_srcptr, unsigned long,
                                   mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_gamma (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_gamma_inc (mpfr_ptr, mpfr_srcptr, mpfr_srcptr,
                                    mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_beta (mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_lngamma (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_lgamma (mpfr_ptr, int *, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_digamma (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_zeta (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_zeta_ui (mpfr_ptr, unsigned long, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_fac_ui (mpfr_ptr, unsigned long, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_j0 (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_j1 (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_jn (mpfr_ptr, long, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_y0 (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_y1 (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_yn (mpfr_ptr, long, mpfr_srcptr, mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_ai (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_min (mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_max (mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_dim (mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_mul_z (mpfr_ptr, mpfr_srcptr, mpz_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_div_z (mpfr_ptr, mpfr_srcptr, mpz_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_add_z (mpfr_ptr, mpfr_srcptr, mpz_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_sub_z (mpfr_ptr, mpfr_srcptr, mpz_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_z_sub (mpfr_ptr, mpz_srcptr, mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_cmp_z (mpfr_srcptr, mpz_srcptr);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_fma (mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_srcptr,
                              mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_fms (mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_srcptr,
                              mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_fmma (mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_srcptr,
                               mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_fmms (mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_srcptr,
                               mpfr_srcptr, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_sum (mpfr_ptr, const mpfr_ptr *, unsigned long,
                              mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_dot (mpfr_ptr, const mpfr_ptr *, const mpfr_ptr *,
                              unsigned long, mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC void dspba_mpfr_free_cache (void);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_free_cache2 (mpfr_free_cache_t);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_free_pool (void);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_mp_memory_cleanup (void);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_subnormalize (mpfr_ptr, int, mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_strtofr (mpfr_ptr, const char *, char **, int,
                                  mpfr_rnd_t);

CSL_API __MPFR_DECLSPEC void dspba_mpfr_round_nearest_away_begin (mpfr_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_round_nearest_away_end (mpfr_t, int);

CSL_API __MPFR_DECLSPEC size_t dspba_mpfr_custom_get_size (mpfr_prec_t);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_custom_init (void *, mpfr_prec_t);
__MPFR_DECLSPEC MPFR_RETURNS_NONNULL void *
  dspba_mpfr_custom_get_significand (mpfr_srcptr);
CSL_API __MPFR_DECLSPEC mpfr_exp_t dspba_mpfr_custom_get_exp (mpfr_srcptr);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_custom_move (mpfr_ptr, void *);
CSL_API __MPFR_DECLSPEC void dspba_mpfr_custom_init_set (mpfr_ptr, int, mpfr_exp_t,
                                           mpfr_prec_t, void *);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_custom_get_kind (mpfr_srcptr);

CSL_API __MPFR_DECLSPEC int dspba_mpfr_total_order_p (mpfr_srcptr, mpfr_srcptr);

#if defined (__cplusplus)
}
#endif

/* Define MPFR_USE_EXTENSION to avoid "gcc -pedantic" warnings. */
#ifndef MPFR_EXTENSION
# if defined(MPFR_USE_EXTENSION)
#  define MPFR_EXTENSION __extension__
# else
#  define MPFR_EXTENSION
# endif
#endif

/* Warning! This macro doesn't work with K&R C (e.g., compare the "gcc -E"
   output with and without -traditional) and shouldn't be used internally.
   For public use only, but see the MPFR manual. */
#define MPFR_DECL_INIT(_x, _p)                                        \
  MPFR_EXTENSION mp_limb_t __gdspba_mpfr_local_tab_##_x[((_p)-1)/GMP_NUMB_BITS+1]; \
  MPFR_EXTENSION mpfr_t _x = {{(_p),1,__MPFR_EXP_NAN,__gdspba_mpfr_local_tab_##_x}}

#if MPFR_USE_C99_FEATURE
/* C99 & C11 version: functions with multiple inputs supported */
#define dspba_mpfr_round_nearest_away(func, rop, ...)                         \
  (dspba_mpfr_round_nearest_away_begin(rop),                                  \
   dspba_mpfr_round_nearest_away_end((rop), func((rop), __VA_ARGS__, MPFR_RNDN)))
#else
/* C89 version: function with one input supported */
#define dspba_mpfr_round_nearest_away(func, rop, op)                          \
  (dspba_mpfr_round_nearest_away_begin(rop),                                  \
   dspba_mpfr_round_nearest_away_end((rop), func((rop), (op), MPFR_RNDN)))
#endif

/* Fast access macros to replace function interface.
   If the USER don't want to use the macro interface, let him make happy
   even if it produces faster and smaller code. */
#ifndef MPFR_USE_NO_MACRO

/* Inlining these functions is both faster and smaller */
#define dspba_mpfr_nan_p(_x)      ((_x)->_mpfr_exp == __MPFR_EXP_NAN)
#define dspba_mpfr_inf_p(_x)      ((_x)->_mpfr_exp == __MPFR_EXP_INF)
#define dspba_mpfr_zero_p(_x)     ((_x)->_mpfr_exp == __MPFR_EXP_ZERO)
#define dspba_mpfr_regular_p(_x)  ((_x)->_mpfr_exp >  __MPFR_EXP_INF)
#define dspba_mpfr_sgn(_x)                                               \
  ((_x)->_mpfr_exp < __MPFR_EXP_INF ?                              \
   (dspba_mpfr_nan_p (_x) ? dspba_mpfr_set_erangeflag () : (mpfr_void) 0), 0 : \
   MPFR_SIGN (_x))

/* Prevent them from using as lvalues */
#define MPFR_VALUE_OF(x)  (0 ? (x) : (x))
#define dspba_mpfr_get_prec(_x) MPFR_VALUE_OF((_x)->_mpfr_prec)
#define dspba_mpfr_get_exp(_x)  MPFR_VALUE_OF((_x)->_mpfr_exp)
/* Note 1: If need be, the MPFR_VALUE_OF can be used for other expressions
   (of any type). Thanks to Wojtek Lerch and Tim Rentsch for the idea.
   Note 2: Defining dspba_mpfr_get_exp() as a macro has the effect to disable
   the check that the argument is a pure FP number (done in the function);
   this increases the risk of undetected error and makes debugging more
   complex. Is it really worth in practice? (Potential FIXME) */

#define dspba_mpfr_round(a,b) dspba_mpfr_rint((a), (b), MPFR_RNDNA)
#define dspba_mpfr_trunc(a,b) dspba_mpfr_rint((a), (b), MPFR_RNDZ)
#define dspba_mpfr_ceil(a,b)  dspba_mpfr_rint((a), (b), MPFR_RNDU)
#define dspba_mpfr_floor(a,b) dspba_mpfr_rint((a), (b), MPFR_RNDD)

#define dspba_mpfr_cmp_ui(b,i) dspba_mpfr_cmp_ui_2exp((b),(i),0)
#define dspba_mpfr_cmp_si(b,i) dspba_mpfr_cmp_si_2exp((b),(i),0)
#define dspba_mpfr_set(a,b,r)  dspba_mpfr_set4(a,b,r,MPFR_SIGN(b))
#define dspba_mpfr_abs(a,b,r)  dspba_mpfr_set4(a,b,r,1)
#define dspba_mpfr_copysign(a,b,c,r) dspba_mpfr_set4(a,b,r,MPFR_SIGN(c))
#define dspba_mpfr_setsign(a,b,s,r) dspba_mpfr_set4(a,b,r,(s) ? -1 : 1)
#define dspba_mpfr_signbit(x)  (MPFR_SIGN(x) < 0)
#define dspba_mpfr_cmp(b, c)   dspba_mpfr_cmp3(b, c, 1)
#define dspba_mpfr_mul_2exp(y,x,n,r) dspba_mpfr_mul_2ui((y),(x),(n),(r))
#define dspba_mpfr_div_2exp(y,x,n,r) dspba_mpfr_div_2ui((y),(x),(n),(r))


/* When using GCC or ICC, optimize certain common comparisons and affectations.
   Added casts to improve robustness in case of undefined behavior and
   compiler extensions based on UB (in particular -fwrapv). MPFR doesn't
   use such extensions, but these macros will be used by 3rd-party code,
   where such extensions may be required.
   Moreover casts to unsigned long have been added to avoid warnings in
   programs that use MPFR and are compiled with -Wconversion; such casts
   are OK since if X is a constant expression, then (unsigned long) X is
   also a constant expression, so that the optimizations still work. The
   warnings are probably related to the following two bugs:
     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=4210
     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=38470 (possibly a variant)
   and the casts could be removed once these bugs are fixed.
   Casts shouldn't be used on the generic calls (to the ..._2exp functions),
   where implicit conversions are performed. Indeed, having at least one
   implicit conversion in the macro allows the compiler to emit diagnostics
   when normally expected, for instance in the following call:
     dspba_mpfr_set_ui (x, "foo", MPFR_RNDN);
   If this is not possible (for future macros), one of the tricks described
   on http://groups.google.com/group/comp.std.c/msg/e92abd24bf9eaf7b could
   be used. */
#if defined (__GNUC__) && !defined(__cplusplus)
#if (__GNUC__ >= 2)

#undef dspba_mpfr_cmp_ui
/* We use the fact that dspba_mpfr_sgn on NaN sets the erange flag and returns 0.
   But warning! dspba_mpfr_sgn is specified as a macro in the API, thus the macro
   mustn't be used if side effects are possible, like here. */
#define dspba_mpfr_cmp_ui(_f,_u)                                      \
  (__builtin_constant_p (_u) && (mpfr_ulong) (_u) == 0 ?        \
   (dspba_mpfr_sgn) (_f) :                                            \
   dspba_mpfr_cmp_ui_2exp ((_f), (_u), 0))

#undef dspba_mpfr_cmp_si
#define dspba_mpfr_cmp_si(_f,_s)                                      \
  (__builtin_constant_p (_s) && (mpfr_long) (_s) >= 0 ?         \
   dspba_mpfr_cmp_ui ((_f), (mpfr_ulong) (mpfr_long) (_s)) :          \
   dspba_mpfr_cmp_si_2exp ((_f), (_s), 0))

#if __GNUC__ > 2 || __GNUC_MINOR__ >= 95
#undef dspba_mpfr_set_ui
#define dspba_mpfr_set_ui(_f,_u,_r)                                   \
  (__builtin_constant_p (_u) && (mpfr_ulong) (_u) == 0 ?        \
   __extension__ ({                                             \
       mpfr_ptr _p = (_f);                                      \
       _p->_mpfr_sign = 1;                                      \
       _p->_mpfr_exp = __MPFR_EXP_ZERO;                         \
       (mpfr_void) (_r); 0; }) :                                \
   dspba_mpfr_set_ui_2exp ((_f), (_u), 0, (_r)))
#endif

#undef dspba_mpfr_set_si
#define dspba_mpfr_set_si(_f,_s,_r)                                   \
  (__builtin_constant_p (_s) && (mpfr_long) (_s) >= 0 ?         \
   dspba_mpfr_set_ui ((_f), (mpfr_ulong) (mpfr_long) (_s), (_r)) :    \
   dspba_mpfr_set_si_2exp ((_f), (_s), 0, (_r)))

#if __GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4)
/* If the source is a constant number that is a power of 2,
   optimize the call */
#undef dspba_mpfr_mul_ui
#define dspba_mpfr_mul_ui(_f, _g, _u,_r)                              \
  (__builtin_constant_p (_u) && (mpfr_ulong) (_u) >= 1 &&       \
   ((mpfr_ulong) (_u) & ((mpfr_ulong) (_u) - 1)) == 0 ?         \
   dspba_mpfr_mul_2si((_f), (_g), __builtin_ctzl (_u), (_r)) :        \
   dspba_mpfr_mul_ui ((_f), (_g), (_u), (_r)))
#undef dspba_mpfr_div_ui
#define dspba_mpfr_div_ui(_f, _g, _u,_r)                              \
  (__builtin_constant_p (_u) && (mpfr_ulong) (_u) >= 1 &&       \
   ((mpfr_ulong) (_u) & ((mpfr_ulong) (_u) - 1)) == 0 ?         \
   dspba_mpfr_mul_2si((_f), (_g), - __builtin_ctzl (_u), (_r)) :      \
   dspba_mpfr_div_ui ((_f), (_g), (_u), (_r)))
#endif

/* If the source is a constant number that is non-negative,
   optimize the call */
#undef dspba_mpfr_mul_si
#define dspba_mpfr_mul_si(_f, _g, _s,_r)                                      \
  (__builtin_constant_p (_s) && (mpfr_long) (_s) >= 0 ?                 \
   dspba_mpfr_mul_ui ((_f), (_g), (mpfr_ulong) (mpfr_long) (_s), (_r)) :      \
   dspba_mpfr_mul_si ((_f), (_g), (_s), (_r)))
#undef dspba_mpfr_div_si
#define dspba_mpfr_div_si(_f, _g, _s,_r)                                      \
  (__builtin_constant_p (_s) && (mpfr_long) (_s) >= 0 ?                 \
   dspba_mpfr_div_ui ((_f), (_g), (mpfr_ulong) (mpfr_long) (_s), (_r)) :      \
   dspba_mpfr_div_si ((_f), (_g), (_s), (_r)))

#endif
#endif

/* Macro version of dspba_mpfr_stack interface for fast access */
#define dspba_mpfr_custom_get_size(p) ((mpfr_size_t)                          \
       (((p)+GMP_NUMB_BITS-1)/GMP_NUMB_BITS*sizeof (mp_limb_t)))
#define dspba_mpfr_custom_init(m,p) do {} while (0)
#define dspba_mpfr_custom_get_significand(x) ((mpfr_void*)((x)->_mpfr_d))
#define dspba_mpfr_custom_get_exp(x) ((x)->_mpfr_exp)
#define dspba_mpfr_custom_move(x,m) do { ((x)->_mpfr_d = (mp_limb_t*)(m)); } while (0)
#define dspba_mpfr_custom_init_set(x,k,e,p,m) do {                   \
  mpfr_ptr _x = (x);                                           \
  mpfr_exp_t _e;                                               \
  mpfr_kind_t _t;                                              \
  mpfr_int _s, _k;                                             \
  _k = (k);                                                    \
  if (_k >= 0)  {                                              \
    _t = (mpfr_kind_t) _k;                                     \
    _s = 1;                                                    \
  } else {                                                     \
    _t = (mpfr_kind_t) - _k;                                   \
    _s = -1;                                                   \
  }                                                            \
  _e = _t == MPFR_REGULAR_KIND ? (e) :                         \
    _t == MPFR_NAN_KIND ? __MPFR_EXP_NAN :                     \
    _t == MPFR_INF_KIND ? __MPFR_EXP_INF : __MPFR_EXP_ZERO;    \
  _x->_mpfr_prec = (p);                                        \
  _x->_mpfr_sign = _s;                                         \
  _x->_mpfr_exp  = _e;                                         \
  _x->_mpfr_d    = (mp_limb_t*) (m);                           \
 } while (0)
#define dspba_mpfr_custom_get_kind(x)                                         \
  ( (x)->_mpfr_exp >  __MPFR_EXP_INF ?                                  \
    (mpfr_int) MPFR_REGULAR_KIND * MPFR_SIGN (x)                        \
  : (x)->_mpfr_exp == __MPFR_EXP_INF ?                                  \
    (mpfr_int) MPFR_INF_KIND * MPFR_SIGN (x)                            \
  : (x)->_mpfr_exp == __MPFR_EXP_NAN ? (mpfr_int) MPFR_NAN_KIND         \
  : (mpfr_int) MPFR_ZERO_KIND * MPFR_SIGN (x) )


#endif /* MPFR_USE_NO_MACRO */

/* These are defined to be macros */
#define dspba_mpfr_init_set_si(x, i, rnd) \
 ( dspba_mpfr_init(x), dspba_mpfr_set_si((x), (i), (rnd)) )
#define dspba_mpfr_init_set_ui(x, i, rnd) \
 ( dspba_mpfr_init(x), dspba_mpfr_set_ui((x), (i), (rnd)) )
#define dspba_mpfr_init_set_d(x, d, rnd) \
 ( dspba_mpfr_init(x), dspba_mpfr_set_d((x), (d), (rnd)) )
#define dspba_mpfr_init_set_ld(x, d, rnd) \
 ( dspba_mpfr_init(x), dspba_mpfr_set_ld((x), (d), (rnd)) )
#define dspba_mpfr_init_set_z(x, i, rnd) \
 ( dspba_mpfr_init(x), dspba_mpfr_set_z((x), (i), (rnd)) )
#ifndef MPFR_USE_MINI_GMP
#define dspba_mpfr_init_set_q(x, i, rnd) \
 ( dspba_mpfr_init(x), dspba_mpfr_set_q((x), (i), (rnd)) )
#define dspba_mpfr_init_set_f(x, y, rnd) \
 ( dspba_mpfr_init(x), dspba_mpfr_set_f((x), (y), (rnd)) )
#endif
#define dspba_mpfr_init_set(x, y, rnd) \
 ( dspba_mpfr_init(x), dspba_mpfr_set((x), (y), (rnd)) )

/* Compatibility layer -- obsolete functions and macros */
/* Note: it is not possible to output warnings, unless one defines
 * a deprecated variable and uses it, e.g.
 *   MPFR_DEPRECATED extern int dspba_mpfr_deprecated_feature;
 *   #define MPFR_EMIN_MIN ((void)dspba_mpfr_deprecated_feature,dspba_mpfr_get_emin_min())
 * (the cast to void avoids a warning because the left-hand operand
 * has no effect).
 */
#define dspba_mpfr_cmp_abs dspba_mpfr_cmpabs
#define dspba_mpfr_round_prec(x,r,p) dspba_mpfr_prec_round(x,p,r)
#define __gmp_default_rounding_mode (dspba_mpfr_get_default_rounding_mode())
#define __dspba_mpfr_emin (dspba_mpfr_get_emin())
#define __dspba_mpfr_emax (dspba_mpfr_get_emax())
#define __dspba_mpfr_default_fp_bit_precision (dspba_mpfr_get_default_fp_bit_precision())
#define MPFR_EMIN_MIN dspba_mpfr_get_emin_min()
#define MPFR_EMIN_MAX dspba_mpfr_get_emin_max()
#define MPFR_EMAX_MIN dspba_mpfr_get_emax_min()
#define MPFR_EMAX_MAX dspba_mpfr_get_emax_max()
#define dspba_mpfr_version (dspba_mpfr_get_version())
#ifndef mpz_set_fr
# define mpz_set_fr dspba_mpfr_get_z
#endif
#define dspba_mpfr_get_z_exp dspba_mpfr_get_z_2exp
#define dspba_mpfr_custom_get_mantissa dspba_mpfr_custom_get_significand

#endif /* __MPFR_H */


/* Check if <stdint.h> / <inttypes.h> is included or if the user
   explicitly wants intmax_t. Automatic detection is done by
   checking:
     - INTMAX_C and UINTMAX_C, but not if the compiler is a C++ one
       (as suggested by Patrick Pelissier) because the test does not
       work well in this case. See:
         https://sympa.inria.fr/sympa/arc/mpfr/2010-02/msg00025.html
       We do not check INTMAX_MAX and UINTMAX_MAX because under Solaris,
       these macros are always defined by <limits.h> (i.e. even when
       <stdint.h> and <inttypes.h> are not included).
     - _STDINT_H (defined by the glibc), _STDINT_H_ (defined under
       Mac OS X) and _STDINT (defined under MS Visual Studio), but
       this test may not work with all implementations.
       Portable software should not rely on these tests.
*/
#if (defined (INTMAX_C) && defined (UINTMAX_C) && !defined(__cplusplus)) || \
  defined (MPFR_USE_INTMAX_T) || \
  defined (_STDINT_H) || defined (_STDINT_H_) || defined (_STDINT) || \
  defined (_SYS_STDINT_H_) /* needed for FreeBSD */
# ifndef _MPFR_H_HAVE_INTMAX_T
# define _MPFR_H_HAVE_INTMAX_T 1

#if defined (__cplusplus)
extern "C" {
#endif

#define dspba_mpfr_set_sj __gdspba_mpfr_set_sj
#define dspba_mpfr_set_sj_2exp __gdspba_mpfr_set_sj_2exp
#define dspba_mpfr_set_uj __gdspba_mpfr_set_uj
#define dspba_mpfr_set_uj_2exp __gdspba_mpfr_set_uj_2exp
#define dspba_mpfr_get_sj __gdspba_mpfr_dspba_mpfr_get_sj
#define dspba_mpfr_get_uj __gdspba_mpfr_dspba_mpfr_get_uj
CSL_API __MPFR_DECLSPEC int dspba_mpfr_set_sj (mpfr_t, intmax_t, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_set_sj_2exp (mpfr_t, intmax_t, intmax_t, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_set_uj (mpfr_t, uintmax_t, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_set_uj_2exp (mpfr_t, uintmax_t, intmax_t, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC intmax_t dspba_mpfr_get_sj (mpfr_srcptr, mpfr_rnd_t);
__MPFR_DECLSPEC uintmax_t dspba_mpfr_get_uj (mpfr_srcptr, mpfr_rnd_t);

#if defined (__cplusplus)
}
#endif

# endif /* _MPFR_H_HAVE_INTMAX_T */
#endif


/* Check if <stdio.h> has been included or if the user wants FILE */
#if defined (_GMP_H_HAVE_FILE) || defined (MPFR_USE_FILE)
# ifndef _MPFR_H_HAVE_FILE
# define _MPFR_H_HAVE_FILE 1

#if defined (__cplusplus)
extern "C" {
#endif

#define dspba_mpfr_inp_str __gdspba_mpfr_inp_str
#define dspba_mpfr_out_str __gdspba_mpfr_out_str
CSL_API __MPFR_DECLSPEC size_t dspba_mpfr_inp_str (mpfr_ptr, FILE*, int, mpfr_rnd_t);
CSL_API __MPFR_DECLSPEC size_t dspba_mpfr_out_str (FILE*, int, size_t, mpfr_srcptr,
                                     mpfr_rnd_t);
#ifndef MPFR_USE_MINI_GMP
#define dspba_mpfr_fprintf __gdspba_mpfr_fprintf
CSL_API __MPFR_DECLSPEC int dspba_mpfr_fprintf (FILE*, const char*, ...);
#endif
#define dspba_mpfr_fpif_export __gdspba_mpfr_fpif_export
#define dspba_mpfr_fpif_import __gdspba_mpfr_fpif_import
CSL_API __MPFR_DECLSPEC int dspba_mpfr_fpif_export (FILE*, mpfr_ptr);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_fpif_import (mpfr_ptr, FILE*);

#if defined (__cplusplus)
}
#endif

# endif /* _MPFR_H_HAVE_FILE */
#endif


/* check if <stdarg.h> has been included or if the user wants va_list */
#if defined (_GMP_H_HAVE_VA_LIST) || defined (MPFR_USE_VA_LIST)
# ifndef _MPFR_H_HAVE_VA_LIST
# define _MPFR_H_HAVE_VA_LIST 1

#if defined (__cplusplus)
extern "C" {
#endif

#define dspba_mpfr_vprintf __gdspba_mpfr_vprintf
#define dspba_mpfr_vasprintf __gdspba_mpfr_vasprintf
#define dspba_mpfr_vsprintf __gdspba_mpfr_vsprintf
#define dspba_mpfr_vsnprintf __gdspba_mpfr_vsnprintf
CSL_API __MPFR_DECLSPEC int dspba_mpfr_vprintf (const char*, va_list);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_vasprintf (char**, const char*, va_list);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_vsprintf (char*, const char*, va_list);
CSL_API __MPFR_DECLSPEC int dspba_mpfr_vsnprintf (char*, size_t, const char*, va_list);

#if defined (__cplusplus)
}
#endif

# endif /* _MPFR_H_HAVE_VA_LIST */
#endif


/* check if <stdarg.h> has been included and if FILE is available
   (see above) */
#if defined (_MPFR_H_HAVE_VA_LIST) && defined (_MPFR_H_HAVE_FILE)
# ifndef _MPFR_H_HAVE_VA_LIST_FILE
# define _MPFR_H_HAVE_VA_LIST_FILE 1

#if defined (__cplusplus)
extern "C" {
#endif

#define dspba_mpfr_vfprintf __gdspba_mpfr_vfprintf
CSL_API __MPFR_DECLSPEC int dspba_mpfr_vfprintf (FILE*, const char*, va_list);

#if defined (__cplusplus)
}
#endif

# endif /* _MPFR_H_HAVE_VA_LIST_FILE */
#endif


// generated dspba_ function name aliases
#ifndef mpfr_get_version
    #define mpfr_get_z_exp dspba_mpfr_get_z_exp
    #define mpfr_get_version dspba_mpfr_get_version
    #define mpfr_get_patches dspba_mpfr_get_patches
    #define mpfr_buildopt_tls_p dspba_mpfr_buildopt_tls_p
    #define mpfr_buildopt_float128_p dspba_mpfr_buildopt_float128_p
    #define mpfr_buildopt_decimal_p dspba_mpfr_buildopt_decimal_p
    #define mpfr_buildopt_gmpinternals_p dspba_mpfr_buildopt_gmpinternals_p
    #define mpfr_buildopt_sharedcache_p dspba_mpfr_buildopt_sharedcache_p
    #define mpfr_buildopt_tune_case dspba_mpfr_buildopt_tune_case
    #define mpfr_get_emin dspba_mpfr_get_emin
    #define mpfr_set_emin dspba_mpfr_set_emin
    #define mpfr_get_emin_min dspba_mpfr_get_emin_min
    #define mpfr_get_emin_max dspba_mpfr_get_emin_max
    #define mpfr_get_emax dspba_mpfr_get_emax
    #define mpfr_set_emax dspba_mpfr_set_emax
    #define mpfr_get_emax_min dspba_mpfr_get_emax_min
    #define mpfr_get_emax_max dspba_mpfr_get_emax_max
    #define mpfr_set_default_rounding_mode dspba_mpfr_set_default_rounding_mode
    #define mpfr_get_default_rounding_mode dspba_mpfr_get_default_rounding_mode
    #define mpfr_print_rnd_mode dspba_mpfr_print_rnd_mode
    #define mpfr_clear_flags dspba_mpfr_clear_flags
    #define mpfr_clear_underflow dspba_mpfr_clear_underflow
    #define mpfr_clear_overflow dspba_mpfr_clear_overflow
    #define mpfr_clear_divby0 dspba_mpfr_clear_divby0
    #define mpfr_clear_nanflag dspba_mpfr_clear_nanflag
    #define mpfr_clear_inexflag dspba_mpfr_clear_inexflag
    #define mpfr_clear_erangeflag dspba_mpfr_clear_erangeflag
    #define mpfr_set_underflow dspba_mpfr_set_underflow
    #define mpfr_set_overflow dspba_mpfr_set_overflow
    #define mpfr_set_divby0 dspba_mpfr_set_divby0
    #define mpfr_set_nanflag dspba_mpfr_set_nanflag
    #define mpfr_set_inexflag dspba_mpfr_set_inexflag
    #define mpfr_set_erangeflag dspba_mpfr_set_erangeflag
    #define mpfr_underflow_p dspba_mpfr_underflow_p
    #define mpfr_overflow_p dspba_mpfr_overflow_p
    #define mpfr_divby0_p dspba_mpfr_divby0_p
    #define mpfr_nanflag_p dspba_mpfr_nanflag_p
    #define mpfr_inexflag_p dspba_mpfr_inexflag_p
    #define mpfr_erangeflag_p dspba_mpfr_erangeflag_p
    #define mpfr_flags_clear dspba_mpfr_flags_clear
    #define mpfr_flags_set dspba_mpfr_flags_set
    #define mpfr_flags_test dspba_mpfr_flags_test
    #define mpfr_flags_save dspba_mpfr_flags_save
    #define mpfr_flags_restore dspba_mpfr_flags_restore
    #define mpfr_check_range dspba_mpfr_check_range
    #define mpfr_init2 dspba_mpfr_init2
    #define mpfr_init dspba_mpfr_init
    #define mpfr_clear dspba_mpfr_clear
    #define mpfr_inits2 dspba_mpfr_inits2
    #define mpfr_inits dspba_mpfr_inits
    #define mpfr_clears dspba_mpfr_clears
    #define mpfr_prec_round dspba_mpfr_prec_round
    #define mpfr_can_round dspba_mpfr_can_round
    #define mpfr_min_prec dspba_mpfr_min_prec
    #define mpfr_get_exp dspba_mpfr_get_exp
    #define mpfr_set_exp dspba_mpfr_set_exp
    #define mpfr_get_prec dspba_mpfr_get_prec
    #define mpfr_set_prec dspba_mpfr_set_prec
    #define mpfr_set_prec_raw dspba_mpfr_set_prec_raw
    #define mpfr_set_default_prec dspba_mpfr_set_default_prec
    #define mpfr_get_default_prec dspba_mpfr_get_default_prec
    #define mpfr_set_d dspba_mpfr_set_d
    #define mpfr_set_flt dspba_mpfr_set_flt
    #define mpfr_set_ld dspba_mpfr_set_ld
    #define mpfr_set_z dspba_mpfr_set_z
    #define mpfr_set_z_2exp dspba_mpfr_set_z_2exp
    #define mpfr_set_nan dspba_mpfr_set_nan
    #define mpfr_set_inf dspba_mpfr_set_inf
    #define mpfr_set_zero dspba_mpfr_set_zero
    #define mpfr_set_f dspba_mpfr_set_f
    #define mpfr_cmp_f dspba_mpfr_cmp_f
    #define mpfr_get_f dspba_mpfr_get_f
    #define mpfr_set_si dspba_mpfr_set_si
    #define mpfr_set_ui dspba_mpfr_set_ui
    #define mpfr_set_si_2exp dspba_mpfr_set_si_2exp
    #define mpfr_set_ui_2exp dspba_mpfr_set_ui_2exp
    #define mpfr_set_q dspba_mpfr_set_q
    #define mpfr_mul_q dspba_mpfr_mul_q
    #define mpfr_div_q dspba_mpfr_div_q
    #define mpfr_add_q dspba_mpfr_add_q
    #define mpfr_sub_q dspba_mpfr_sub_q
    #define mpfr_cmp_q dspba_mpfr_cmp_q
    #define mpfr_get_q dspba_mpfr_get_q
    #define mpfr_set_str dspba_mpfr_set_str
    #define mpfr_init_set_str dspba_mpfr_init_set_str
    #define mpfr_set4 dspba_mpfr_set4
    #define mpfr_abs dspba_mpfr_abs
    #define mpfr_set dspba_mpfr_set
    #define mpfr_neg dspba_mpfr_neg
    #define mpfr_signbit dspba_mpfr_signbit
    #define mpfr_setsign dspba_mpfr_setsign
    #define mpfr_copysign dspba_mpfr_copysign
    #define mpfr_get_z_2exp dspba_mpfr_get_z_2exp
    #define mpfr_get_flt dspba_mpfr_get_flt
    #define mpfr_get_d dspba_mpfr_get_d
    #define mpfr_get_ld dspba_mpfr_get_ld
    #define mpfr_get_d1 dspba_mpfr_get_d1
    #define mpfr_get_d_2exp dspba_mpfr_get_d_2exp
    #define mpfr_get_ld_2exp dspba_mpfr_get_ld_2exp
    #define mpfr_frexp dspba_mpfr_frexp
    #define mpfr_get_si dspba_mpfr_get_si
    #define mpfr_get_ui dspba_mpfr_get_ui
    #define mpfr_get_str dspba_mpfr_get_str
    #define mpfr_get_z dspba_mpfr_get_z
    #define mpfr_free_str dspba_mpfr_free_str
    #define mpfr_urandom dspba_mpfr_urandom
    #define mpfr_grandom dspba_mpfr_grandom
    #define mpfr_nrandom dspba_mpfr_nrandom
    #define mpfr_erandom dspba_mpfr_erandom
    #define mpfr_urandomb dspba_mpfr_urandomb
    #define mpfr_nextabove dspba_mpfr_nextabove
    #define mpfr_nextbelow dspba_mpfr_nextbelow
    #define mpfr_nexttoward dspba_mpfr_nexttoward
    #define mpfr_printf dspba_mpfr_printf
    #define mpfr_asprintf dspba_mpfr_asprintf
    #define mpfr_sprintf dspba_mpfr_sprintf
    #define mpfr_snprintf dspba_mpfr_snprintf
    #define mpfr_pow dspba_mpfr_pow
    #define mpfr_pow_si dspba_mpfr_pow_si
    #define mpfr_pow_ui dspba_mpfr_pow_ui
    #define mpfr_ui_pow_ui dspba_mpfr_ui_pow_ui
    #define mpfr_ui_pow dspba_mpfr_ui_pow
    #define mpfr_pow_z dspba_mpfr_pow_z
    #define mpfr_sqrt dspba_mpfr_sqrt
    #define mpfr_sqrt_ui dspba_mpfr_sqrt_ui
    #define mpfr_rec_sqrt dspba_mpfr_rec_sqrt
    #define mpfr_add dspba_mpfr_add
    #define mpfr_sub dspba_mpfr_sub
    #define mpfr_mul dspba_mpfr_mul
    #define mpfr_div dspba_mpfr_div
    #define mpfr_add_ui dspba_mpfr_add_ui
    #define mpfr_sub_ui dspba_mpfr_sub_ui
    #define mpfr_ui_sub dspba_mpfr_ui_sub
    #define mpfr_mul_ui dspba_mpfr_mul_ui
    #define mpfr_div_ui dspba_mpfr_div_ui
    #define mpfr_ui_div dspba_mpfr_ui_div
    #define mpfr_add_si dspba_mpfr_add_si
    #define mpfr_sub_si dspba_mpfr_sub_si
    #define mpfr_si_sub dspba_mpfr_si_sub
    #define mpfr_mul_si dspba_mpfr_mul_si
    #define mpfr_div_si dspba_mpfr_div_si
    #define mpfr_si_div dspba_mpfr_si_div
    #define mpfr_add_d dspba_mpfr_add_d
    #define mpfr_sub_d dspba_mpfr_sub_d
    #define mpfr_d_sub dspba_mpfr_d_sub
    #define mpfr_mul_d dspba_mpfr_mul_d
    #define mpfr_div_d dspba_mpfr_div_d
    #define mpfr_d_div dspba_mpfr_d_div
    #define mpfr_sqr dspba_mpfr_sqr
    #define mpfr_const_pi dspba_mpfr_const_pi
    #define mpfr_const_log2 dspba_mpfr_const_log2
    #define mpfr_const_euler dspba_mpfr_const_euler
    #define mpfr_const_catalan dspba_mpfr_const_catalan
    #define mpfr_agm dspba_mpfr_agm
    #define mpfr_log dspba_mpfr_log
    #define mpfr_log2 dspba_mpfr_log2
    #define mpfr_log10 dspba_mpfr_log10
    #define mpfr_log1p dspba_mpfr_log1p
    #define mpfr_log_ui dspba_mpfr_log_ui
    #define mpfr_exp dspba_mpfr_exp
    #define mpfr_exp2 dspba_mpfr_exp2
    #define mpfr_exp10 dspba_mpfr_exp10
    #define mpfr_expm1 dspba_mpfr_expm1
    #define mpfr_eint dspba_mpfr_eint
    #define mpfr_li2 dspba_mpfr_li2
    #define mpfr_cmp dspba_mpfr_cmp
    #define mpfr_cmp3 dspba_mpfr_cmp3
    #define mpfr_cmp_d dspba_mpfr_cmp_d
    #define mpfr_cmp_ld dspba_mpfr_cmp_ld
    #define mpfr_cmpabs dspba_mpfr_cmpabs
    #define mpfr_cmp_ui dspba_mpfr_cmp_ui
    #define mpfr_cmp_si dspba_mpfr_cmp_si
    #define mpfr_cmp_ui_2exp dspba_mpfr_cmp_ui_2exp
    #define mpfr_cmp_si_2exp dspba_mpfr_cmp_si_2exp
    #define mpfr_reldiff dspba_mpfr_reldiff
    #define mpfr_eq dspba_mpfr_eq
    #define mpfr_sgn dspba_mpfr_sgn
    #define mpfr_mul_2exp dspba_mpfr_mul_2exp
    #define mpfr_div_2exp dspba_mpfr_div_2exp
    #define mpfr_mul_2ui dspba_mpfr_mul_2ui
    #define mpfr_div_2ui dspba_mpfr_div_2ui
    #define mpfr_mul_2si dspba_mpfr_mul_2si
    #define mpfr_div_2si dspba_mpfr_div_2si
    #define mpfr_rint dspba_mpfr_rint
    #define mpfr_roundeven dspba_mpfr_roundeven
    #define mpfr_round dspba_mpfr_round
    #define mpfr_trunc dspba_mpfr_trunc
    #define mpfr_ceil dspba_mpfr_ceil
    #define mpfr_floor dspba_mpfr_floor
    #define mpfr_rint_roundeven dspba_mpfr_rint_roundeven
    #define mpfr_rint_round dspba_mpfr_rint_round
    #define mpfr_rint_trunc dspba_mpfr_rint_trunc
    #define mpfr_rint_ceil dspba_mpfr_rint_ceil
    #define mpfr_rint_floor dspba_mpfr_rint_floor
    #define mpfr_frac dspba_mpfr_frac
    #define mpfr_modf dspba_mpfr_modf
    #define mpfr_remquo dspba_mpfr_remquo
    #define mpfr_remainder dspba_mpfr_remainder
    #define mpfr_fmod dspba_mpfr_fmod
    #define mpfr_fmodquo dspba_mpfr_fmodquo
    #define mpfr_fits_ulong_p dspba_mpfr_fits_ulong_p
    #define mpfr_fits_slong_p dspba_mpfr_fits_slong_p
    #define mpfr_fits_uint_p dspba_mpfr_fits_uint_p
    #define mpfr_fits_sint_p dspba_mpfr_fits_sint_p
    #define mpfr_fits_ushort_p dspba_mpfr_fits_ushort_p
    #define mpfr_fits_sshort_p dspba_mpfr_fits_sshort_p
    #define mpfr_fits_uintmax_p dspba_mpfr_fits_uintmax_p
    #define mpfr_fits_intmax_p dspba_mpfr_fits_intmax_p
    #define mpfr_extract dspba_mpfr_extract
    #define mpfr_swap dspba_mpfr_swap
    #define mpfr_dump dspba_mpfr_dump
    #define mpfr_nan_p dspba_mpfr_nan_p
    #define mpfr_inf_p dspba_mpfr_inf_p
    #define mpfr_number_p dspba_mpfr_number_p
    #define mpfr_integer_p dspba_mpfr_integer_p
    #define mpfr_zero_p dspba_mpfr_zero_p
    #define mpfr_regular_p dspba_mpfr_regular_p
    #define mpfr_greater_p dspba_mpfr_greater_p
    #define mpfr_greaterequal_p dspba_mpfr_greaterequal_p
    #define mpfr_less_p dspba_mpfr_less_p
    #define mpfr_lessequal_p dspba_mpfr_lessequal_p
    #define mpfr_lessgreater_p dspba_mpfr_lessgreater_p
    #define mpfr_equal_p dspba_mpfr_equal_p
    #define mpfr_unordered_p dspba_mpfr_unordered_p
    #define mpfr_atanh dspba_mpfr_atanh
    #define mpfr_acosh dspba_mpfr_acosh
    #define mpfr_asinh dspba_mpfr_asinh
    #define mpfr_cosh dspba_mpfr_cosh
    #define mpfr_sinh dspba_mpfr_sinh
    #define mpfr_tanh dspba_mpfr_tanh
    #define mpfr_sinh_cosh dspba_mpfr_sinh_cosh
    #define mpfr_sech dspba_mpfr_sech
    #define mpfr_csch dspba_mpfr_csch
    #define mpfr_coth dspba_mpfr_coth
    #define mpfr_acos dspba_mpfr_acos
    #define mpfr_asin dspba_mpfr_asin
    #define mpfr_atan dspba_mpfr_atan
    #define mpfr_sin dspba_mpfr_sin
    #define mpfr_sin_cos dspba_mpfr_sin_cos
    #define mpfr_cos dspba_mpfr_cos
    #define mpfr_tan dspba_mpfr_tan
    #define mpfr_atan2 dspba_mpfr_atan2
    #define mpfr_sec dspba_mpfr_sec
    #define mpfr_csc dspba_mpfr_csc
    #define mpfr_cot dspba_mpfr_cot
    #define mpfr_hypot dspba_mpfr_hypot
    #define mpfr_erf dspba_mpfr_erf
    #define mpfr_erfc dspba_mpfr_erfc
    #define mpfr_cbrt dspba_mpfr_cbrt
    #define mpfr_root dspba_mpfr_root
    #define mpfr_rootn_ui dspba_mpfr_rootn_ui
    #define mpfr_gamma dspba_mpfr_gamma
    #define mpfr_gamma_inc dspba_mpfr_gamma_inc
    #define mpfr_beta dspba_mpfr_beta
    #define mpfr_lngamma dspba_mpfr_lngamma
    #define mpfr_lgamma dspba_mpfr_lgamma
    #define mpfr_digamma dspba_mpfr_digamma
    #define mpfr_zeta dspba_mpfr_zeta
    #define mpfr_zeta_ui dspba_mpfr_zeta_ui
    #define mpfr_fac_ui dspba_mpfr_fac_ui
    #define mpfr_j0 dspba_mpfr_j0
    #define mpfr_j1 dspba_mpfr_j1
    #define mpfr_jn dspba_mpfr_jn
    #define mpfr_y0 dspba_mpfr_y0
    #define mpfr_y1 dspba_mpfr_y1
    #define mpfr_yn dspba_mpfr_yn
    #define mpfr_ai dspba_mpfr_ai
    #define mpfr_min dspba_mpfr_min
    #define mpfr_max dspba_mpfr_max
    #define mpfr_dim dspba_mpfr_dim
    #define mpfr_mul_z dspba_mpfr_mul_z
    #define mpfr_div_z dspba_mpfr_div_z
    #define mpfr_add_z dspba_mpfr_add_z
    #define mpfr_sub_z dspba_mpfr_sub_z
    #define mpfr_z_sub dspba_mpfr_z_sub
    #define mpfr_cmp_z dspba_mpfr_cmp_z
    #define mpfr_fma dspba_mpfr_fma
    #define mpfr_fms dspba_mpfr_fms
    #define mpfr_fmma dspba_mpfr_fmma
    #define mpfr_fmms dspba_mpfr_fmms
    #define mpfr_sum dspba_mpfr_sum
    #define mpfr_free_cache dspba_mpfr_free_cache
    #define mpfr_free_cache2 dspba_mpfr_free_cache2
    #define mpfr_free_pool dspba_mpfr_free_pool
    #define mpfr_mp_memory_cleanup dspba_mpfr_mp_memory_cleanup
    #define mpfr_subnormalize dspba_mpfr_subnormalize
    #define mpfr_strtofr dspba_mpfr_strtofr
    #define mpfr_round_nearest_away_begin dspba_mpfr_round_nearest_away_begin
    #define mpfr_round_nearest_away_end dspba_mpfr_round_nearest_away_end
    #define mpfr_custom_get_size dspba_mpfr_custom_get_size
    #define mpfr_custom_init dspba_mpfr_custom_init
    #define mpfr_custom_get_significand dspba_mpfr_custom_get_significand
    #define mpfr_custom_get_exp dspba_mpfr_custom_get_exp
    #define mpfr_custom_move dspba_mpfr_custom_move
    #define mpfr_custom_init_set dspba_mpfr_custom_init_set
    #define mpfr_custom_get_kind dspba_mpfr_custom_get_kind
    #define __gmpfr_inp_str __gdspba_mpfr_inp_str
    #define __gmpfr_out_str __gdspba_mpfr_out_str
    #define __gmpfr_fprintf __gdspba_mpfr_fprintf
    #define __gmpfr_fpif_export __gdspba_mpfr_fpif_export
    #define __gmpfr_fpif_import __gdspba_mpfr_fpif_import
#endif