/** 
 * csl.h declares/defines the functions/types used by
 * the DSPBA software model implementation.
 *
 * csl is short for 'C++ software model Support Library'
 *
 * The csl library and generated software models
 * are header-only to simplify user integration
 * maximize support for all possible use cases.
 * For the most typical use case, where the model
 * is accessed in a single source file by the user
 * this can speed up compilation as it effectively
 * results in a unity build. Conversely, including
 * potentially large software model headers across
 * a project could significant increase overall compile
 * times so such an approach is best avoided.
 */
#pragma once

#ifndef CSL_CSL_H
#define CSL_CSL_H

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <type_traits>

#if _MSC_VER
#if _MSVC_LANG < 201402L
#error "The software model code requires C++14 or later."
#endif
#else
#if __cplusplus < 201402L
#error "The software model code requires C++14 or later."
#endif
#endif

#ifdef CSL_USE_GMP
#include "gmp.h"
#endif
#ifdef CSL_USE_MPFR
#include "mpfr.h"
#endif

#ifdef _MSC_VER
#define CSL_FORCE_INLINE __forceinline
#else
#define CSL_FORCE_INLINE inline __attribute__((always_inline))
#endif

namespace csl
{

/**
 * Temporary storage span that is used to prevent dynamic allocation
 * of multi-precision integer types during simulations.
 */
struct temps_span;

/**
 * Application must implement error, warning and info functions.
 * Info may be ignored, but warnings and errors indicate problems with the
 * design so should be handled appropriately.
 */
extern void error(const char* msg);
extern void warning(const char* msg);
extern void info(const char* msg);

/** 
 * Utility functions for native types
 */
constexpr bool is_zero(int64_t& value) noexcept;
constexpr bool is_one(int64_t& value) noexcept;
constexpr void negate(int64_t& result, int64_t v) noexcept;
constexpr void complement(int64_t& result, int64_t v) noexcept;
constexpr uint64_t bit_mask_u64(size_t amount) noexcept;
constexpr int64_t bit_mask_i64(size_t amount) noexcept;
constexpr void mask_lower(int64_t& result, int64_t value, size_t width) noexcept;
constexpr bool test_bit(size_t bit_position, int64_t value) noexcept;
constexpr void mask(int64_t& result, int64_t v, size_t bit_width) noexcept;
constexpr void set_upper(int64_t& value, size_t bit_width) noexcept;
constexpr void not_n(int64_t& result, int64_t value, int bit_width) noexcept;
constexpr void boolean_complement(int64_t& result, int64_t v) noexcept;
constexpr void set_bit(int64_t& result, size_t position, bool clear) noexcept;
constexpr void set(int64_t& dst, int64_t src) noexcept;
constexpr void ld_exp(int64_t& result, int64_t value, int exp2) noexcept;
constexpr int64_t abs(int64_t value) noexcept;
constexpr int64_t pow2(int64_t value) noexcept;
constexpr int64_t to_i64(int64_t v) noexcept;
constexpr uint64_t to_u64(int64_t value) noexcept;
constexpr void safe_sub(int64_t& result, int64_t a, int64_t b, temps_span temps) noexcept;
float flush_subnormals(float f) noexcept;
float flush_signed_nan(float f) noexcept;

/**
 * As per std::fill_n.
 * <algorithm> is a large header so avoid including it just for this and std::min.
 */
template <class It, class T> It fill_n(It first, size_t count, const T& value) noexcept
{
    for (size_t i = 0; i < count; i++)
    {
        *first++ = value;
    }
    return first;
}

/**
 * Returns the minimum of a and b
 */
template <class T> const T& min(const T& a, const T& b) noexcept
{
    return (b < a) ? b : a;
}

/**
 * Returns the array value if it is in bounds, else 0
 */
template <class T, size_t N> T checked_array_value(const T (&values)[N], int64_t index) noexcept
{
    return ((index < N) && (index >= 0)) ? values[index] : T(0);
}

/**
 * Wrapper around mpz_t to automate lifetime management
 * and provide conversion functions.
 */
#ifdef CSL_USE_GMP

static_assert(sizeof(mpir_ui) == sizeof(uint64_t), "Expected mpir_ui to be size of uint64_t");
static_assert(sizeof(mpir_si) == sizeof(int64_t), "Expected mpir_si to be size of int64_t");

struct mp_int_info
{
    uint32_t offset;
    uint16_t count;
    char sign;
};

class mp_int
{
public:
    mp_int();
    mp_int(const char* str);
    mp_int(int64_t v);
    mp_int(uint64_t v);
    mp_int(int32_t v);
    mp_int(uint32_t v);

    mp_int(const mp_int& other) = delete;
    mp_int(mp_int&& other);
    ~mp_int() noexcept;

    /**
     * Returns the value of mpz as a string,
     * e.g. returns "2" if the underlying value is 2
     */
    void str(char* dst, size_t max_size) const noexcept;
    /**
     * Returns the binary value of mpz as a string,
     * e.g. returns "10" if the underlying value is 2
     */
    void str_bin(char* dst, size_t max_size) const noexcept;
    /**
     * Sets the value from n integers, where the first element contains the LSB
     */
    void set_from_uint_array(const uint32_t* array, size_t n, size_t bit_width, temps_span temps) noexcept;
    /**
     * Gets the value of this mp int as an array of integers
     */
    void get_as_uint_array(uint32_t* array, size_t n, temps_span temps) const noexcept;

    /**
     * Assignment operators
     */
    mp_int& operator=(const mp_int& other) noexcept;
    mp_int& operator=(mp_int&& other) noexcept;
    mp_int& operator=(const mpf_t val) noexcept;
    mp_int& operator=(const mpz_t val) noexcept;
    mp_int& operator=(int64_t i) noexcept;
    mp_int& operator=(uint64_t i) noexcept;
    mp_int& operator=(int32_t i) noexcept;
    mp_int& operator=(uint32_t i) noexcept;
    mp_int& operator=(const char* s) noexcept;

    /** Get the underlying gmp type */
    constexpr mpz_t& get() noexcept;
    constexpr const mpz_t& get() const noexcept;

private:
    mpz_t m_value;
};

inline void fill_mpz_data(mp_int& dst, const uint64_t* words, uint16_t count, int8_t sign)
{
    mpz_import(dst.get(), count, -1, sizeof(uint64_t), 0, 0, words);
    if (sign < 0)
    {
        mpz_neg(dst.get(), dst.get());
    }
}

inline void fill_mpz_data(mp_int& dst, const uint64_t* words, const mp_int_info* infos, size_t index)
{
    mpz_import(dst.get(), infos[index].count, -1, sizeof(uint64_t), 0, 0, words + infos[index].offset);
    if (infos[index].sign < 0)
    {
        mpz_neg(dst.get(), dst.get());
    }
}

/**
 * Conversion functions
 */
int64_t to_i64(const mp_int& value) noexcept;
uint64_t to_u64(const mp_int& value) noexcept;
int32_t to_i32(const mp_int& value) noexcept;
uint32_t to_u32(const mp_int& value) noexcept;

/**
 * Inline operators
 */
bool operator==(const mp_int& a, const mp_int& b) noexcept;
bool operator==(const mp_int& a, int32_t b) noexcept;
bool operator==(const mp_int& a, uint32_t b) noexcept;
bool operator==(const mp_int& a, int64_t b) noexcept;
bool operator==(const mp_int& a, uint64_t b) noexcept;
bool operator==(int32_t a, const mp_int& b) noexcept;
bool operator==(uint32_t a, const mp_int& b) noexcept;
bool operator==(int64_t a, const mp_int& b) noexcept;
bool operator==(uint64_t a, const mp_int& b) noexcept;
bool operator!=(const mp_int& a, const mp_int& b) noexcept;
bool operator!=(const mp_int& a, int64_t b) noexcept;
bool operator!=(const mp_int& a, uint64_t b) noexcept;
bool operator!=(const mp_int& a, int32_t b) noexcept;
bool operator!=(const mp_int& a, uint32_t b) noexcept;
bool operator!=(int32_t a, const mp_int& b) noexcept;
bool operator!=(int64_t a, const mp_int& b) noexcept;
bool operator!=(uint32_t a, const mp_int& b) noexcept;
bool operator!=(uint64_t a, const mp_int& b) noexcept;
bool operator<(const mp_int& a, const mp_int& b) noexcept;
bool operator<(const mp_int& a, int64_t b) noexcept;
bool operator>(const mp_int& a, const mp_int& b) noexcept;
bool operator>(const mp_int& a, int64_t b) noexcept;
bool operator<=(const mp_int& a, const mp_int& b) noexcept;
bool operator<=(const mp_int& a, int64_t b) noexcept;
bool operator>=(const mp_int& a, const mp_int& b) noexcept;
bool operator>=(const mp_int& a, int64_t b) noexcept;

mp_int& operator%=(mp_int& a, const mp_int& b) noexcept;
mp_int& operator+=(mp_int& a, const mp_int& b) noexcept;
mp_int& operator-=(mp_int& a, const mp_int& b) noexcept;
mp_int& operator*=(mp_int& a, const mp_int& b) noexcept;
mp_int& operator|=(mp_int& a, const mp_int& b) noexcept;
mp_int& operator&=(mp_int& a, const mp_int& b) noexcept;
mp_int& operator^=(mp_int& a, const mp_int& b) noexcept;
mp_int& operator<<=(mp_int& num, mp_bitcnt_t amount) noexcept;
mp_int& operator>>=(mp_int& num, mp_bitcnt_t amount) noexcept;
mp_int& operator%=(mp_int& a, int64_t b) noexcept;
mp_int& operator+=(mp_int& a, int64_t b) noexcept;
mp_int& operator-=(mp_int& a, int64_t b) noexcept;
mp_int& operator*=(mp_int& a, int64_t b) noexcept;

/**
 * Utility functions for multi-precision types
 */
bool is_zero(const mp_int& value) noexcept;
bool is_one(const mp_int& value) noexcept;
void negate(mp_int& result, const mp_int& v) noexcept;
void complement(mp_int& result, const mp_int& v) noexcept;

void bit_mask_mp_int(mp_int& result, size_t amount) noexcept;
uint64_t to_u64(const mp_int& value) noexcept;
void mask(mp_int& result, const mp_int& v, size_t bit_width) noexcept;
void set(int64_t& dst, const mp_int& src) noexcept;
void set(mp_int& dst, int64_t src) noexcept;
void set(mp_int& dst, const mp_int& src) noexcept;
void mask_lower(mp_int& result, const mp_int& value, size_t width) noexcept;
bool test_bit(size_t bit_position, const mp_int& value) noexcept;
void ld_exp(mp_int& result, const mp_int& value, int exp2) noexcept;
void set_upper(mp_int& value, size_t bit_width) noexcept;
void not_n(mp_int& result, const mp_int& value, int bit_width) noexcept;
void boolean_complement(mp_int& result, const mp_int& v) noexcept;
void set_bit(mp_int& result, size_t position, bool clear) noexcept;

/**
 * Returns the array value if it is in bounds, else 0
 */
template <class T, size_t N> T checked_array_value(const T (&values)[N], const mp_int& index) noexcept
{
    int64_t index_i64 = to_i64(index);
    return ((index_i64 < N) && (index_i64 >= 0)) ? values[index_i64] : T(0);
}

#endif

/**
 * Temporaries used to avoid dynamic allocations during execution
 */
struct mp_int_temps
{
    static constexpr size_t capacity = 8;
#ifdef CSL_USE_GMP
    mp_int values[capacity];
#endif
};

struct temps_span
{
    temps_span() = default;
#ifdef CSL_USE_GMP
    temps_span(mp_int* values, size_t count) : m_values(values), m_count(count) {}
#endif

    temps_span(mp_int_temps& temps, size_t offset = 0) :
        m_count(temps.capacity - offset)
#ifdef CSL_USE_GMP
        ,
        m_values(temps.values + offset)
#endif
    {
        if (offset >= temps.capacity)
        {
            error("Multi-precision temporaries exhausted");
        }
    }

    temps_span next(uint32_t offset)
    {
#ifdef CSL_USE_GMP
        if (offset >= m_count)
        {
            error("Multi-precision temporaries exhausted");
        }
        return {m_values + offset, m_count - offset};
#else
        return *this;
#endif
    }

#ifdef CSL_USE_GMP
    mp_int& at(size_t index)
    {
        if (index < m_count)
        {
            return m_values[index];
        }
        error("Multi-precision temporaries exhausted");
        exit(1);
    }
#endif

private:
#ifdef CSL_USE_GMP
    mp_int* m_values = nullptr;
#endif
    size_t m_count = 0;
};

#ifdef CSL_USE_MPFR
struct mp_float_init_token
{};

/**
 * Wrapper around mpfr_t to automate lifetime management
 */
struct mp_float
{
    mp_float(mp_prec_t precision);
    mp_float(mp_float_init_token);
    ~mp_float() noexcept;

    mp_float& operator=(const mp_float& x) noexcept;

    constexpr mpfr_t& get() noexcept;
    constexpr const mpfr_t& get() const noexcept;

private:
    mpfr_t m_value;
};

/**
 * Simple wrapper around a float that provides some conversion
 * functions
 */
class fp32
{
public:
    fp32(int64_t in);
#ifdef CSL_USE_GMP
    fp32(const mp_int& in);
#endif
    fp32(float f);

    constexpr float get() const;
    uint32_t get_u32() const;
    constexpr void set(float f);
    void set(uint32_t i);

private:
    float m_value;
};

void set(int64_t& dst, fp32 src) noexcept;
void set(mp_float& dst, int64_t src) noexcept;
void set(mp_float& dst, const mp_int& src) noexcept;

void flush_bad_values(int exponent_width, mpfr_t& o) noexcept;
void mult_fp16_extend(mpfr_t& o, mpfr_t& a, mpfr_t& b) noexcept;
void transfer_fp(int exponent_width, int mantissa_width, mp_float& dst, const mp_int& src) noexcept;
#ifdef CSL_USE_GMP
void float_pack_bits_default(mpfr_t& ref, int wExp, int wFrac, mpz_t& z0, bool subnormals_to_zero) noexcept;
#endif

#endif

/**
 * Configuration parameters for the FIFO step
 */
struct fifo_params
{
    /** number of words of data in FIFO */
    int depth;
    /** the number of words that must be in FIFO before filled is set high */
    int fill_threshold;
    /** the number of words that are in FIFO before fullness flag is set */
    int full_threshold;
    int write_latency;
    int user_sclr;
    int base_index;
};

/**
 * Configuration parameters for the
 * enable generator step
 */
struct enable_gen_params
{
    int decim;
    int interp;
    int n_chans;
    int compute_cycle_length;
    int adder_width;
    int valid_inc;
    int ena_inc;
    int last_enable_inc;
    int num_forced_zeros;
    bool use_sequencer_disable;
    bool use_delay_disable;
};

/**
 * Configuration parameters for the
 * chain mult add step
 */
struct cma_add_params
{
    int pipeline_depth;
    int systolic_region_count;
    int systolic_region_size;
    int n_mults;
};

#ifdef CSL_USE_GMP
void safe_sub(mp_int& result, int64_t a, int64_t b, temps_span temps) noexcept;
void safe_sub(mp_int& result, const mp_int& a, int64_t b, temps_span temps) noexcept;
void safe_sub(mp_int& result, int64_t a, const mp_int& b, temps_span temps) noexcept;
void safe_sub(mp_int& result, const mp_int& a, const mp_int& b, temps_span temps) noexcept;
void safe_sub(int64_t& result, const mp_int& a, int64_t b, temps_span temps) noexcept;
void safe_sub(int64_t& result, int64_t a, const mp_int& b, temps_span temps) noexcept;
void safe_sub(int64_t& result, const mp_int& a, const mp_int& b, temps_span temps) noexcept;
#endif

/** Step implementations start */

/**
 * Returns true if the given address is valid within the size
 * of the memory
 */
constexpr bool dual_mem_is_valid_address(int size, int addr) noexcept;
/**
 * Gets a word from the memory.
 * If the given address is invalid, sets the output value to 0xcdcdcdcd.
 */
void dual_mem_get_word(int index, int size, int byte_width, int64_t* store, int64_t& word, int64_t uaddr, int word_size,
                       temps_span temps) noexcept;
/**
 * Puts a word to the memory.
 * Returns true if successful (always unless address is invalid)
 */
bool dual_mem_put_word(int index, int size, int byte_width, int64_t* store, const int64_t& word, int64_t uaddr, int word_size,
                       temps_span temps) noexcept;

#ifdef CSL_USE_GMP
void dual_mem_get_word(int index, int size, int byte_width, mp_int* store, mp_int& word, const mp_int& addr, int word_size,
                       temps_span temps) noexcept;
void dual_mem_get_word(int index, int size, int byte_width, mp_int* store, mp_int& word, int64_t uaddr, int word_size,
                       temps_span temps) noexcept;
void dual_mem_get_word(int index, int size, int byte_width, int64_t* store, int64_t& word, const mp_int& addr, int word_size,
                       temps_span temps) noexcept;
bool dual_mem_put_word(int index, int size, int byte_width, mp_int* store, const mp_int& word, int64_t uaddr, int word_size,
                       temps_span temps) noexcept;
bool dual_mem_put_word(int index, int size, int byte_width, mp_int* store, const mp_int& word, const mp_int& addr, int word_size,
                       temps_span temps) noexcept;
bool dual_mem_put_word(int index, int size, int byte_width, int64_t* store, const int64_t& word, const mp_int& addr, int word_size,
                       temps_span temps) noexcept;
#endif

#ifdef CSL_USE_MPFR
fp32 fp_mul_impl(const fp32& a, const fp32& b) noexcept;
#endif

/**
 * Utility for buffering model outputs
 * of type T for latency N
 */
template <typename T, size_t N> struct delay_correction
{
    using value_type = typename std::conditional<std::is_fundamental<T>::value, T, const T&>::type;

    delay_correction() noexcept
    {
        reset();
    }

    void reset() noexcept
    {
        for (size_t i = 0; i < N; ++i)
        {
            m_buffer[i] = 0;
        }
        m_curr = 0;
    }

    value_type delay(int64_t value) noexcept
    {
        m_curr = m_buffer[m_offset];
        set(m_buffer[m_offset], value);
        m_offset = (m_offset + 1) % N;
        return m_curr;
    }

#ifdef CSL_USE_GMP
    value_type delay(const mp_int& value) noexcept
    {
        m_curr = m_buffer[m_offset];
        set(m_buffer[m_offset], value);
        m_offset = (m_offset + 1) % N;
        return m_curr;
    }
#endif

    constexpr const T& get() const noexcept
    {
        return m_curr;
    }

private:
    size_t m_offset = 0;
    T m_curr;
    T m_buffer[N];
};

/**
 * Utility for buffering model output arrays of type T
 * for latency N.
 * Used for the arrays that wrap arbitrary size types in the
 * HLD library function wrappers.
 */
template <typename T, size_t N, size_t COUNT> struct delay_correction_array
{
    constexpr delay_correction_array() noexcept
    {
        reset();
    }

    constexpr void reset() noexcept
    {
        for (size_t i = 0; i < COUNT; ++i)
        {
            fill_n(m_buffer[i], N, T(0));
        }
    }

    template <unsigned int ARR_COUNT> constexpr T* delay(const T (&values)[ARR_COUNT]) noexcept
    {
        static_assert(ARR_COUNT == COUNT);
        for (size_t i = 0; i < COUNT; ++i)
        {
            m_output[i] = m_buffer[i][m_offset];
            m_buffer[i][m_offset] = values[i];
        }
        m_offset = (m_offset + 1) % N;
        return m_output;
    }

private:
    size_t m_offset = 0;
    T m_buffer[COUNT][N];
    T m_output[COUNT];
};

/** Generated steps begin */

// Revision: 2

void step_fp_add(int64_t& iq, int64_t ia, int64_t ib) noexcept;
void step_fp_sub(int64_t& iq, int64_t ia, int64_t ib) noexcept;
void step_fp_mul(int64_t& iq, int64_t ia, int64_t ib) noexcept;
void step_add(int64_t& iq, int64_t ia, int64_t ib, temps_span temps) noexcept;
void step_mul(int64_t& iq, int64_t ia, int64_t ib, temps_span temps) noexcept;
void step_sub(int64_t& iq, int64_t ia, int64_t ib, temps_span temps) noexcept;
void step_addsub(int64_t ctrl, int64_t& iq, int64_t ia, int64_t ib, temps_span temps) noexcept;
void step_subadd(int64_t ctrl, int64_t& iq, int64_t ia, int64_t ib, temps_span temps) noexcept;
void step_const(int64_t& iq, int64_t ia) noexcept;
void step_and(int64_t& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_or(int64_t& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_xor(int64_t& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_nand(int64_t& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_nor(int64_t& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_nxor(int64_t& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_reducing_or(int64_t& iq, int64_t ia) noexcept;
void step_reducing_nor(int64_t& iq, int64_t ia) noexcept;
void step_reducing_and(int64_t& iq, int64_t ia, size_t bit_width, bool is_signed, temps_span temps) noexcept;
void step_ld_exp(int64_t& iq, int64_t ia, int64_t ib, bool reverse, temps_span temps) noexcept;
void step_equal(int64_t& iq, int64_t ia, int64_t ib) noexcept;
void step_nequal(int64_t& iq, int64_t ia, int64_t ib) noexcept;
void step_reducing_nand(int64_t& iq, int64_t ia, size_t bit_width, bool is_signed, temps_span temps) noexcept;
void step_reducing_xor(int64_t& iq, int64_t ia, size_t bit_width) noexcept;
void step_reducing_nxor(int64_t& iq, int64_t ia, size_t bit_width) noexcept;
void step_bit_extract(int64_t& iq, int64_t ia, size_t width, bool signed_extend, int bit_pos, temps_span temps) noexcept;
void step_biased_round(int64_t& iq, int64_t ia, int bit, temps_span temps) noexcept;
void step_unbiased_round(int64_t& iq, int64_t ia, int bit, temps_span temps) noexcept;
void step_bit_reverse(int64_t& iq, int64_t ia, size_t bit_width) noexcept;
void step_sign_bit(int64_t& iq, int64_t ia, size_t bit_width) noexcept;
void step_nsign_bit(int64_t& iq, int64_t ia, size_t bit_width) noexcept;
void step_shift_right(int64_t& iq, int64_t ia, size_t amount, temps_span temps) noexcept;
void step_shift_left(int64_t& iq, int64_t ia, size_t amount, temps_span temps) noexcept;
void step_test_bit(int64_t& iq, int64_t ia, size_t bit_position) noexcept;
void step_set_bit(int64_t& iq, int64_t ia, size_t bit_position) noexcept;
void step_not(int64_t& iq, int64_t ia, int bit_width, temps_span temps) noexcept;
void step_not_signed(int64_t& iq, int64_t ia, int bit_width, temps_span temps) noexcept;
void step_sequencer(int64_t& state, int64_t& iq, int64_t ia, size_t offset, int64_t mod, int64_t cross, temps_span temps) noexcept;
void step_reduce(int64_t& iq, int64_t ia, size_t bit_width, temps_span temps) noexcept;
void step_counter(int64_t& counter, int64_t& iq, int64_t ia, size_t offset, int64_t inc, int64_t mod) noexcept;
void step_bit_combine(int64_t& iq, int64_t ia, int64_t ib, size_t num_bits, size_t index, size_t bit_pos, size_t next_bit_pos,
                      temps_span temps) noexcept;
bool get_lookup_value(const int64_t* values, uint64_t n, uint64_t key, int64_t& out) noexcept;
bool get_sparse_lookup_value(const int64_t* values, const unsigned char* exists, uint64_t n, uint64_t key, int64_t& out) noexcept;
bool get_sorted_lookup_value(const uint64_t* keys, const int64_t* values, uint64_t n, uint64_t key, int64_t& out) noexcept;
void step_lookup(const int64_t* values, uint64_t n, uint64_t offset, int64_t& iq, int64_t ia_in, temps_span temps);
void step_lookup_with_valid(const int64_t* values, uint64_t n, uint64_t offset, int64_t& iq, int64_t& ivalid, int64_t ia_in,
                            temps_span temps);
void step_decode(int64_t& iq0, int64_t ia, int64_t ib, int32_t low, int32_t high, int32_t decode, temps_span temps) noexcept;
void step_decode(int64_t& iq0, int64_t& iq1, int64_t ia, int64_t ib, int32_t low, int32_t high, int32_t decode,
                 temps_span temps) noexcept;
void step_fp_acc(int64_t& iacc, int64_t control, int64_t& iq, int64_t ix) noexcept;
void step_fp_mult_acc(int64_t& iacc, int64_t control, int64_t& iq, int64_t ix, int64_t iy) noexcept;
void step_loadable_counter(int64_t& state_counter, int64_t& state_mod, int64_t& state_inc, int64_t& iq, int64_t ienable, int64_t iload,
                           int64_t load_count, int64_t load_mod, int64_t load_inc) noexcept;
void step_enable_generator(const enable_gen_params& params, int64_t valid, int64_t enable_in, int64_t& enable_out, int64_t& count,
                           int64_t& en_count, int64_t& zero_force_count, int64_t& enable,
                           int64_t& enable_zero_forcing_sequencer) noexcept;
void step_cma_add(const int64_t* const prod_arr, int64_t* const sums_arr, const cma_add_params& params, int64_t sub_ctrl,
                  int64_t neg_ctrl, int64_t& region_sum) noexcept;
void step_fifo(int64_t* store, const fifo_params& params, int64_t& data, int64_t write_en, int64_t read_en, int64_t flush) noexcept;

#ifdef CSL_USE_GMP

void step_fp_add(int64_t& iq, int64_t ia, const mp_int& ib) noexcept;
void step_fp_add(int64_t& iq, const mp_int& ia, int64_t ib) noexcept;
void step_fp_add(int64_t& iq, const mp_int& ia, const mp_int& ib) noexcept;
void step_fp_add(mp_int& iq, int64_t ia, int64_t ib) noexcept;
void step_fp_add(mp_int& iq, int64_t ia, const mp_int& ib) noexcept;
void step_fp_add(mp_int& iq, const mp_int& ia, int64_t ib) noexcept;
void step_fp_add(mp_int& iq, const mp_int& ia, const mp_int& ib) noexcept;
void step_fp_sub(int64_t& iq, int64_t ia, const mp_int& ib) noexcept;
void step_fp_sub(int64_t& iq, const mp_int& ia, int64_t ib) noexcept;
void step_fp_sub(int64_t& iq, const mp_int& ia, const mp_int& ib) noexcept;
void step_fp_sub(mp_int& iq, int64_t ia, int64_t ib) noexcept;
void step_fp_sub(mp_int& iq, int64_t ia, const mp_int& ib) noexcept;
void step_fp_sub(mp_int& iq, const mp_int& ia, int64_t ib) noexcept;
void step_fp_sub(mp_int& iq, const mp_int& ia, const mp_int& ib) noexcept;
void step_fp_mul(int64_t& iq, int64_t ia, const mp_int& ib) noexcept;
void step_fp_mul(int64_t& iq, const mp_int& ia, int64_t ib) noexcept;
void step_fp_mul(int64_t& iq, const mp_int& ia, const mp_int& ib) noexcept;
void step_fp_mul(mp_int& iq, int64_t ia, int64_t ib) noexcept;
void step_fp_mul(mp_int& iq, int64_t ia, const mp_int& ib) noexcept;
void step_fp_mul(mp_int& iq, const mp_int& ia, int64_t ib) noexcept;
void step_fp_mul(mp_int& iq, const mp_int& ia, const mp_int& ib) noexcept;
void step_add(int64_t& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept;
void step_add(int64_t& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept;
void step_add(int64_t& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept;
void step_add(mp_int& iq, int64_t ia, int64_t ib, temps_span temps) noexcept;
void step_add(mp_int& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept;
void step_add(mp_int& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept;
void step_add(mp_int& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept;
void step_mul(int64_t& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept;
void step_mul(int64_t& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept;
void step_mul(int64_t& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept;
void step_mul(mp_int& iq, int64_t ia, int64_t ib, temps_span temps) noexcept;
void step_mul(mp_int& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept;
void step_mul(mp_int& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept;
void step_mul(mp_int& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept;
void step_sub(int64_t& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept;
void step_sub(int64_t& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept;
void step_sub(int64_t& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept;
void step_sub(mp_int& iq, int64_t ia, int64_t ib, temps_span temps) noexcept;
void step_sub(mp_int& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept;
void step_sub(mp_int& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept;
void step_sub(mp_int& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept;
void step_addsub(int64_t ctrl, int64_t& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept;
void step_addsub(int64_t ctrl, int64_t& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept;
void step_addsub(int64_t ctrl, int64_t& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept;
void step_addsub(int64_t ctrl, mp_int& iq, int64_t ia, int64_t ib, temps_span temps) noexcept;
void step_addsub(int64_t ctrl, mp_int& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept;
void step_addsub(int64_t ctrl, mp_int& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept;
void step_addsub(int64_t ctrl, mp_int& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept;
void step_addsub(const mp_int& ctrl, int64_t& iq, int64_t ia, int64_t ib, temps_span temps) noexcept;
void step_addsub(const mp_int& ctrl, int64_t& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept;
void step_addsub(const mp_int& ctrl, int64_t& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept;
void step_addsub(const mp_int& ctrl, int64_t& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept;
void step_addsub(const mp_int& ctrl, mp_int& iq, int64_t ia, int64_t ib, temps_span temps) noexcept;
void step_addsub(const mp_int& ctrl, mp_int& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept;
void step_addsub(const mp_int& ctrl, mp_int& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept;
void step_addsub(const mp_int& ctrl, mp_int& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept;
void step_subadd(int64_t ctrl, int64_t& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept;
void step_subadd(int64_t ctrl, int64_t& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept;
void step_subadd(int64_t ctrl, int64_t& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept;
void step_subadd(int64_t ctrl, mp_int& iq, int64_t ia, int64_t ib, temps_span temps) noexcept;
void step_subadd(int64_t ctrl, mp_int& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept;
void step_subadd(int64_t ctrl, mp_int& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept;
void step_subadd(int64_t ctrl, mp_int& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept;
void step_subadd(const mp_int& ctrl, int64_t& iq, int64_t ia, int64_t ib, temps_span temps) noexcept;
void step_subadd(const mp_int& ctrl, int64_t& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept;
void step_subadd(const mp_int& ctrl, int64_t& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept;
void step_subadd(const mp_int& ctrl, int64_t& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept;
void step_subadd(const mp_int& ctrl, mp_int& iq, int64_t ia, int64_t ib, temps_span temps) noexcept;
void step_subadd(const mp_int& ctrl, mp_int& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept;
void step_subadd(const mp_int& ctrl, mp_int& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept;
void step_subadd(const mp_int& ctrl, mp_int& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept;
void step_const(int64_t& iq, const mp_int& ia) noexcept;
void step_const(mp_int& iq, int64_t ia) noexcept;
void step_const(mp_int& iq, const mp_int& ia) noexcept;
void step_and(int64_t& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_and(int64_t& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_and(int64_t& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_and(mp_int& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_and(mp_int& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_and(mp_int& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_and(mp_int& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_or(int64_t& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_or(int64_t& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_or(int64_t& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_or(mp_int& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_or(mp_int& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_or(mp_int& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_or(mp_int& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_xor(int64_t& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_xor(int64_t& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_xor(int64_t& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_xor(mp_int& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_xor(mp_int& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_xor(mp_int& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_xor(mp_int& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_nand(int64_t& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_nand(int64_t& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_nand(int64_t& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_nand(mp_int& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_nand(mp_int& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_nand(mp_int& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_nand(mp_int& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_nor(int64_t& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_nor(int64_t& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_nor(int64_t& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_nor(mp_int& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_nor(mp_int& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_nor(mp_int& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_nor(mp_int& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_nxor(int64_t& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_nxor(int64_t& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_nxor(int64_t& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_nxor(mp_int& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_nxor(mp_int& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_nxor(mp_int& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept;
void step_nxor(mp_int& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept;
void step_reducing_or(int64_t& iq, const mp_int& ia) noexcept;
void step_reducing_or(mp_int& iq, int64_t ia) noexcept;
void step_reducing_or(mp_int& iq, const mp_int& ia) noexcept;
void step_reducing_nor(int64_t& iq, const mp_int& ia) noexcept;
void step_reducing_nor(mp_int& iq, int64_t ia) noexcept;
void step_reducing_nor(mp_int& iq, const mp_int& ia) noexcept;
void step_reducing_and(mp_int& iq, int64_t ia, size_t bit_width, bool is_signed, temps_span temps) noexcept;
void step_reducing_and(int64_t& iq, const mp_int& ia, size_t bit_width, bool is_signed, temps_span temps) noexcept;
void step_reducing_and(mp_int& iq, const mp_int& ia, size_t bit_width, bool is_signed, temps_span temps) noexcept;
void step_ld_exp(int64_t& iq, const mp_int& ia, int64_t ib, bool reverse, temps_span temps) noexcept;
void step_ld_exp(mp_int& iq, int64_t ia, int64_t ib, bool reverse, temps_span temps) noexcept;
void step_ld_exp(mp_int& iq, const mp_int& ia, int64_t ib, bool reverse, temps_span temps) noexcept;
void step_ld_exp(int64_t& iq, int64_t ia, const mp_int& ib, bool reverse, temps_span temps) noexcept;
void step_ld_exp(int64_t& iq, const mp_int& ia, const mp_int& ib, bool reverse, temps_span temps) noexcept;
void step_ld_exp(mp_int& iq, int64_t ia, const mp_int& ib, bool reverse, temps_span temps) noexcept;
void step_ld_exp(mp_int& iq, const mp_int& ia, const mp_int& ib, bool reverse, temps_span temps) noexcept;
void step_equal(int64_t& iq, int64_t ia, const mp_int& ib) noexcept;
void step_equal(int64_t& iq, const mp_int& ia, int64_t ib) noexcept;
void step_equal(int64_t& iq, const mp_int& ia, const mp_int& ib) noexcept;
void step_equal(mp_int& iq, int64_t ia, int64_t ib) noexcept;
void step_equal(mp_int& iq, int64_t ia, const mp_int& ib) noexcept;
void step_equal(mp_int& iq, const mp_int& ia, int64_t ib) noexcept;
void step_equal(mp_int& iq, const mp_int& ia, const mp_int& ib) noexcept;
void step_nequal(int64_t& iq, int64_t ia, const mp_int& ib) noexcept;
void step_nequal(int64_t& iq, const mp_int& ia, int64_t ib) noexcept;
void step_nequal(int64_t& iq, const mp_int& ia, const mp_int& ib) noexcept;
void step_nequal(mp_int& iq, int64_t ia, int64_t ib) noexcept;
void step_nequal(mp_int& iq, int64_t ia, const mp_int& ib) noexcept;
void step_nequal(mp_int& iq, const mp_int& ia, int64_t ib) noexcept;
void step_nequal(mp_int& iq, const mp_int& ia, const mp_int& ib) noexcept;
void step_reducing_nand(int64_t& iq, const mp_int& ia, size_t bit_width, bool is_signed, temps_span temps) noexcept;
void step_reducing_nand(mp_int& iq, int64_t ia, size_t bit_width, bool is_signed, temps_span temps) noexcept;
void step_reducing_nand(mp_int& iq, const mp_int& ia, size_t bit_width, bool is_signed, temps_span temps) noexcept;
void step_reducing_xor(int64_t& iq, const mp_int& ia, size_t bit_width) noexcept;
void step_reducing_xor(mp_int& iq, int64_t ia, size_t bit_width) noexcept;
void step_reducing_xor(mp_int& iq, const mp_int& ia, size_t bit_width) noexcept;
void step_reducing_nxor(int64_t& iq, const mp_int& ia, size_t bit_width) noexcept;
void step_reducing_nxor(mp_int& iq, int64_t ia, size_t bit_width) noexcept;
void step_reducing_nxor(mp_int& iq, const mp_int& ia, size_t bit_width) noexcept;
void step_bit_extract(int64_t& iq, const mp_int& ia, size_t width, bool signed_extend, int bit_pos, temps_span temps) noexcept;
void step_bit_extract(mp_int& iq, int64_t ia, size_t width, bool signed_extend, int bit_pos, temps_span temps) noexcept;
void step_bit_extract(mp_int& iq, const mp_int& ia, size_t width, bool signed_extend, int bit_pos, temps_span temps) noexcept;
void step_biased_round(int64_t& iq, const mp_int& ia, int bit, temps_span temps) noexcept;
void step_biased_round(mp_int& iq, int64_t ia, int bit, temps_span temps) noexcept;
void step_biased_round(mp_int& iq, const mp_int& ia, int bit, temps_span temps) noexcept;
void step_unbiased_round(int64_t& iq, const mp_int& ia, int bit, temps_span temps) noexcept;
void step_unbiased_round(mp_int& iq, int64_t ia, int bit, temps_span temps) noexcept;
void step_unbiased_round(mp_int& iq, const mp_int& ia, int bit, temps_span temps) noexcept;
void step_bit_reverse(int64_t& iq, const mp_int& ia, size_t bit_width) noexcept;
void step_bit_reverse(mp_int& iq, int64_t ia, size_t bit_width) noexcept;
void step_bit_reverse(mp_int& iq, const mp_int& ia, size_t bit_width) noexcept;
void step_sign_bit(int64_t& iq, const mp_int& ia, size_t bit_width) noexcept;
void step_sign_bit(mp_int& iq, int64_t ia, size_t bit_width) noexcept;
void step_sign_bit(mp_int& iq, const mp_int& ia, size_t bit_width) noexcept;
void step_nsign_bit(int64_t& iq, const mp_int& ia, size_t bit_width) noexcept;
void step_nsign_bit(mp_int& iq, int64_t ia, size_t bit_width) noexcept;
void step_nsign_bit(mp_int& iq, const mp_int& ia, size_t bit_width) noexcept;
void step_shift_right(int64_t& iq, const mp_int& ia, size_t amount, temps_span temps) noexcept;
void step_shift_right(mp_int& iq, int64_t ia, size_t amount, temps_span temps) noexcept;
void step_shift_right(mp_int& iq, const mp_int& ia, size_t amount, temps_span temps) noexcept;
void step_shift_left(int64_t& iq, const mp_int& ia, size_t amount, temps_span temps) noexcept;
void step_shift_left(mp_int& iq, int64_t ia, size_t amount, temps_span temps) noexcept;
void step_shift_left(mp_int& iq, const mp_int& ia, size_t amount, temps_span temps) noexcept;
void step_test_bit(int64_t& iq, const mp_int& ia, size_t bit_position) noexcept;
void step_test_bit(mp_int& iq, int64_t ia, size_t bit_position) noexcept;
void step_test_bit(mp_int& iq, const mp_int& ia, size_t bit_position) noexcept;
void step_set_bit(int64_t& iq, const mp_int& ia, size_t bit_position) noexcept;
void step_set_bit(mp_int& iq, int64_t ia, size_t bit_position) noexcept;
void step_set_bit(mp_int& iq, const mp_int& ia, size_t bit_position) noexcept;
void step_not(int64_t& iq, const mp_int& ia, int bit_width, temps_span temps) noexcept;
void step_not(mp_int& iq, int64_t ia, int bit_width, temps_span temps) noexcept;
void step_not(mp_int& iq, const mp_int& ia, int bit_width, temps_span temps) noexcept;
void step_not_signed(int64_t& iq, const mp_int& ia, int bit_width, temps_span temps) noexcept;
void step_not_signed(mp_int& iq, int64_t ia, int bit_width, temps_span temps) noexcept;
void step_not_signed(mp_int& iq, const mp_int& ia, int bit_width, temps_span temps) noexcept;
void step_sequencer(mp_int& state, mp_int& iq, const mp_int& ia, size_t offset, int64_t mod, int64_t cross, temps_span temps) noexcept;
void step_reduce(int64_t& iq, const mp_int& ia, size_t bit_width, temps_span temps) noexcept;
void step_reduce(mp_int& iq, int64_t ia, size_t bit_width, temps_span temps) noexcept;
void step_reduce(mp_int& iq, const mp_int& ia, size_t bit_width, temps_span temps) noexcept;
void step_counter(int64_t& counter, int64_t& iq, const mp_int& ia, size_t offset, int64_t inc, int64_t mod) noexcept;
void step_counter(int64_t& counter, mp_int& iq, int64_t ia, size_t offset, int64_t inc, int64_t mod) noexcept;
void step_counter(int64_t& counter, mp_int& iq, const mp_int& ia, size_t offset, int64_t inc, int64_t mod) noexcept;
void step_counter(mp_int& counter, int64_t& iq, int64_t ia, size_t offset, int64_t inc, int64_t mod) noexcept;
void step_counter(mp_int& counter, int64_t& iq, const mp_int& ia, size_t offset, int64_t inc, int64_t mod) noexcept;
void step_counter(mp_int& counter, mp_int& iq, int64_t ia, size_t offset, int64_t inc, int64_t mod) noexcept;
void step_counter(mp_int& counter, mp_int& iq, const mp_int& ia, size_t offset, int64_t inc, int64_t mod) noexcept;
void step_bit_combine(int64_t& iq, int64_t ia, const mp_int& ib, size_t num_bits, size_t index, size_t bit_pos, size_t next_bit_pos,
                      temps_span temps) noexcept;
void step_bit_combine(int64_t& iq, const mp_int& ia, int64_t ib, size_t num_bits, size_t index, size_t bit_pos, size_t next_bit_pos,
                      temps_span temps) noexcept;
void step_bit_combine(int64_t& iq, const mp_int& ia, const mp_int& ib, size_t num_bits, size_t index, size_t bit_pos,
                      size_t next_bit_pos, temps_span temps) noexcept;
void step_bit_combine(mp_int& iq, int64_t ia, int64_t ib, size_t num_bits, size_t index, size_t bit_pos, size_t next_bit_pos,
                      temps_span temps) noexcept;
void step_bit_combine(mp_int& iq, int64_t ia, const mp_int& ib, size_t num_bits, size_t index, size_t bit_pos, size_t next_bit_pos,
                      temps_span temps) noexcept;
void step_bit_combine(mp_int& iq, const mp_int& ia, int64_t ib, size_t num_bits, size_t index, size_t bit_pos, size_t next_bit_pos,
                      temps_span temps) noexcept;
void step_bit_combine(mp_int& iq, const mp_int& ia, const mp_int& ib, size_t num_bits, size_t index, size_t bit_pos,
                      size_t next_bit_pos, temps_span temps) noexcept;
bool get_lookup_value(const uint64_t* values, const mp_int_info* infos, uint64_t n, uint64_t key, mp_int& out) noexcept;
bool get_sparse_lookup_value(const mp_int* values, const unsigned char* exists, uint64_t n, uint64_t key, mp_int& out) noexcept;
bool get_sorted_lookup_value(const uint64_t* keys, const mp_int* values, uint64_t n, uint64_t key, mp_int& out) noexcept;
void step_lookup(const int64_t* values, uint64_t n, uint64_t offset, mp_int& iq, int64_t ia_in, temps_span temps);
void step_lookup(const uint64_t* values, const mp_int_info* infos, uint64_t n, uint64_t offset, int64_t& iq, int64_t ia_in,
                 temps_span temps);
void step_lookup(const uint64_t* values, const mp_int_info* infos, uint64_t n, uint64_t offset, mp_int& iq, int64_t ia_in,
                 temps_span temps);
void step_lookup_with_valid(const int64_t* values, uint64_t n, uint64_t offset, int64_t& iq, mp_int& ivalid, int64_t ia_in,
                            temps_span temps);
void step_lookup_with_valid(const int64_t* values, uint64_t n, uint64_t offset, mp_int& iq, int64_t& ivalid, int64_t ia_in,
                            temps_span temps);
void step_lookup_with_valid(const int64_t* values, uint64_t n, uint64_t offset, mp_int& iq, mp_int& ivalid, int64_t ia_in,
                            temps_span temps);
void step_lookup_with_valid(const uint64_t* values, const mp_int_info* infos, uint64_t n, uint64_t offset, int64_t& iq,
                            int64_t& ivalid, int64_t ia_in, temps_span temps);
void step_lookup_with_valid(const uint64_t* values, const mp_int_info* infos, uint64_t n, uint64_t offset, int64_t& iq, mp_int& ivalid,
                            int64_t ia_in, temps_span temps);
void step_lookup_with_valid(const uint64_t* values, const mp_int_info* infos, uint64_t n, uint64_t offset, mp_int& iq, int64_t& ivalid,
                            int64_t ia_in, temps_span temps);
void step_lookup_with_valid(const uint64_t* values, const mp_int_info* infos, uint64_t n, uint64_t offset, mp_int& iq, mp_int& ivalid,
                            int64_t ia_in, temps_span temps);
void step_decode(mp_int& iq0, const mp_int& ia, const mp_int& ib, int32_t low, int32_t high, int32_t decode,
                 temps_span temps) noexcept;
void step_decode(mp_int& iq0, mp_int& iq1, const mp_int& ia, const mp_int& ib, int32_t low, int32_t high, int32_t decode,
                 temps_span temps) noexcept;
void step_fp_acc(mp_int& iacc, int64_t control, int64_t& iq, int64_t ix) noexcept;
void step_fp_mult_acc(mp_int& iacc, int64_t control, int64_t& iq, int64_t ix, int64_t iy) noexcept;
void step_loadable_counter(int64_t& state_counter, int64_t& state_mod, int64_t& state_inc, mp_int& iq, int64_t ienable, int64_t iload,
                           int64_t load_count, int64_t load_mod, int64_t load_inc) noexcept;
void step_loadable_counter(mp_int& state_counter, mp_int& state_mod, mp_int& state_inc, int64_t& iq, const mp_int& ienable,
                           const mp_int& iload, const mp_int& load_count, const mp_int& load_mod, const mp_int& load_inc) noexcept;
void step_loadable_counter(mp_int& state_counter, mp_int& state_mod, mp_int& state_inc, mp_int& iq, const mp_int& ienable,
                           const mp_int& iload, const mp_int& load_count, const mp_int& load_mod, const mp_int& load_inc) noexcept;
void step_cma_add(const mp_int* const prod_arr, mp_int* const sums_arr, const cma_add_params& params, int64_t sub_ctrl,
                  int64_t neg_ctrl, mp_int& region_sum) noexcept;
void step_fifo(mp_int* store, const fifo_params& params, mp_int& data, int64_t write_en, int64_t read_en, int64_t flush) noexcept;

#endif // CSL_USE_GMP

/** Generated steps end */
} // namespace csl

/**
 * csl_def.h contains the implementation details for the functions
 * defined in this header.
 */
#include "csl_def.h"

#endif // CSL_CSL_H
