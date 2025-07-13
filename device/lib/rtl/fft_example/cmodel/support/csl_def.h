namespace csl
{

constexpr inline bool is_zero(int64_t& value) noexcept
{
    return (value == 0);
}

constexpr inline bool is_one(int64_t& value) noexcept
{
    return (value == 1);
}

constexpr inline void negate(int64_t& result, int64_t v) noexcept
{
    result = -v;
}

constexpr inline void complement(int64_t& result, int64_t v) noexcept
{
    result = ~v;
}

inline constexpr void safe_sub(int64_t& result, int64_t a, int64_t b, temps_span temps) noexcept
{
    (void)temps;
    // casting to u64 works around inlining issue in gcc8.x and gcc9.x
    result = (int64_t)(a - (uint64_t)b);
}

inline constexpr int64_t to_i64(int64_t v) noexcept
{
    return v;
}

inline constexpr uint64_t bit_mask_u64(size_t amount) noexcept
{
    return (amount < 64) ? (1ull << amount) - 1 : 0xFFFFFFFFFFFFFFFFull;
}

inline constexpr int64_t bit_mask_i64(size_t amount) noexcept
{
    return (amount < 64) ? (1ll << amount) - 1 : 0xFFFFFFFFFFFFFFFFll;
}

inline constexpr void mask_lower(int64_t& result, int64_t value, size_t width) noexcept
{
    result = value & bit_mask_i64(width);
}

template <typename TQ, typename TA, typename = std::enable_if_t<std::is_fundamental<TQ>::value>,
          typename = std::enable_if_t<std::is_fundamental<TA>::value>>
inline void mask_lower(TQ& result, TA ia, int64_t width, temps_span temps) noexcept
{
    int64_t i64 = 0;
    std::memcpy(&i64, &ia, sizeof(ia));
    int64_t tmp = i64 & bit_mask_i64(width);
    std::memcpy(&result, &tmp, sizeof(result));
}

inline constexpr bool test_bit(size_t bit_position, int64_t value) noexcept
{
    if (bit_position > 63)
    {
        // sign extend
        return value < 0;
    }
    else
    {
        uint64_t mask = (1ull << bit_position);
        uint64_t result = (value & mask);
        return result != 0;
    }
}

inline constexpr void mask(int64_t& result, int64_t v, size_t bit_width) noexcept
{
    result = v & bit_mask_i64(bit_width);
}

inline constexpr void set_upper(int64_t& value, size_t bit_width) noexcept
{
    value = ~value;
    mask_lower(value, value, bit_width);
    value = ~value;
}

inline constexpr void not_n(int64_t& result, int64_t value, int bit_width) noexcept
{
    result = value;
    for (int i = 0; i < bit_width; ++i)
    {
        result ^= 1ll << i;
    }
}

inline constexpr void boolean_complement(int64_t& result, int64_t v) noexcept
{
    result = ~v & 1;
}

inline constexpr void set_bit(int64_t& result, size_t position, bool clear) noexcept
{
    if (!clear)
    {
        result |= 1ull << position;
    }
    else
    {
        result &= ~(1ull << position);
    }
}

inline float flush_subnormals(float f) noexcept
{
    if (fabs(f) < 1.17549435082229e-38f)
    {
        // scaling by 0.0f preserves the sign bit
        return f * 0.0f;
    }
    return f;
}

inline float flush_signed_nan(float f) noexcept
{
    if (std::isnan(f))
    {
        uint32_t mask = ((uint32_t)1 << 31) - 1;
        uint32_t f_ui;
        std::memcpy(&f_ui, &f, sizeof(float));
        uint32_t x_post_mask = mask & f_ui;
        std::memcpy(&f, &x_post_mask, sizeof(float));
    }
    return f;
}

inline constexpr void set(int64_t& dst, int64_t src) noexcept
{
    dst = src;
}

inline constexpr uint64_t to_u64(int64_t value) noexcept
{
    return static_cast<uint64_t>(value);
}

inline constexpr void ld_exp(int64_t& result, int64_t value, int exp2) noexcept
{
    if (exp2 < 0)
    {
        exp2 = -exp2;
        if (exp2 > 63)
        {
            result = 0;
        }
        result = value >> exp2;
    }
    else
    {
        if (exp2 > 63)
        {
            // saturate if not 0
            result = (value == 0) ? 0 : ~0;
        }
        result = value << exp2;
    }
}

inline constexpr bool dual_mem_is_valid_address(int size, int addr) noexcept
{
    return ((addr >= 0) && (addr < static_cast<int>(size)));
}

inline constexpr int64_t abs(int64_t value) noexcept
{
    int64_t shr = value >> 63;
    return (value ^ shr) - shr;
}

inline constexpr int64_t pow2(int64_t value) noexcept
{
    return (((int64_t)1) << value);
}

#ifdef CSL_USE_GMP

inline int64_t to_i64(const mp_int& value) noexcept
{
    int64_t v = 0;
    size_t count = 1;
    mpz_export(&v, &count, 1, sizeof(int64_t), 0, 0, value.get());
    v *= mpz_sgn(value.get()); // signedness
    return v;
}

inline uint64_t to_u64(const mp_int& value) noexcept
{
    uint64_t v = 0;
    size_t count = 1;
    mpz_export(&v, &count, 1, sizeof(uint64_t), 0, 0, value.get());
    return v;
}

inline int32_t to_i32(const mp_int& value) noexcept
{
    int32_t v = 0;
    size_t count = 1;
    mpz_export(&v, &count, 1, sizeof(int32_t), 0, 0, value.get());
    v *= mpz_sgn(value.get()); // signedness
    return v;
}

inline uint32_t to_u32(const mp_int& value) noexcept
{
    uint32_t v = 0;
    size_t count = 1;
    mpz_export(&v, &count, 1, sizeof(uint32_t), 0, 0, value.get());
    return v;
}

inline void safe_sub(mp_int& result, int64_t a, int64_t b, temps_span temps) noexcept
{
    mp_int& bmp = temps.at(0);
    mp_int& amp = temps.at(1);
    bmp = b;
    amp = a;
    mpz_sub(result.get(), amp.get(), bmp.get());
}

inline void safe_sub(mp_int& result, const mp_int& a, int64_t b, temps_span temps) noexcept
{
    mp_int& bmp = temps.at(0);
    bmp = b;
    mpz_sub(result.get(), a.get(), bmp.get());
}

inline void safe_sub(mp_int& result, int64_t a, const mp_int& b, temps_span temps) noexcept
{
    mp_int& amp = temps.at(0);
    amp = a;
    mpz_sub(result.get(), amp.get(), b.get());
}

inline void safe_sub(mp_int& result, const mp_int& a, const mp_int& b, temps_span temps) noexcept
{
    (void)temps;
    mpz_sub(result.get(), a.get(), b.get());
}

inline void safe_sub(int64_t& result, const mp_int& a, int64_t b, temps_span temps) noexcept
{
    mp_int& bmp = temps.at(0);
    bmp = b;
    mpz_sub(bmp.get(), a.get(), bmp.get());
    set(result, bmp);
}

inline void safe_sub(int64_t& result, int64_t a, const mp_int& b, temps_span temps) noexcept
{
    mp_int& amp = temps.at(0);
    amp = a;
    mpz_sub(amp.get(), amp.get(), b.get());
    set(result, amp);
}

inline void safe_sub(int64_t& result, const mp_int& a, const mp_int& b, temps_span temps) noexcept
{
    (void)temps;
    mp_int& result_temp = temps.at(0);
    mpz_sub(result_temp.get(), a.get(), b.get());
    set(result, result_temp);
}

inline void dual_mem_get_word(int index, int size, int byte_width, int64_t* store, int64_t& word, const mp_int& addr, int word_size,
                              temps_span temps) noexcept
{
    dual_mem_get_word(index, size, byte_width, store, word, to_i64(addr), word_size, temps);
}

inline bool dual_mem_put_word(int index, int size, int byte_width, mp_int* store, const mp_int& word, const mp_int& addr,
                              int word_size, temps_span temps) noexcept
{
    return dual_mem_put_word(index, size, byte_width, store, word, to_i64(addr), word_size, temps);
}

inline bool dual_mem_put_word(int index, int size, int byte_width, int64_t* store, const int64_t& word, const mp_int& addr,
                              int word_size, temps_span temps) noexcept
{
    return dual_mem_put_word(index, size, byte_width, store, word, to_i64(addr), word_size, temps);
}

inline void dual_mem_get_word(int index, int size, int byte_width, mp_int* store, mp_int& word, const mp_int& addr, int word_size,
                              temps_span temps) noexcept
{
    dual_mem_get_word(index, size, byte_width, store, word, to_i64(addr), word_size, temps);
}

inline mp_int& mp_int::operator=(const char* s) noexcept
{
    size_t n = s ? std::strlen(s) : 0;
    int radix = 10;
    if ((n > 0) && (s[0] == '0'))
    {
        if ((n > 1) && ((s[1] == 'x') || (s[1] == 'X')))
        {
            radix = 16;
            s += 2;
            n -= 2;
        }
        else
        {
            radix = 8;
            n -= 1;
        }
    }
    if (n > 0)
    {
        if (mpz_set_str(m_value, s, radix) != 0)
        {
            error("Failed to set mp_int from string");
            mpz_set_ui(m_value, 0);
        }
    }
    else
    {
        mpz_set_ui(m_value, 0);
    }
    return *this;
}

inline mp_int& mp_int::operator=(uint64_t i) noexcept
{
    mpz_set_ui(m_value, i);
    return *this;
}

inline mp_int& mp_int::operator=(int64_t i) noexcept
{
    mpz_set_si(m_value, i);
    return *this;
}

inline mp_int& mp_int::operator=(uint32_t i) noexcept
{
    mpz_set_ui(m_value, i);
    return *this;
}

inline mp_int& mp_int::operator=(int32_t i) noexcept
{
    mpz_set_si(m_value, i);
    return *this;
}

inline mp_int& mp_int::operator=(const mpz_t val) noexcept
{
    mpz_set(this->m_value, val);
    return *this;
}

inline mp_int& mp_int::operator=(const mpf_t val) noexcept
{
    mpz_set_f(this->m_value, val);
    return *this;
}

inline mp_int& mp_int::operator=(const mp_int& other) noexcept
{
    if (this->m_value != other.m_value)
    {
        if (this->m_value[0]._mp_d)
        {
            mpz_set(this->m_value, other.get());
        }
        else
        {
            mpz_init_set(this->m_value, other.get());
        }
    }
    return *this;
}

inline mp_int& mp_int::operator=(mp_int&& other) noexcept
{
    if (this->m_value != other.m_value)
    {
        this->m_value->_mp_alloc = other.m_value->_mp_alloc;
        this->m_value->_mp_d = other.m_value->_mp_d;
        this->m_value->_mp_size = other.m_value->_mp_size;
        other.m_value->_mp_alloc = 0;
        other.m_value->_mp_d = nullptr;
        other.m_value->_mp_size = 0;
    }
    return *this;
}

inline mp_int::mp_int(uint64_t v)
{
    mpz_init_set_ui(this->m_value, v);
}

inline mp_int::mp_int(int64_t v)
{
    mpz_init_set_si(this->m_value, v);
}

inline mp_int::mp_int(uint32_t v)
{
    mpz_init_set_ui(this->m_value, v);
}

inline mp_int::mp_int(int32_t v)
{
    mpz_init_set_si(this->m_value, v);
}

inline mp_int::mp_int(const char* str)
{
    mpz_init(this->m_value);
    *this = str;
}

inline mp_int::mp_int()
{
    mpz_init(this->m_value);
}

inline mp_int::mp_int(mp_int&& other)
{
    this->m_value->_mp_alloc = other.m_value->_mp_alloc;
    this->m_value->_mp_d = other.m_value->_mp_d;
    this->m_value->_mp_size = other.m_value->_mp_size;
    other.m_value->_mp_alloc = 0;
    other.m_value->_mp_d = nullptr;
    other.m_value->_mp_size = 0;
}

inline mp_int::~mp_int() noexcept
{
    if (this->m_value[0]._mp_d)
    {
        mpz_clear(this->m_value);
    }
}

inline constexpr mpz_t& mp_int::get() noexcept
{
    return m_value;
}

inline constexpr const mpz_t& mp_int::get() const noexcept
{
    return m_value;
}

inline void mp_int::str(char* dst, size_t max_size) const noexcept
{
    // mpz_get_str assumes the storage location is large enough,
    // so only invoke it if it actually is.
    if (max_size >= (mpz_sizeinbase(m_value, 2) + 2))
    {
        mpz_get_str(dst, 10, m_value);
    }
    else
    {
        warning("mp_int::str(): dst array was too small for output");
    }
}

inline void mp_int::str_bin(char* dst, size_t max_size) const noexcept
{
    // mpz_get_str assumes the storage location is large enough,
    // so only invoke it if it actually is.
    if (max_size >= (mpz_sizeinbase(m_value, 2) + 2))
    {
        mpz_get_str(dst, 2, m_value);
    }
    else
    {
        warning("mp_int::str_bin(): dst array was too small for output");
    }
}

inline void mp_int::set_from_uint_array(const uint32_t* array, size_t n, size_t bit_width, temps_span temps) noexcept
{
    *this = 0;
    for (int32_t i = static_cast<int32_t>(n) - 1; i >= 0; --i)
    {
        *this += array[i];
        if (i != 0)
        {
            *this <<= 32;
        }
    }

    // flip if necessary
    if (bit_width > 1)
    {
        if (test_bit(bit_width - 1, *this))
        {
            mp_int& a2 = temps.at(0);
            mp_int& t = temps.at(1);
            mp_int& one = temps.at(2);
            one = int64_t(1);
            mask_lower(a2, *this, bit_width - 1);
            ld_exp(t, one, static_cast<int32_t>(bit_width - 1));
            mpz_sub(get(), a2.get(), t.get());
        }
    }
}

inline void mp_int::get_as_uint_array(uint32_t* arr, size_t n, temps_span temps) const noexcept
{
    mp_int& curr = temps.at(0);
    curr = *this;
    for (size_t i = 0; i < n; ++i)
    {
        uint64_t v = mpz_get_ui(curr.get());
        std::memcpy(&arr[i], &v, sizeof(uint32_t));
        curr >>= 32;
    }
}

inline bool is_zero(const mp_int& value) noexcept
{
    return mpz_cmp_si(value.get(), 0) == 0;
}

inline bool is_one(const mp_int& value) noexcept
{
    return mpz_cmp_si(value.get(), 1) == 0;
}

inline void negate(mp_int& result, const mp_int& v) noexcept
{
    result = v;
    mpz_neg(result.get(), result.get());
}

inline void complement(mp_int& result, const mp_int& v) noexcept
{
    result = v;
    mpz_com(result.get(), result.get());
}

inline bool operator==(const mp_int& a, const mp_int& b) noexcept
{
    return mpz_cmp(a.get(), b.get()) == 0;
}

inline bool operator==(const mp_int& a, int64_t b) noexcept
{
    return mpz_cmp_si(a.get(), b) == 0;
}

inline bool operator==(const mp_int& a, uint64_t b) noexcept
{
    return mpz_cmp_ui(a.get(), b) == 0;
}

inline bool operator==(const mp_int& a, int32_t b) noexcept
{
    return mpz_cmp_si(a.get(), b) == 0;
}

inline bool operator==(const mp_int& a, uint32_t b) noexcept
{
    return mpz_cmp_ui(a.get(), b) == 0;
}

inline bool operator==(int64_t a, const mp_int& b) noexcept
{
    return mpz_cmp_si(b.get(), a) == 0;
}

inline bool operator==(int32_t a, const mp_int& b) noexcept
{
    return mpz_cmp_si(b.get(), a) == 0;
}

inline bool operator==(uint64_t a, const mp_int& b) noexcept
{
    return mpz_cmp_ui(b.get(), a) == 0;
}

inline bool operator==(uint32_t a, const mp_int& b) noexcept
{
    return mpz_cmp_ui(b.get(), a) == 0;
}

inline mp_int& operator%=(mp_int& a, const mp_int& b) noexcept
{
    mpz_mod(a.get(), a.get(), b.get());
    return a;
}

inline mp_int& operator+=(mp_int& a, const mp_int& b) noexcept
{
    mpz_add(a.get(), a.get(), b.get());
    return a;
}

inline mp_int& operator-=(mp_int& a, const mp_int& b) noexcept
{
    mpz_sub(a.get(), a.get(), b.get());
    return a;
}

inline mp_int& operator*=(mp_int& a, const mp_int& b) noexcept
{
    mpz_mul(a.get(), a.get(), b.get());
    return a;
}

inline mp_int& operator|=(mp_int& a, const mp_int& b) noexcept
{
    mpz_ior(a.get(), a.get(), b.get());
    return a;
}

inline mp_int& operator&=(mp_int& a, const mp_int& b) noexcept
{
    mpz_and(a.get(), a.get(), b.get());
    return a;
}

inline mp_int& operator^=(mp_int& a, const mp_int& b) noexcept
{
    mpz_xor(a.get(), a.get(), b.get());
    return a;
}

inline mp_int& operator%=(mp_int& a, int64_t b) noexcept
{
    mpz_mod_ui(a.get(), a.get(), static_cast<uint64_t>(b));
    return a;
}

inline mp_int& operator+=(mp_int& a, int64_t b) noexcept
{
    if (b >= 0)
    {
        mpz_add_ui(a.get(), a.get(), static_cast<uint64_t>(b));
    }
    else
    {
        mpz_sub_ui(a.get(), a.get(), static_cast<uint64_t>(-b));
    }
    return a;
}

inline mp_int& operator-=(mp_int& a, int64_t b) noexcept
{
    if (b >= 0)
    {
        mpz_sub_ui(a.get(), a.get(), static_cast<uint64_t>(b));
    }
    else
    {
        mpz_add_ui(a.get(), a.get(), static_cast<uint64_t>(-b));
    }
    return a;
}

inline mp_int& operator*=(mp_int& a, int64_t b) noexcept
{
    mpz_mul_si(a.get(), a.get(), b);
    return a;
}

inline mp_int& operator<<=(mp_int& num, mp_bitcnt_t amount) noexcept
{
    if (amount > 0)
    {
        mpz_mul_2exp(num.get(), num.get(), amount);
    }
    return num;
}

inline mp_int& operator>>=(mp_int& num, mp_bitcnt_t amount) noexcept
{
    if (amount > 0)
    {
        mpz_fdiv_q_2exp(num.get(), num.get(), amount);
    }
    return num;
}

inline bool operator!=(const mp_int& a, const mp_int& b) noexcept
{
    return mpz_cmp(a.get(), b.get()) != 0;
}

inline bool operator!=(const mp_int& a, int64_t b) noexcept
{
    return mpz_cmp_si(a.get(), b) != 0;
}

inline bool operator!=(const mp_int& a, uint64_t b) noexcept
{
    return mpz_cmp_ui(a.get(), b) != 0;
}

inline bool operator!=(const mp_int& a, int32_t b) noexcept
{
    return mpz_cmp_si(a.get(), b) != 0;
}

inline bool operator!=(const mp_int& a, uint32_t b) noexcept
{
    return mpz_cmp_ui(a.get(), b) != 0;
}

inline bool operator!=(int64_t a, const mp_int& b) noexcept
{
    return mpz_cmp_si(b.get(), a) != 0;
}

inline bool operator!=(uint64_t a, const mp_int& b) noexcept
{
    return mpz_cmp_ui(b.get(), a) != 0;
}

inline bool operator!=(int32_t a, const mp_int& b) noexcept
{
    return mpz_cmp_si(b.get(), a) != 0;
}

inline bool operator!=(uint32_t a, const mp_int& b) noexcept
{
    return mpz_cmp_ui(b.get(), a) != 0;
}

inline bool operator<(const mp_int& a, const mp_int& b) noexcept
{
    return mpz_cmp(a.get(), b.get()) < 0;
}

inline bool operator>(const mp_int& a, const mp_int& b) noexcept
{
    return mpz_cmp(a.get(), b.get()) > 0;
}

inline bool operator<(const mp_int& a, int64_t b) noexcept
{
    return mpz_cmp_si(a.get(), b) < 0;
}

inline bool operator>(const mp_int& a, int64_t b) noexcept
{
    return mpz_cmp_si(a.get(), b) > 0;
}

inline bool operator<=(const mp_int& a, const mp_int& b) noexcept
{
    return !(mpz_cmp(a.get(), b.get()) > 0);
}

inline bool operator>=(const mp_int& a, const mp_int& b) noexcept
{
    return !(mpz_cmp(a.get(), b.get()) < 0);
}

inline bool operator<=(const mp_int& a, int64_t b) noexcept
{
    return !(mpz_cmp_si(a.get(), b) > 0);
}

inline bool operator>=(const mp_int& a, int64_t b) noexcept
{
    return !(mpz_cmp_si(a.get(), b) < 0);
}

inline void bit_mask_mp_int(mp_int& result, size_t amount) noexcept
{
    result = 1;
    result <<= amount;
    result -= 1;
}

inline void mask(mp_int& result, const mp_int& v, size_t bit_width) noexcept
{
    mpz_fdiv_r_2exp(result.get(), v.get(), bit_width);
}

inline void set(int64_t& dst, const mp_int& src) noexcept
{
    dst = to_i64(src);
}
inline void set(mp_int& dst, int64_t src) noexcept
{
    dst = src;
}
inline void set(mp_int& dst, const mp_int& src) noexcept
{
    dst = src;
}

inline void mask_lower(mp_int& result, const mp_int& value, size_t width) noexcept
{
    mpz_fdiv_r_2exp(result.get(), value.get(), width);
}

template <typename T, typename = std::enable_if_t<std::is_fundamental<T>::value>>
inline void mask_lower(T& result, const mp_int& ia, int64_t width, temps_span temps) noexcept
{
    mp_int& tmp = temps.at(0);
    tmp = ia;
    mask_lower(tmp, tmp, width);
    int64_t i64;
    set(i64, tmp);
    std::memcpy(&result, &i64, sizeof(result));
}

inline void mask_lower(mp_int& result, int64_t ia, int64_t width, temps_span temps) noexcept
{
    mp_int& tmp = temps.at(0);
    tmp = ia;
    mask_lower(tmp, tmp, width);
    set(result, tmp);
}

template <typename T, typename = std::enable_if_t<std::is_fundamental<T>::value>>
inline void mask_lower(mp_int& result, T ia, int64_t width, temps_span temps) noexcept
{
    int64_t i64 = 0;
    std::memcpy(&i64, &ia, sizeof(ia));
    mp_int& tmp = temps.at(0);
    tmp = i64;
    mask_lower(tmp, tmp, width);
    set(result, tmp);
}

inline void mask_lower(mp_int& result, const mp_int& ia, int64_t width, temps_span temps) noexcept
{
    mp_int& tmp = temps.at(0);
    tmp = ia;
    mask_lower(tmp, tmp, width);
    set(result, tmp);
}

inline bool test_bit(size_t bit_position, const mp_int& value) noexcept
{
    return mpz_tstbit(value.get(), bit_position) != 0;
}

inline void ld_exp(mp_int& result, const mp_int& value, int exp2) noexcept
{
    if (exp2 < 0)
    {
        mpz_fdiv_q_2exp(result.get(), value.get(), -exp2);
    }
    else
    {
        mpz_mul_2exp(result.get(), value.get(), exp2);
    }
}

inline void set_upper(mp_int& value, size_t bit_width) noexcept
{
    mpz_com(value.get(), value.get());
    mask_lower(value, value, bit_width);
    mpz_com(value.get(), value.get());
}

inline void not_n(mp_int& result, const mp_int& value, int bit_width) noexcept
{
    mpz_set(result.get(), value.get());
    for (int i = 0; i < bit_width; ++i)
    {
        mpz_combit(result.get(), i);
    }
}

inline void boolean_complement(mp_int& result, const mp_int& v) noexcept
{
    complement(result, v);
    result &= 1;
}

inline void set_bit(mp_int& result, size_t position, bool clear) noexcept
{
    if (!clear)
    {
        mpz_setbit(result.get(), position);
    }
    else
    {
        mpz_clrbit(result.get(), position);
    }
}

#endif

#ifdef CSL_USE_MPFR
inline void set(int64_t& dst, fp32 src) noexcept
{
    set(dst, static_cast<int64_t>(src.get_u32()));
}

inline void set(mp_float& dst, int64_t src) noexcept
{
    fp32 tmp(src);
    mpfr_set_d(dst.get(), static_cast<double>(tmp.get()), MPFR_RNDN);
}

inline void set(mp_float& dst, const mp_int& src) noexcept
{
    fp32 tmp(src);
    mpfr_set_d(dst.get(), static_cast<double>(tmp.get()), MPFR_RNDN);
}

inline mp_float& mp_float::operator=(const mp_float& x) noexcept
{
    mpfr_set(m_value, x.get(), GMP_RNDN);
    return *this;
}

inline mp_float::mp_float(mp_float_init_token)
{
    mpfr_init(m_value);
}

inline mp_float::mp_float(mp_prec_t precision)
{
    mpfr_init2(m_value, precision);
}

inline mp_float::~mp_float() noexcept
{
    mpfr_clear(m_value);
}

inline constexpr mpfr_t& mp_float::get() noexcept
{
    return m_value;
}
inline constexpr const mpfr_t& mp_float::get() const noexcept
{
    return m_value;
}

inline fp32::fp32(float f)
{
    set(f);
    set(flush_subnormals(get()));
    set(flush_signed_nan(get()));
}

inline fp32::fp32(const mp_int& in)
{
    set(to_u32(in));
    set(flush_subnormals(get()));
    set(flush_signed_nan(get()));
}

inline fp32::fp32(int64_t in)
{
    set((unsigned int)(in));
    set(flush_subnormals(get()));
    set(flush_signed_nan(get()));
}

inline constexpr float fp32::get() const
{
    return m_value;
}

inline uint32_t fp32::get_u32() const
{
    uint32_t v;
    std::memcpy(&v, &m_value, sizeof(uint32_t));
    return v;
}

inline constexpr void fp32::set(float f)
{
    m_value = f;
}

inline void fp32::set(uint32_t i)
{
    std::memcpy(&m_value, &i, sizeof(uint32_t));
}

#endif

#ifdef CSL_USE_MPFR
#ifdef CSL_USE_GMP
inline void float_pack_bits_default(mpfr_t& ref, int w_exp, int w_frac, mpz_t& z0, bool subnormals_to_zero) noexcept
{
    int bias = (1 << (w_exp - 1)) - 1;

    mpz_t z1;
    mpz_init(z1);

    bool nan = mpfr_nan_p(ref);
    bool inf = mpfr_inf_p(ref);
    int sgn = mpfr_sgn(ref);
    if (nan)
    {
        mpz_set_ui(z1, (1 << (w_exp + 1)) - 1);
        mpz_mul_2exp(z0, z1, w_frac);
        mpz_add_ui(z0, z0, 3);
    }
    else if (sgn == 0)
    {
        mpz_set_ui(z0, 0);
        if (mpfr_signbit(ref))
        {
            mpz_setbit(z0, w_exp + w_frac);
        }
    }
    else if (!inf)
    {
        mp_exp_t ex = mpfr_get_z_exp(z0, ref);
        ex = ex + bias + w_frac;
        if (ex <= 0)
        {
            if (subnormals_to_zero)
            {
                mpz_set_ui(z0, 0);
            }
            else
            {
                mpz_abs(z0, z0);

                mpfr_t y1;
                mpfr_init2(y1, 1 + w_frac);
                mpfr_set_z(y1, z0, GMP_RNDN);
                mpfr_div_2exp(y1, y1, 1 - ex, GMP_RNDN);
                mpfr_rint(y1, y1, GMP_RNDN);
                mpfr_get_z(z0, y1, GMP_RNDN);
                mpfr_clear(y1);
            }
            if (sgn < 0)
            {
                mpz_setbit(z0, w_exp + w_frac);
            }
        }
        else if (ex < ((1 << w_exp) - 1))
        {
            if (sgn < 0)
            {
                mpz_neg(z0, z0);
                mpz_setbit(z0, w_exp + w_frac);
            }
            mpz_clrbit(z0, w_frac);
            mpz_set_ui(z1, ex);
            mpz_mul_2exp(z1, z1, w_frac);
            mpz_ior(z0, z0, z1);
        }
        else
        {
            inf = true;
        }
    }

    if (inf)
    {
        mpz_set_ui(z1, (1 << w_exp) - 1);
        mpz_mul_2exp(z0, z1, w_frac);
        if (sgn < 0)
        {
            mpz_setbit(z0, w_exp + w_frac);
        }
    }

    mpz_clear(z1);
}
#endif

inline void flush_bad_values(int exponent_width, mpfr_t& o) noexcept
{
    if (mpfr_regular_p(o))
    {
        const int64_t exponent_max = bit_mask_i64(exponent_width);
        const int64_t exponent_bias = bit_mask_i64(exponent_width - 1);
        const int64_t exponent_adjust = exponent_bias - 1;
        const int64_t adjusted_exponent = mpfr_get_exp(o) + exponent_adjust;

        if (exponent_max <= adjusted_exponent)
        {
            const int mpfr_sign_v = mpfr_sgn(o);
            mpfr_set_inf(o, mpfr_sign_v);
        }
        else if (adjusted_exponent <= 0)
        {
            const int mpfr_sign_v = mpfr_sgn(o);
            mpfr_set_zero(o, mpfr_sign_v);
        }
    }
}

inline void mult_fp16_extend(mpfr_t& o, mpfr_t& a, mpfr_t& b) noexcept
{
    // 2^-14
    const double smallest_normal = 6.103515625e-5;
    const double in0 = mpfr_get_d(a, MPFR_RNDN);
    const double in1 = mpfr_get_d(b, MPFR_RNDN);
    const bool in0_denorm = (fabs(in0) < smallest_normal);
    const bool in1_denorm = (fabs(in1) < smallest_normal);
    const bool extra_test = (in0_denorm || in1_denorm) && !(mpfr_zero_p(a) || mpfr_zero_p(b));

    if (extra_test)
    {
        // 10 fraction bits + 1 implicit (hidden) bit
        const int out_prec = 11;

        mpfr_t in0_man, in1_man, man_prod, cst_rnd_n;
        mpfr_inits2(out_prec, in0_man, in1_man, cst_rnd_n, (mpfr_ptr)0);
        mpfr_init2(man_prod, 2 * out_prec);

        const int in0_exp = mpfr_get_exp(a);
        const int in1_exp = mpfr_get_exp(b);

        mpfr_set(in0_man, a, MPFR_RNDN);
        mpfr_set(in1_man, b, MPFR_RNDN);
        // For aligning denormalized operands. Smallest normal on half is 1.0 2^-14 but in MPFR this is 0.5 2^-13.
        mpfr_set_exp(in0_man, 1 + (in0_denorm ? (in0_exp + 13) : 0));
        mpfr_set_exp(in1_man, 1 + (in1_denorm ? (in1_exp + 13) : 0));

        mpfr_mul(man_prod, in0_man, in1_man, MPFR_RNDN);
        mpfr_abs(man_prod, man_prod, MPFR_RNDN);

        mpfr_set_si(cst_rnd_n, 1, MPFR_RNDN);
        // -11 is mantissa_width (10) + 1 bit
        mpfr_mul_2si(cst_rnd_n, cst_rnd_n, -11, MPFR_RNDN);
        mpfr_si_sub(cst_rnd_n, 1, cst_rnd_n, MPFR_RNDN);

        if (mpfr_cmp(man_prod, cst_rnd_n) < 0)
        {
            mpfr_mul(o, a, b, MPFR_RNDZ);
        }
        else
        {
            if (mpfr_cmp_si(man_prod, 1) >= 1)
            {
                mpfr_mul(o, a, b, MPFR_RNDN);
            }
            else
            {
                mpfr_mul(o, a, b, MPFR_RNDZ);
                if (mpfr_cmp_si(o, 0) > 0)
                {
                    mpfr_nextabove(o);
                }
                else
                {
                    mpfr_nextbelow(o);
                }
            }
        }
        mpfr_clears(in0_man, in1_man, cst_rnd_n, (mpfr_ptr)0);
    }
    else
    {
        mpfr_mul(o, a, b, MPFR_RNDN);
    }
}

inline int64_t extract_field(int64_t src, int offset, int width) noexcept
{
    return ((src >> offset) & bit_mask_i64(width));
}

inline void transfer_fp(int exponent_width, int mantissa_width, mp_float& dst, const mp_int& src) noexcept
{
    const int64_t exponent_max = bit_mask_i64(exponent_width);
    const int64_t exponent_bias = bit_mask_i64(exponent_width - 1);

    const int64_t src_int = static_cast<int64_t>(mpz_get_si(src.get()));
    const int64_t src_sign_field = extract_field(src_int, exponent_width + mantissa_width, 1);
    const int64_t src_exponent_field = extract_field(src_int, mantissa_width, exponent_width);
    const int64_t src_mantissa_field = extract_field(src_int, 0, mantissa_width);

    const int64_t mpfr_sign_v = (src_sign_field == 0) ? 1 : -1;

    if (src_exponent_field == exponent_max)
    {
        if (src_mantissa_field == 0)
        {
            mpfr_set_inf(dst.get(), static_cast<int>(mpfr_sign_v));
        }
        else
        {
            mpfr_set_nan(dst.get());
        }
    }
    else if (src_exponent_field == 0)
    {
        if (src_mantissa_field == 0)
        {
            mpfr_set_zero(dst.get(), static_cast<int>(mpfr_sign_v));
        }
        else
        {
            const int64_t src_signed_mantissa = mpfr_sign_v * src_mantissa_field;
            const int64_t exponent_adjust = (exponent_bias - 1) + mantissa_width;
            mpfr_set_si_2exp(dst.get(), static_cast<long>(src_signed_mantissa),
                             static_cast<mpfr_exp_t>(src_exponent_field - exponent_adjust), MPFR_RNDN);
        }
    }
    else
    {
        const int64_t src_full_mantissa = src_mantissa_field + pow2(mantissa_width);
        const int64_t src_signed_mantissa = mpfr_sign_v * src_full_mantissa;
        const int64_t exponent_adjust = exponent_bias + mantissa_width;
        mpfr_set_si_2exp(dst.get(), static_cast<long>(src_signed_mantissa),
                         static_cast<mpfr_exp_t>(src_exponent_field - exponent_adjust), MPFR_RNDN);
    }
}
#endif

#ifdef CSL_USE_MPFR
inline fp32 fp_mul_impl(const fp32& x, const fp32& y) noexcept
{
    // single precision, 24-bit mantissa
    mp_float mp_x(24);
    mp_float mp_y(24);

    // use MPFR for single precision multiply to handle subtlety when rounding close to a subnormal
    mpfr_set_d(mp_x.get(), static_cast<double>(x.get()), GMP_RNDN);
    mpfr_set_d(mp_y.get(), static_cast<double>(y.get()), GMP_RNDN);
    mpfr_mul(mp_x.get(), mp_x.get(), mp_y.get(), GMP_RNDN);

    // single precision exponent
    const int expWidth = 8;
    const int eMinNormal = 2 - (1 << (expWidth - 1)) + 1;
    const int eMaxNormal = (1 << (expWidth - 1)) - 1 + 1;
    // non-zero for negative
    int sign_x = mpfr_signbit(mp_x.get());
    int current_exp = mpfr_get_exp(mp_x.get());
    if (current_exp < eMinNormal)
    {
        // flush subnormals to zero
        mpfr_set_zero(mp_x.get(), (sign_x == 0) ? 1 : -1);
    }
    else if (current_exp > eMaxNormal)
    {
        // overflow to infinity
        mpfr_set_inf(mp_x.get(), (sign_x == 0) ? 1 : -1);
    }

    float q = static_cast<float>(mpfr_get_d(mp_x.get(), GMP_RNDN));
    q = flush_signed_nan(q);
    return fp32(q);
}
#endif

#ifdef CSL_USE_GMP
inline void dual_mem_get_word(int index, int size, int byte_width, mp_int* store, mp_int& word, int64_t uaddr, int word_size,
                              temps_span temps) noexcept
{
    int addr = static_cast<int>(uaddr);
    const int mem_base = index * size;
    const int byte_addr = addr * word_size;

    if (!dual_mem_is_valid_address(size, byte_addr))
    {
        mpz_set_si(word.get(), 0xcdcdcdcd);
    }
    else if (word_size == 1)
    {
        word = store[mem_base + byte_addr];
    }
    else
    {
        // Little-endian i.e. LS byte (processed last here)
        // is stored at the lowest address address
        mpz_set_ui(word.get(), 0);
        mp_int& temp = temps.at(0);
        for (int i = word_size; --i;)
        {
            const mp_int& byte = store[mem_base + byte_addr + i];
            mask_lower(temp, byte, byte_width);
            ld_exp(word, word, byte_width);
            mpz_ior(word.get(), word.get(), temp.get());
        }
    }
}
#endif

inline void dual_mem_get_word(int index, int size, int byte_width, int64_t* store, int64_t& word, int64_t uaddr, int word_size,
                              temps_span temps) noexcept
{
    int addr = static_cast<int>(uaddr);
    const int mem_base = index * size;
    const int byte_addr = addr * word_size;

    if (!dual_mem_is_valid_address(size, byte_addr))
    {
        word = 0xcdcdcdcd;
    }
    else if (word_size == 1)
    {
        word = store[mem_base + byte_addr];
    }
    else
    {
        // Little-endian i.e. LS byte (processed last here)
        // is stored at the lowest address address
        word = 0;
        int64_t temp;
        for (int i = word_size; --i;)
        {
            const int64_t& byte = store[mem_base + byte_addr + i];
            mask_lower(temp, byte, byte_width);
            ld_exp(word, word, byte_width);
            word |= temp;
        }
    }
}

#ifdef CSL_USE_GMP

inline bool dual_mem_put_word(int index, int size, int byte_width, mp_int* store, const mp_int& word, int64_t uaddr, int word_size,
                              temps_span temps) noexcept
{
    int addr = static_cast<int>(uaddr);
    const int mem_base = index * size;
    const int byte_addr = addr * word_size;

    if (!dual_mem_is_valid_address(size, byte_addr))
    {
        return false;
    }
    else if (word_size == 1)
    {
        store[mem_base + byte_addr] = word;
        return true;
    }
    else
    {
        // Little-endian i.e. LS byte (processed first here) is stored at the lowest address
        mp_int& tmp = temps.at(0);
        tmp = word;
        for (int i = 0; i < word_size; ++i)
        {
            mp_int& byte = store[mem_base + byte_addr + i];
            mask_lower(byte, tmp, byte_width);
            ld_exp(tmp, tmp, byte_width);
        }
        return true;
    }
}

#endif

inline bool dual_mem_put_word(int index, int size, int byte_width, int64_t* store, const int64_t& word, int64_t uaddr, int word_size,
                              temps_span temps) noexcept
{
    int addr = static_cast<int>(uaddr);
    const int mem_base = index * size;
    const int byte_addr = addr * word_size;

    if (!dual_mem_is_valid_address(size, byte_addr))
    {
        return false;
    }
    else if (word_size == 1)
    {
        store[mem_base + byte_addr] = word;
        return true;
    }
    else
    {
        // Little-endian i.e. LS byte (processed first here)
        // is stored at the lowest address
        int64_t tmp = word;
        for (int i = 0; i < word_size; ++i)
        {
            int64_t& byte = store[mem_base + byte_addr + i];
            mask_lower(byte, tmp, byte_width);
            ld_exp(tmp, tmp, byte_width);
        }
        return true;
    }
}

/** Generated steps begin */

#ifdef CSL_USE_MPFR

inline void step_fp_add(int64_t& iq, int64_t ia, int64_t ib) noexcept
{
    set(iq, fp32(fp32(ia).get() + fp32(ib).get()).get_u32());
}

#endif // CSL_USE_MPFR
// --------------------------------

#ifdef CSL_USE_MPFR

inline void step_fp_sub(int64_t& iq, int64_t ia, int64_t ib) noexcept
{
    set(iq, fp32(fp32(ia).get() - fp32(ib).get()).get_u32());
}

#endif // CSL_USE_MPFR
// --------------------------------

#ifdef CSL_USE_MPFR

inline void step_fp_mul(int64_t& iq, int64_t ia, int64_t ib) noexcept
{
    set(iq, fp_mul_impl(fp32(ia), fp32(ib)).get_u32());
}

#endif // CSL_USE_MPFR
// --------------------------------

inline void step_add(int64_t& iq, int64_t ia, int64_t ib, temps_span temps) noexcept
{
    int64_t ta;
    ta = ia;
    ta += ib;
    set(iq, ta);
}

// --------------------------------

inline void step_mul(int64_t& iq, int64_t ia, int64_t ib, temps_span temps) noexcept
{
    int64_t ta;
    ta = ia;
    ta *= ib;
    set(iq, ta);
}

// --------------------------------

inline void step_sub(int64_t& iq, int64_t ia, int64_t ib, temps_span temps) noexcept
{
    int64_t ta;
    ta = ia;
    ta -= ib;
    set(iq, ta);
}

// --------------------------------

inline void step_addsub(int64_t ctrl, int64_t& iq, int64_t ia, int64_t ib, temps_span temps) noexcept
{
    if (!is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

// --------------------------------

inline void step_subadd(int64_t ctrl, int64_t& iq, int64_t ia, int64_t ib, temps_span temps) noexcept
{
    if (is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

// --------------------------------

inline void step_const(int64_t& iq, int64_t ia) noexcept
{
    set(iq, ia);
}

// --------------------------------

inline void step_and(int64_t& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    int64_t ta;
    ta = ia;
    ta &= ib;
    set(iq, ta);
}

// --------------------------------

inline void step_or(int64_t& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    int64_t ta;
    ta = ia;
    ta |= ib;
    set(iq, ta);
}

// --------------------------------

inline void step_xor(int64_t& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    int64_t ta;
    ta = ia;
    ta ^= ib;
    set(iq, ta);
}

// --------------------------------

inline void step_nand(int64_t& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    step_and(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

// --------------------------------

inline void step_nor(int64_t& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    step_or(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

// --------------------------------

inline void step_nxor(int64_t& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    step_xor(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

// --------------------------------

inline void step_reducing_or(int64_t& iq, int64_t ia) noexcept
{
    set(iq, (ia == 0) ? 0 : 1);
}

// --------------------------------

inline void step_reducing_nor(int64_t& iq, int64_t ia) noexcept
{
    step_reducing_or(iq, ia);
    boolean_complement(iq, iq);
}

// --------------------------------

inline void step_reducing_and(int64_t& iq, int64_t ia, size_t bit_width, bool is_signed, temps_span temps) noexcept
{
    (void)is_signed;
    (void)temps;
    uint64_t uia = *reinterpret_cast<const uint64_t*>(&ia);
    uint64_t mask = bit_mask_u64(bit_width);
    set(iq, (mask & uia) == mask);
}

// --------------------------------

// --------------------------------

inline void step_ld_exp(int64_t& iq, int64_t ia, int64_t ib, bool reverse, temps_span temps) noexcept
{
    int exp = static_cast<int>(ib);
    int64_t tmp;
    ld_exp(tmp, ia, reverse ? -exp : exp);
    set(iq, tmp);
}

// --------------------------------

// --------------------------------

inline void step_equal(int64_t& iq, int64_t ia, int64_t ib) noexcept
{
    set(iq, (ia == ib) ? 1 : 0);
}

// --------------------------------

inline void step_nequal(int64_t& iq, int64_t ia, int64_t ib) noexcept
{
    set(iq, (ia != ib) ? 1 : 0);
}

// --------------------------------

inline void step_reducing_nand(int64_t& iq, int64_t ia, size_t bit_width, bool is_signed, temps_span temps) noexcept
{
    step_reducing_and(iq, ia, bit_width, is_signed, temps);
    boolean_complement(iq, iq);
}

// --------------------------------

inline void step_reducing_xor(int64_t& iq, int64_t ia, size_t bit_width) noexcept
{
    bool result = test_bit(0, ia);
    for (unsigned int i = 1; i < bit_width; ++i)
    {
        result = result ^ test_bit(i, ia);
    }
    set(iq, result ? 1 : 0);
}

// --------------------------------

inline void step_reducing_nxor(int64_t& iq, int64_t ia, size_t bit_width) noexcept
{
    step_reducing_xor(iq, ia, bit_width);
    boolean_complement(iq, iq);
}

// --------------------------------

CSL_FORCE_INLINE void step_bit_extract(int64_t& iq, int64_t ia, size_t width, bool signed_extend, int bit_pos,
                                       temps_span temps) noexcept
{
    int64_t result;
    ld_exp(result, ia, -bit_pos);
    if (signed_extend && test_bit(width - 1, result))
    {
        set_upper(result, width - 1);
    }
    else
    {
        mask_lower(result, result, width);
    }
    set(iq, result);
}

// --------------------------------

inline void step_biased_round(int64_t& iq, int64_t ia, int bit, temps_span temps) noexcept
{
    static constexpr int64_t one{1};
    int64_t half;
    ld_exp(half, one, bit - 1);
    half += ia;
    ld_exp(half, half, -bit);
    set(iq, half);
}

// --------------------------------

inline void step_unbiased_round(int64_t& iq, int64_t ia, int bit, temps_span temps) noexcept
{
    int64_t half;
    int64_t frac_mask;
    static constexpr int64_t one{1};
    ld_exp(half, one, bit - 1);

    mask_lower(frac_mask, ia, bit);
    if (frac_mask == half)
    {
        int64_t& q = half;
        ld_exp(q, ia, -bit);
        if (test_bit(0, q))
        {
            q += 1;
        }
        set(iq, q);
    }
    else
    {
        step_biased_round(iq, ia, bit, temps.next(3));
    }
}

// --------------------------------

inline void step_bit_reverse(int64_t& iq, int64_t ia, size_t bit_width) noexcept
{
    set(iq, 0);
    for (size_t i = 0; i < bit_width; ++i)
    {
        if (test_bit(i, ia))
        {
            set_bit(iq, bit_width - 1 - i, false);
        }
    }
}

// --------------------------------

inline void step_sign_bit(int64_t& iq, int64_t ia, size_t bit_width) noexcept
{
    set(iq, (test_bit(bit_width - 1, ia) == true) ? 1 : 0);
}

// --------------------------------

inline void step_nsign_bit(int64_t& iq, int64_t ia, size_t bit_width) noexcept
{
    set(iq, (test_bit(bit_width - 1, ia) == false) ? 1 : 0);
}

// --------------------------------

inline void step_shift_right(int64_t& iq, int64_t ia, size_t amount, temps_span temps) noexcept
{
    int64_t ta;
    ta = ia;
    ta >>= amount;
    set(iq, ta);
}

// --------------------------------

inline void step_shift_left(int64_t& iq, int64_t ia, size_t amount, temps_span temps) noexcept
{
    int64_t ta;
    ta = ia;
    ta <<= amount;
    set(iq, ta);
}

// --------------------------------

inline void step_test_bit(int64_t& iq, int64_t ia, size_t bit_position) noexcept
{
    set(iq, test_bit(bit_position, ia) ? 1 : 0);
}

// --------------------------------

inline void step_set_bit(int64_t& iq, int64_t ia, size_t bit_position) noexcept
{
    set_bit(iq, bit_position, ia == 0);
}

// --------------------------------

inline void step_not(int64_t& iq, int64_t ia, int bit_width, temps_span temps) noexcept
{
    int64_t result;
    not_n(result, ia, bit_width);
    set(iq, result);
}

// --------------------------------

inline void step_not_signed(int64_t& iq, int64_t ia, int bit_width, temps_span temps) noexcept
{
    int64_t result;
    not_n(result, ia, bit_width);
    set(iq, result);
    if (test_bit(static_cast<size_t>(bit_width) - 1, iq))
    {
        set_upper(iq, static_cast<size_t>(bit_width));
    }
    else
    {
        mask_lower(iq, iq, static_cast<size_t>(bit_width));
    }
}

// --------------------------------

inline void step_sequencer(int64_t& state, int64_t& iq, int64_t ia, size_t offset, int64_t mod, int64_t cross,
                           temps_span temps) noexcept
{
    (void)offset;
    const bool enable = ia != 0;
    if (enable)
    {
        state += 1;
        state %= mod;
    }
    int64_t tmp;
    tmp = cross;
    iq = (state >= tmp) ? 1 : 0;
}

// --------------------------------

CSL_FORCE_INLINE void step_reduce(int64_t& iq, int64_t ia, size_t bit_width, temps_span temps) noexcept
{
    // test sign bit
    int64_t a;
    if (test_bit(bit_width - 1, ia) != 0)
    {
        mask_lower(a, ia, bit_width - 1);
        static constexpr int64_t one{1};
        int64_t b;
        ld_exp(b, one, (int)(bit_width - 1));
        safe_sub(iq, a, b, temps.next(3));
    }
    else
    {
        mask_lower(a, ia, bit_width - 1);
        set(iq, a);
    }
}

// --------------------------------

inline void step_counter(int64_t& counter, int64_t& iq, int64_t ia, size_t offset, int64_t inc, int64_t mod) noexcept
{
    const bool enable = (ia != 0);
    if (enable)
    {
        counter += inc;
        counter %= mod;
    }
    set(iq, counter);
    iq += offset;
}

// --------------------------------

inline void step_bit_combine(int64_t& iq, int64_t ia, int64_t ib, size_t num_bits, size_t index, size_t bit_pos, size_t next_bit_pos,
                             temps_span temps) noexcept
{
    if (index == 0)
    {
        int64_t ord;
        ord = ia;
        ord |= ib;
        set(iq, ord);
    }
    else
    {
        int64_t shifted_b;
        int64_t ibsz;
        ibsz = ib;
        int64_t result;
        ld_exp(shifted_b, ibsz, static_cast<int>(bit_pos));
        if (index < num_bits - 1)
        {
            // avoid masking out sign of last element
            mask_lower(shifted_b, shifted_b, next_bit_pos);
        }
        if (index == 1)
        {
            // mask 1st element during this iteration, as index = 0 can sometimes be skipped
            set(result, ia);
            mask_lower(result, result, bit_pos);
        }
        else
        {
            result = ia;
        }
        result |= shifted_b;
        set(iq, result);
    }
}

// --------------------------------

inline bool get_lookup_value(const int64_t* values, uint64_t n, uint64_t key, int64_t& out) noexcept
{
    if (key < n)
    {
        out = values[key];
        return true;
    }
    out = 0;
    return false;
}

// --------------------------------

// --------------------------------

inline bool get_sparse_lookup_value(const int64_t* values, const unsigned char* exists, uint64_t n, uint64_t key,
                                    int64_t& out) noexcept
{
    if (exists[key] && (key > 0) && (key < n))
    {
        out = values[key];
        return true;
    }
    out = 0;
    return false;
}

// --------------------------------

inline bool get_sorted_lookup_value(const uint64_t* keys, const int64_t* values, uint64_t n, uint64_t key, int64_t& out) noexcept
{
    uint64_t l = 0;
    uint64_t r = n - 1;
    while (l <= r)
    {
        uint64_t curr = l + (r - l) / 2;
        if (keys[curr] == key)
        {
            out = values[curr];
            return true;
        }
        if (keys[curr] < key)
        {
            l = curr + 1;
        }
        else
        {
            r = curr - 1;
        }
    }
    set(out, 0);
    return false;
}

// --------------------------------

inline void step_lookup(const int64_t* values, uint64_t n, uint64_t offset, int64_t& iq, int64_t ia_in, temps_span temps)
{
    uint64_t ia_u64;
    memcpy(&ia_u64, &ia_in, sizeof(uint64_t));
    uint64_t ia = ia_u64 - offset;
    int64_t result;
    get_lookup_value(values, n, ia, result);
    set(iq, result);
}

// --------------------------------

// --------------------------------

inline void step_lookup_with_valid(const int64_t* values, uint64_t n, uint64_t offset, int64_t& iq, int64_t& ivalid, int64_t ia_in,
                                   temps_span temps)
{
    uint64_t ia_u64;
    memcpy(&ia_u64, &ia_in, sizeof(uint64_t));
    uint64_t ia = ia_u64 - offset;
    int64_t result;
    bool valid = get_lookup_value(values, n, ia, result);
    set(iq, result);
    set(ivalid, valid ? 1 : 0);
}

// --------------------------------

// --------------------------------

inline void step_decode(int64_t& iq0, int64_t ia, int64_t ib, int32_t low, int32_t high, int32_t decode, temps_span temps) noexcept
{
    int64_t tmp_a;
    tmp_a = ia;
    ld_exp(tmp_a, tmp_a, -low);
    mask_lower(tmp_a, tmp_a, static_cast<int64_t>(high) - low + 1);
    bool hit = (tmp_a == decode);
    if (hit)
    {
        set(iq0, ib);
    }
    else
    {
        set(iq0, int64_t(0));
    }
}

// --------------------------------

inline void step_decode(int64_t& iq0, int64_t& iq1, int64_t ia, int64_t ib, int32_t low, int32_t high, int32_t decode,
                        temps_span temps) noexcept
{
    int64_t tmp_a;
    tmp_a = ia;
    ld_exp(tmp_a, tmp_a, -low);
    mask_lower(tmp_a, tmp_a, static_cast<int64_t>(high) - low + 1);
    bool hit = (tmp_a == decode);
    if (hit)
    {
        set(iq0, ib);
        set(iq1, int64_t(1));
    }
    else
    {
        set(iq0, int64_t(0));
        set(iq1, int64_t(0));
    }
}

// --------------------------------

#ifdef CSL_USE_MPFR

inline void step_fp_acc(int64_t& iacc, int64_t control, int64_t& iq, int64_t ix) noexcept
{
    fp32 x(ix);

    mp_float mp_acc(24);
    if (control == 0)
    {
        // acc = x
        mpfr_set_d(mp_acc.get(), static_cast<double>(x.get()), GMP_RNDN);
    }
    else
    {
        mp_float mp_x(24);
        mpfr_set_d(mp_x.get(), static_cast<double>(x.get()), GMP_RNDN);

        fp32 acc(iacc);
        mpfr_set_d(mp_acc.get(), static_cast<double>(acc.get()), GMP_RNDN);

        // acc = acc + x
        mpfr_add(mp_acc.get(), mp_acc.get(), mp_x.get(), GMP_RNDN);
    }
    float v = static_cast<float>(mpfr_get_d(mp_acc.get(), GMP_RNDN));
    int v_int;
    memcpy(&v_int, &v, sizeof(v));
    iacc = v_int;

    // set output
    iq = to_i64(iacc);
}

#endif // CSL_USE_MPFR
// --------------------------------

#ifdef CSL_USE_MPFR

inline void step_fp_mult_acc(int64_t& iacc, int64_t control, int64_t& iq, int64_t ix, int64_t iy) noexcept
{
    fp32 x(ix);
    fp32 y(iy);

    // x = x*y
    mp_float mp_x(24);
    mp_float mp_y(24);
    mp_float mp_acc(24);
    mpfr_set_d(mp_x.get(), static_cast<double>(x.get()), GMP_RNDN);
    mpfr_set_d(mp_y.get(), static_cast<double>(y.get()), GMP_RNDN);
    mpfr_mul(mp_x.get(), mp_x.get(), mp_y.get(), GMP_RNDN);

    if (control == 0)
    {
        // acc = (x*y)
        mpfr_set(mp_acc.get(), mp_x.get(), GMP_RNDN);
    }
    else
    {
        fp32 acc(iacc);
        mpfr_set_d(mp_acc.get(), static_cast<double>(acc.get()), GMP_RNDN);
        // acc = acc + (x*y)
        mpfr_add(mp_acc.get(), mp_acc.get(), mp_x.get(), GMP_RNDN);
    }
    float v = static_cast<float>(mpfr_get_d(mp_acc.get(), GMP_RNDN));
    int v_int;
    memcpy(&v_int, &v, sizeof(v));
    iacc = v_int;

    // set output
    iq = to_i64(iacc);
}

#endif // CSL_USE_MPFR
// --------------------------------

inline void step_loadable_counter(int64_t& state_counter, int64_t& state_mod, int64_t& state_inc, int64_t& iq, int64_t ienable,
                                  int64_t iload, int64_t load_count, int64_t load_mod, int64_t load_inc) noexcept
{
    const bool enable = ienable != 0;
    const bool load = iload != 0;

    if (load)
    {
        state_counter = load_count;
        state_mod = load_mod;
        state_inc = load_inc;
        if (state_inc < 0)
        {
            state_inc += load_mod;
        }
    }
    else if (enable)
    {
        // Modulo zero is undefined - just like divide by zero. The hardware will count as
        // if there is no modulo however, and we will do the same. Currently we don't issue
        // a warning.
        state_counter += state_inc;
        if (state_mod != 0)
        {
            state_counter %= state_mod;
        }
    }
    set(iq, state_counter);
}

// --------------------------------

inline void step_enable_generator(const enable_gen_params& params, int64_t valid, int64_t enable_in, int64_t& enable_out,
                                  int64_t& count, int64_t& en_count, int64_t& zero_force_count, int64_t& enable,
                                  int64_t& enable_zero_forcing_sequencer) noexcept
{
    bool last_cycle = false;
    bool force_disable = false;

    // count enabled cycles so we know whether we're on the last enabled cycle
    if (enable != 0)
    {
        en_count++;
        if (en_count == params.compute_cycle_length)
        {
            en_count = 0;
            last_cycle = true;
        }
    }

    // the accumulator
    if (valid != 0)
    {
        count += params.valid_inc;
    }

    if (enable != 0)
    {
        count += params.ena_inc;
        if (last_cycle && (params.compute_cycle_length > 1))
        {
            count += params.last_enable_inc;
        }
    }

    if (params.use_sequencer_disable)
    {
        if (enable_zero_forcing_sequencer != 0)
        {
            ++zero_force_count;
            if (zero_force_count == params.num_forced_zeros)
            {
                enable_zero_forcing_sequencer = 0;
            }
        }

        if (last_cycle)
        {
            enable_zero_forcing_sequencer = 1;
            zero_force_count = 0;
        }

        force_disable = zero_force_count != 0;
    }
    else if (params.use_delay_disable)
    {
        force_disable = last_cycle;
    }

    // update the enable value
    enable = ((count < 0) && (!force_disable) && (enable_in != 0)) ? 1 : 0;

    // latency not modeled at this level, copy current value to output
    enable_out = enable;
}

// --------------------------------

inline void step_cma_add(const int64_t* const prod_arr, int64_t* const sums_arr, const cma_add_params& params, int64_t sub_ctrl,
                         int64_t neg_ctrl, int64_t& region_sum) noexcept
{
    bool sub_ctrl_value = test_bit(0, sub_ctrl);
    bool neg_ctrl_value = test_bit(0, neg_ctrl);

    for (int k = 0, i = 0; i < params.systolic_region_count; ++i)
    {
        const int mults_in_region = csl::min(params.systolic_region_size, params.n_mults - k);
        region_sum = 0;

        if (sub_ctrl_value)
        {
            region_sum = prod_arr[params.pipeline_depth + 1];
            region_sum -= prod_arr[0];
            k += 2;
        }
        else
        {
            for (int j = 0; j < mults_in_region; ++j)
            {
                region_sum += prod_arr[k * (params.pipeline_depth + 1)];
                k++;
            }
        }

        const int i1 = i + 1;
        if (i1 == params.systolic_region_count)
        {
            sums_arr[i] = region_sum;
        }
        else
        {
            if (neg_ctrl_value)
            {
                sums_arr[i] = sums_arr[i1];
                sums_arr[i] -= region_sum;
            }
            else
            {
                sums_arr[i] = sums_arr[i1];
                sums_arr[i] += region_sum;
            }
        }
    }
}

// --------------------------------

inline void step_fifo(int64_t* store, const fifo_params& params, int64_t& data, int64_t write_en, int64_t read_en,
                      int64_t flush) noexcept
{
    const int base_index = params.base_index;
    const int read_ptr_index = base_index + 4 + params.depth;
    const int write_ptr_index = base_index + 5 + params.depth;
    const int user_sclr = params.user_sclr;

    int64_t read_ptr_64, write_ptr_64;
    set(read_ptr_64, store[read_ptr_index]);
    set(write_ptr_64, store[write_ptr_index]);
    int read_ptr = static_cast<int>(read_ptr_64);
    int write_ptr = static_cast<int>(write_ptr_64);

    const int twice_depth = 2 * params.depth;
    // (mod 2n) so that full and empty queue states can be disambiguated
    // in both of these cases
    //               mod(write_ptr - read_ptr,   m_depth) == 0
    // but if empty, mod(write_ptr - read_ptr, 2*m_depth) == 0
    // and if full,  mod(write_ptr - read_ptr, 2*m_depth) == m_depth

    int count = ((write_ptr - read_ptr) % twice_depth + twice_depth) % twice_depth;

    // update state independent of input signals
    int write_ptr_delayed;
    int count_delayed;
    bool clearing = user_sclr && flush != 0;

    // update state according to input operands
    if (clearing)
    {
        // Operation during SCLR

        // Flush input is high - clear FIFO state
        int frame_size = 7 + params.depth + params.write_latency; // room for FIFO + circular Buffer + state variables

        for (int i = 0; i < frame_size; ++i)
        {
            // delayed write pipeline -> set zero across the whole frame
            store[base_index + i] = 0;
        }

        // Flush the FIFO by resetting the address counters
        read_ptr = base_index; // Pointers to the start
        write_ptr = base_index;
        write_ptr_delayed = base_index;
        count_delayed = 0; // Counters to zero
        count = 0;

        store[read_ptr_index] = read_ptr; // Reset read pointer
        store[write_ptr_index] = write_ptr; // Reset write pointer
    }
    else
    {
        // Normal operation

        // update state independent of input signals (other than sclr)
        for (int i = params.write_latency; i > 0; --i)
        {
            // delayed write pipeline
            store[write_ptr_index + i] = store[write_ptr_index + i - 1];
        }

        if (write_en != 0)
        {
            if (count == params.depth)
            {
                // Warn on write when full
                warning("FIFO_WRITE_WHILE_FULL");
            }
            if (count < params.depth)
            {
                // enqueue (write enable is high) in circular buffer
                store[base_index + 4 + (write_ptr % params.depth)] = data;
                write_ptr = (write_ptr + 1) % twice_depth;
                ++count;
                store[write_ptr_index] = write_ptr;
            }
        }

        int64_t write_ptr_delayed_64;
        set(write_ptr_delayed_64, store[write_ptr_index + params.write_latency]);
        write_ptr_delayed = static_cast<int>(write_ptr_delayed_64);
        count_delayed = ((write_ptr_delayed - read_ptr) % twice_depth + twice_depth) % twice_depth;

        const int maxCountIdx = base_index + 6 + params.depth + params.write_latency;
        int64_t maxCount;
        set(maxCount, store[maxCountIdx]);
        if (count > maxCount)
        {
            store[maxCountIdx] = count;
        }

        if (read_en != 0)
        {
            // dequeue (read enable is high) from circular buffer
            if (count_delayed > 0)
            {
                read_ptr = (read_ptr + 1) % twice_depth;
                --count_delayed;
                --count;
                store[read_ptr_index] = read_ptr;
            }
            else
            {
                // Warn on read when empty
                warning("FIFO_READACK_VALID_LOW");
            }
        }
    }

    // output register update
    bool valid = count_delayed > 0;
    store[base_index + 0] = valid ? 1 : 0;
    store[base_index + 1] = (count >= params.fill_threshold) ? 1 : 0;
    store[base_index + 2] = (count >= params.full_threshold) ? 1 : 0;

    if (count_delayed > 0)
    {
        // update data output register
        store[base_index + 3] = store[base_index + 4 + (read_ptr % params.depth)];
    }
}

// --------------------------------

#ifdef CSL_USE_GMP

#ifdef CSL_USE_MPFR

inline void step_fp_add(int64_t& iq, int64_t ia, const mp_int& ib) noexcept
{
    set(iq, fp32(fp32(ia).get() + fp32(ib).get()).get_u32());
}

#endif // CSL_USE_MPFR
#ifdef CSL_USE_MPFR

inline void step_fp_add(int64_t& iq, const mp_int& ia, int64_t ib) noexcept
{
    set(iq, fp32(fp32(ia).get() + fp32(ib).get()).get_u32());
}

#endif // CSL_USE_MPFR
#ifdef CSL_USE_MPFR

inline void step_fp_add(int64_t& iq, const mp_int& ia, const mp_int& ib) noexcept
{
    set(iq, fp32(fp32(ia).get() + fp32(ib).get()).get_u32());
}

#endif // CSL_USE_MPFR
#ifdef CSL_USE_MPFR

inline void step_fp_add(mp_int& iq, int64_t ia, int64_t ib) noexcept
{
    set(iq, fp32(fp32(ia).get() + fp32(ib).get()).get_u32());
}

#endif // CSL_USE_MPFR
#ifdef CSL_USE_MPFR

inline void step_fp_add(mp_int& iq, int64_t ia, const mp_int& ib) noexcept
{
    set(iq, fp32(fp32(ia).get() + fp32(ib).get()).get_u32());
}

#endif // CSL_USE_MPFR
#ifdef CSL_USE_MPFR

inline void step_fp_add(mp_int& iq, const mp_int& ia, int64_t ib) noexcept
{
    set(iq, fp32(fp32(ia).get() + fp32(ib).get()).get_u32());
}

#endif // CSL_USE_MPFR
#ifdef CSL_USE_MPFR

inline void step_fp_add(mp_int& iq, const mp_int& ia, const mp_int& ib) noexcept
{
    set(iq, fp32(fp32(ia).get() + fp32(ib).get()).get_u32());
}

#endif // CSL_USE_MPFR
#ifdef CSL_USE_MPFR

inline void step_fp_sub(int64_t& iq, int64_t ia, const mp_int& ib) noexcept
{
    set(iq, fp32(fp32(ia).get() - fp32(ib).get()).get_u32());
}

#endif // CSL_USE_MPFR
#ifdef CSL_USE_MPFR

inline void step_fp_sub(int64_t& iq, const mp_int& ia, int64_t ib) noexcept
{
    set(iq, fp32(fp32(ia).get() - fp32(ib).get()).get_u32());
}

#endif // CSL_USE_MPFR
#ifdef CSL_USE_MPFR

inline void step_fp_sub(int64_t& iq, const mp_int& ia, const mp_int& ib) noexcept
{
    set(iq, fp32(fp32(ia).get() - fp32(ib).get()).get_u32());
}

#endif // CSL_USE_MPFR
#ifdef CSL_USE_MPFR

inline void step_fp_sub(mp_int& iq, int64_t ia, int64_t ib) noexcept
{
    set(iq, fp32(fp32(ia).get() - fp32(ib).get()).get_u32());
}

#endif // CSL_USE_MPFR
#ifdef CSL_USE_MPFR

inline void step_fp_sub(mp_int& iq, int64_t ia, const mp_int& ib) noexcept
{
    set(iq, fp32(fp32(ia).get() - fp32(ib).get()).get_u32());
}

#endif // CSL_USE_MPFR
#ifdef CSL_USE_MPFR

inline void step_fp_sub(mp_int& iq, const mp_int& ia, int64_t ib) noexcept
{
    set(iq, fp32(fp32(ia).get() - fp32(ib).get()).get_u32());
}

#endif // CSL_USE_MPFR
#ifdef CSL_USE_MPFR

inline void step_fp_sub(mp_int& iq, const mp_int& ia, const mp_int& ib) noexcept
{
    set(iq, fp32(fp32(ia).get() - fp32(ib).get()).get_u32());
}

#endif // CSL_USE_MPFR
#ifdef CSL_USE_MPFR

inline void step_fp_mul(int64_t& iq, int64_t ia, const mp_int& ib) noexcept
{
    set(iq, fp_mul_impl(fp32(ia), fp32(ib)).get_u32());
}

#endif // CSL_USE_MPFR
#ifdef CSL_USE_MPFR

inline void step_fp_mul(int64_t& iq, const mp_int& ia, int64_t ib) noexcept
{
    set(iq, fp_mul_impl(fp32(ia), fp32(ib)).get_u32());
}

#endif // CSL_USE_MPFR
#ifdef CSL_USE_MPFR

inline void step_fp_mul(int64_t& iq, const mp_int& ia, const mp_int& ib) noexcept
{
    set(iq, fp_mul_impl(fp32(ia), fp32(ib)).get_u32());
}

#endif // CSL_USE_MPFR
#ifdef CSL_USE_MPFR

inline void step_fp_mul(mp_int& iq, int64_t ia, int64_t ib) noexcept
{
    set(iq, fp_mul_impl(fp32(ia), fp32(ib)).get_u32());
}

#endif // CSL_USE_MPFR
#ifdef CSL_USE_MPFR

inline void step_fp_mul(mp_int& iq, int64_t ia, const mp_int& ib) noexcept
{
    set(iq, fp_mul_impl(fp32(ia), fp32(ib)).get_u32());
}

#endif // CSL_USE_MPFR
#ifdef CSL_USE_MPFR

inline void step_fp_mul(mp_int& iq, const mp_int& ia, int64_t ib) noexcept
{
    set(iq, fp_mul_impl(fp32(ia), fp32(ib)).get_u32());
}

#endif // CSL_USE_MPFR
#ifdef CSL_USE_MPFR

inline void step_fp_mul(mp_int& iq, const mp_int& ia, const mp_int& ib) noexcept
{
    set(iq, fp_mul_impl(fp32(ia), fp32(ib)).get_u32());
}

#endif // CSL_USE_MPFR
inline void step_add(int64_t& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta += ib;
    set(iq, ta);
}

inline void step_add(int64_t& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta += ib;
    set(iq, ta);
}

inline void step_add(int64_t& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta += ib;
    set(iq, ta);
}

inline void step_add(mp_int& iq, int64_t ia, int64_t ib, temps_span temps) noexcept
{
    int64_t ta;
    ta = ia;
    ta += ib;
    set(iq, ta);
}

inline void step_add(mp_int& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta += ib;
    set(iq, ta);
}

inline void step_add(mp_int& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta += ib;
    set(iq, ta);
}

inline void step_add(mp_int& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta += ib;
    set(iq, ta);
}

inline void step_mul(int64_t& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta *= ib;
    set(iq, ta);
}

inline void step_mul(int64_t& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta *= ib;
    set(iq, ta);
}

inline void step_mul(int64_t& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta *= ib;
    set(iq, ta);
}

inline void step_mul(mp_int& iq, int64_t ia, int64_t ib, temps_span temps) noexcept
{
    int64_t ta;
    ta = ia;
    ta *= ib;
    set(iq, ta);
}

inline void step_mul(mp_int& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta *= ib;
    set(iq, ta);
}

inline void step_mul(mp_int& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta *= ib;
    set(iq, ta);
}

inline void step_mul(mp_int& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta *= ib;
    set(iq, ta);
}

inline void step_sub(int64_t& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta -= ib;
    set(iq, ta);
}

inline void step_sub(int64_t& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta -= ib;
    set(iq, ta);
}

inline void step_sub(int64_t& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta -= ib;
    set(iq, ta);
}

inline void step_sub(mp_int& iq, int64_t ia, int64_t ib, temps_span temps) noexcept
{
    int64_t ta;
    ta = ia;
    ta -= ib;
    set(iq, ta);
}

inline void step_sub(mp_int& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta -= ib;
    set(iq, ta);
}

inline void step_sub(mp_int& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta -= ib;
    set(iq, ta);
}

inline void step_sub(mp_int& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta -= ib;
    set(iq, ta);
}

inline void step_addsub(int64_t ctrl, int64_t& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept
{
    if (!is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_addsub(int64_t ctrl, int64_t& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept
{
    if (!is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_addsub(int64_t ctrl, int64_t& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept
{
    if (!is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_addsub(int64_t ctrl, mp_int& iq, int64_t ia, int64_t ib, temps_span temps) noexcept
{
    if (!is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_addsub(int64_t ctrl, mp_int& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept
{
    if (!is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_addsub(int64_t ctrl, mp_int& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept
{
    if (!is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_addsub(int64_t ctrl, mp_int& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept
{
    if (!is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_addsub(const mp_int& ctrl, int64_t& iq, int64_t ia, int64_t ib, temps_span temps) noexcept
{
    if (!is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_addsub(const mp_int& ctrl, int64_t& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept
{
    if (!is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_addsub(const mp_int& ctrl, int64_t& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept
{
    if (!is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_addsub(const mp_int& ctrl, int64_t& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept
{
    if (!is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_addsub(const mp_int& ctrl, mp_int& iq, int64_t ia, int64_t ib, temps_span temps) noexcept
{
    if (!is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_addsub(const mp_int& ctrl, mp_int& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept
{
    if (!is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_addsub(const mp_int& ctrl, mp_int& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept
{
    if (!is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_addsub(const mp_int& ctrl, mp_int& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept
{
    if (!is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_subadd(int64_t ctrl, int64_t& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept
{
    if (is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_subadd(int64_t ctrl, int64_t& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept
{
    if (is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_subadd(int64_t ctrl, int64_t& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept
{
    if (is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_subadd(int64_t ctrl, mp_int& iq, int64_t ia, int64_t ib, temps_span temps) noexcept
{
    if (is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_subadd(int64_t ctrl, mp_int& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept
{
    if (is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_subadd(int64_t ctrl, mp_int& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept
{
    if (is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_subadd(int64_t ctrl, mp_int& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept
{
    if (is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_subadd(const mp_int& ctrl, int64_t& iq, int64_t ia, int64_t ib, temps_span temps) noexcept
{
    if (is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_subadd(const mp_int& ctrl, int64_t& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept
{
    if (is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_subadd(const mp_int& ctrl, int64_t& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept
{
    if (is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_subadd(const mp_int& ctrl, int64_t& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept
{
    if (is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_subadd(const mp_int& ctrl, mp_int& iq, int64_t ia, int64_t ib, temps_span temps) noexcept
{
    if (is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_subadd(const mp_int& ctrl, mp_int& iq, int64_t ia, const mp_int& ib, temps_span temps) noexcept
{
    if (is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_subadd(const mp_int& ctrl, mp_int& iq, const mp_int& ia, int64_t ib, temps_span temps) noexcept
{
    if (is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_subadd(const mp_int& ctrl, mp_int& iq, const mp_int& ia, const mp_int& ib, temps_span temps) noexcept
{
    if (is_zero(ctrl))
    {
        step_add(iq, ia, ib, temps);
    }
    else
    {
        step_sub(iq, ia, ib, temps);
    }
}

inline void step_const(int64_t& iq, const mp_int& ia) noexcept
{
    set(iq, ia);
}

inline void step_const(mp_int& iq, int64_t ia) noexcept
{
    set(iq, ia);
}

inline void step_const(mp_int& iq, const mp_int& ia) noexcept
{
    set(iq, ia);
}

inline void step_and(int64_t& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    mp_int& ta = temps.at(0);
    ta = ia;
    ta &= ib;
    set(iq, ta);
}

inline void step_and(int64_t& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    mp_int& ta = temps.at(0);
    ta = ia;
    ta &= ib;
    set(iq, ta);
}

inline void step_and(int64_t& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    mp_int& ta = temps.at(0);
    ta = ia;
    ta &= ib;
    set(iq, ta);
}

inline void step_and(mp_int& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    int64_t ta;
    ta = ia;
    ta &= ib;
    set(iq, ta);
}

inline void step_and(mp_int& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    mp_int& ta = temps.at(0);
    ta = ia;
    ta &= ib;
    set(iq, ta);
}

inline void step_and(mp_int& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    mp_int& ta = temps.at(0);
    ta = ia;
    ta &= ib;
    set(iq, ta);
}

inline void step_and(mp_int& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    mp_int& ta = temps.at(0);
    ta = ia;
    ta &= ib;
    set(iq, ta);
}

inline void step_or(int64_t& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    mp_int& ta = temps.at(0);
    ta = ia;
    ta |= ib;
    set(iq, ta);
}

inline void step_or(int64_t& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    mp_int& ta = temps.at(0);
    ta = ia;
    ta |= ib;
    set(iq, ta);
}

inline void step_or(int64_t& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    mp_int& ta = temps.at(0);
    ta = ia;
    ta |= ib;
    set(iq, ta);
}

inline void step_or(mp_int& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    int64_t ta;
    ta = ia;
    ta |= ib;
    set(iq, ta);
}

inline void step_or(mp_int& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    mp_int& ta = temps.at(0);
    ta = ia;
    ta |= ib;
    set(iq, ta);
}

inline void step_or(mp_int& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    mp_int& ta = temps.at(0);
    ta = ia;
    ta |= ib;
    set(iq, ta);
}

inline void step_or(mp_int& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    mp_int& ta = temps.at(0);
    ta = ia;
    ta |= ib;
    set(iq, ta);
}

inline void step_xor(int64_t& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    mp_int& ta = temps.at(0);
    ta = ia;
    ta ^= ib;
    set(iq, ta);
}

inline void step_xor(int64_t& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    mp_int& ta = temps.at(0);
    ta = ia;
    ta ^= ib;
    set(iq, ta);
}

inline void step_xor(int64_t& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    mp_int& ta = temps.at(0);
    ta = ia;
    ta ^= ib;
    set(iq, ta);
}

inline void step_xor(mp_int& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    int64_t ta;
    ta = ia;
    ta ^= ib;
    set(iq, ta);
}

inline void step_xor(mp_int& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    mp_int& ta = temps.at(0);
    ta = ia;
    ta ^= ib;
    set(iq, ta);
}

inline void step_xor(mp_int& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    mp_int& ta = temps.at(0);
    ta = ia;
    ta ^= ib;
    set(iq, ta);
}

inline void step_xor(mp_int& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    (void)bit_width;
    mp_int& ta = temps.at(0);
    ta = ia;
    ta ^= ib;
    set(iq, ta);
}

inline void step_nand(int64_t& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    step_and(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

inline void step_nand(int64_t& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    step_and(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

inline void step_nand(int64_t& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    step_and(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

inline void step_nand(mp_int& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    step_and(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

inline void step_nand(mp_int& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    step_and(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

inline void step_nand(mp_int& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    step_and(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

inline void step_nand(mp_int& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    step_and(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

inline void step_nor(int64_t& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    step_or(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

inline void step_nor(int64_t& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    step_or(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

inline void step_nor(int64_t& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    step_or(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

inline void step_nor(mp_int& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    step_or(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

inline void step_nor(mp_int& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    step_or(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

inline void step_nor(mp_int& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    step_or(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

inline void step_nor(mp_int& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    step_or(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

inline void step_nxor(int64_t& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    step_xor(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

inline void step_nxor(int64_t& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    step_xor(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

inline void step_nxor(int64_t& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    step_xor(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

inline void step_nxor(mp_int& iq, int64_t ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    step_xor(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

inline void step_nxor(mp_int& iq, int64_t ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    step_xor(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

inline void step_nxor(mp_int& iq, const mp_int& ia, int64_t ib, int bit_width, temps_span temps) noexcept
{
    step_xor(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

inline void step_nxor(mp_int& iq, const mp_int& ia, const mp_int& ib, int bit_width, temps_span temps) noexcept
{
    step_xor(iq, ia, ib, bit_width, temps);
    not_n(iq, iq, bit_width);
}

inline void step_reducing_or(int64_t& iq, const mp_int& ia) noexcept
{
    set(iq, (ia == 0) ? 0 : 1);
}

inline void step_reducing_or(mp_int& iq, int64_t ia) noexcept
{
    set(iq, (ia == 0) ? 0 : 1);
}

inline void step_reducing_or(mp_int& iq, const mp_int& ia) noexcept
{
    set(iq, (ia == 0) ? 0 : 1);
}

inline void step_reducing_nor(int64_t& iq, const mp_int& ia) noexcept
{
    step_reducing_or(iq, ia);
    boolean_complement(iq, iq);
}

inline void step_reducing_nor(mp_int& iq, int64_t ia) noexcept
{
    step_reducing_or(iq, ia);
    boolean_complement(iq, iq);
}

inline void step_reducing_nor(mp_int& iq, const mp_int& ia) noexcept
{
    step_reducing_or(iq, ia);
    boolean_complement(iq, iq);
}

inline void step_reducing_and(mp_int& iq, int64_t ia, size_t bit_width, bool is_signed, temps_span temps) noexcept
{
    (void)is_signed;
    (void)temps;
    uint64_t uia = *reinterpret_cast<const uint64_t*>(&ia);
    uint64_t mask = bit_mask_u64(bit_width);
    set(iq, (mask & uia) == mask);
}

inline void step_reducing_and(int64_t& iq, const mp_int& ia, size_t bit_width, bool is_signed, temps_span temps) noexcept
{
    bool result = false;
    if (is_signed)
    {
        result = (ia == -1);
    }
    else
    {
        mp_int& q = temps.at(0);
        q = ia;
        q += 1;
        // is adding 1 causes nth-bit carry, then all n bits were set
        result = test_bit(bit_width, q) && !test_bit(bit_width, ia);
    }
    set(iq, result ? 1 : 0);
}

inline void step_reducing_and(mp_int& iq, const mp_int& ia, size_t bit_width, bool is_signed, temps_span temps) noexcept
{
    bool result = false;
    if (is_signed)
    {
        result = (ia == -1);
    }
    else
    {
        mp_int& q = temps.at(0);
        q = ia;
        q += 1;
        // is adding 1 causes nth-bit carry, then all n bits were set
        result = test_bit(bit_width, q) && !test_bit(bit_width, ia);
    }
    set(iq, result ? 1 : 0);
}

inline void step_ld_exp(int64_t& iq, const mp_int& ia, int64_t ib, bool reverse, temps_span temps) noexcept
{
    int exp = static_cast<int>(ib);
    mp_int& tmp = temps.at(0);
    ld_exp(tmp, ia, reverse ? -exp : exp);
    set(iq, tmp);
}

inline void step_ld_exp(mp_int& iq, int64_t ia, int64_t ib, bool reverse, temps_span temps) noexcept
{
    int exp = static_cast<int>(ib);
    int64_t tmp;
    ld_exp(tmp, ia, reverse ? -exp : exp);
    set(iq, tmp);
}

inline void step_ld_exp(mp_int& iq, const mp_int& ia, int64_t ib, bool reverse, temps_span temps) noexcept
{
    int exp = static_cast<int>(ib);
    mp_int& tmp = temps.at(0);
    ld_exp(tmp, ia, reverse ? -exp : exp);
    set(iq, tmp);
}

inline void step_ld_exp(int64_t& iq, int64_t ia, const mp_int& ib, bool reverse, temps_span temps) noexcept
{
    int ibi = to_i32(ib);
    int64_t tmp;
    ld_exp(tmp, ia, reverse ? (-ibi) : ibi);
    set(iq, tmp);
}

inline void step_ld_exp(int64_t& iq, const mp_int& ia, const mp_int& ib, bool reverse, temps_span temps) noexcept
{
    int ibi = to_i32(ib);
    mp_int& tmp = temps.at(0);
    ld_exp(tmp, ia, reverse ? (-ibi) : ibi);
    set(iq, tmp);
}

inline void step_ld_exp(mp_int& iq, int64_t ia, const mp_int& ib, bool reverse, temps_span temps) noexcept
{
    int ibi = to_i32(ib);
    int64_t tmp;
    ld_exp(tmp, ia, reverse ? (-ibi) : ibi);
    set(iq, tmp);
}

inline void step_ld_exp(mp_int& iq, const mp_int& ia, const mp_int& ib, bool reverse, temps_span temps) noexcept
{
    int ibi = to_i32(ib);
    mp_int& tmp = temps.at(0);
    ld_exp(tmp, ia, reverse ? (-ibi) : ibi);
    set(iq, tmp);
}

inline void step_equal(int64_t& iq, int64_t ia, const mp_int& ib) noexcept
{
    set(iq, (ia == ib) ? 1 : 0);
}

inline void step_equal(int64_t& iq, const mp_int& ia, int64_t ib) noexcept
{
    set(iq, (ia == ib) ? 1 : 0);
}

inline void step_equal(int64_t& iq, const mp_int& ia, const mp_int& ib) noexcept
{
    set(iq, (ia == ib) ? 1 : 0);
}

inline void step_equal(mp_int& iq, int64_t ia, int64_t ib) noexcept
{
    set(iq, (ia == ib) ? 1 : 0);
}

inline void step_equal(mp_int& iq, int64_t ia, const mp_int& ib) noexcept
{
    set(iq, (ia == ib) ? 1 : 0);
}

inline void step_equal(mp_int& iq, const mp_int& ia, int64_t ib) noexcept
{
    set(iq, (ia == ib) ? 1 : 0);
}

inline void step_equal(mp_int& iq, const mp_int& ia, const mp_int& ib) noexcept
{
    set(iq, (ia == ib) ? 1 : 0);
}

inline void step_nequal(int64_t& iq, int64_t ia, const mp_int& ib) noexcept
{
    set(iq, (ia != ib) ? 1 : 0);
}

inline void step_nequal(int64_t& iq, const mp_int& ia, int64_t ib) noexcept
{
    set(iq, (ia != ib) ? 1 : 0);
}

inline void step_nequal(int64_t& iq, const mp_int& ia, const mp_int& ib) noexcept
{
    set(iq, (ia != ib) ? 1 : 0);
}

inline void step_nequal(mp_int& iq, int64_t ia, int64_t ib) noexcept
{
    set(iq, (ia != ib) ? 1 : 0);
}

inline void step_nequal(mp_int& iq, int64_t ia, const mp_int& ib) noexcept
{
    set(iq, (ia != ib) ? 1 : 0);
}

inline void step_nequal(mp_int& iq, const mp_int& ia, int64_t ib) noexcept
{
    set(iq, (ia != ib) ? 1 : 0);
}

inline void step_nequal(mp_int& iq, const mp_int& ia, const mp_int& ib) noexcept
{
    set(iq, (ia != ib) ? 1 : 0);
}

inline void step_reducing_nand(int64_t& iq, const mp_int& ia, size_t bit_width, bool is_signed, temps_span temps) noexcept
{
    step_reducing_and(iq, ia, bit_width, is_signed, temps);
    boolean_complement(iq, iq);
}

inline void step_reducing_nand(mp_int& iq, int64_t ia, size_t bit_width, bool is_signed, temps_span temps) noexcept
{
    step_reducing_and(iq, ia, bit_width, is_signed, temps);
    boolean_complement(iq, iq);
}

inline void step_reducing_nand(mp_int& iq, const mp_int& ia, size_t bit_width, bool is_signed, temps_span temps) noexcept
{
    step_reducing_and(iq, ia, bit_width, is_signed, temps);
    boolean_complement(iq, iq);
}

inline void step_reducing_xor(int64_t& iq, const mp_int& ia, size_t bit_width) noexcept
{
    bool result = test_bit(0, ia);
    for (unsigned int i = 1; i < bit_width; ++i)
    {
        result = result ^ test_bit(i, ia);
    }
    set(iq, result ? 1 : 0);
}

inline void step_reducing_xor(mp_int& iq, int64_t ia, size_t bit_width) noexcept
{
    bool result = test_bit(0, ia);
    for (unsigned int i = 1; i < bit_width; ++i)
    {
        result = result ^ test_bit(i, ia);
    }
    set(iq, result ? 1 : 0);
}

inline void step_reducing_xor(mp_int& iq, const mp_int& ia, size_t bit_width) noexcept
{
    bool result = test_bit(0, ia);
    for (unsigned int i = 1; i < bit_width; ++i)
    {
        result = result ^ test_bit(i, ia);
    }
    set(iq, result ? 1 : 0);
}

inline void step_reducing_nxor(int64_t& iq, const mp_int& ia, size_t bit_width) noexcept
{
    step_reducing_xor(iq, ia, bit_width);
    boolean_complement(iq, iq);
}

inline void step_reducing_nxor(mp_int& iq, int64_t ia, size_t bit_width) noexcept
{
    step_reducing_xor(iq, ia, bit_width);
    boolean_complement(iq, iq);
}

inline void step_reducing_nxor(mp_int& iq, const mp_int& ia, size_t bit_width) noexcept
{
    step_reducing_xor(iq, ia, bit_width);
    boolean_complement(iq, iq);
}

CSL_FORCE_INLINE void step_bit_extract(int64_t& iq, const mp_int& ia, size_t width, bool signed_extend, int bit_pos,
                                       temps_span temps) noexcept
{
    mp_int& result = temps.at(0);
    ld_exp(result, ia, -bit_pos);
    if (signed_extend && test_bit(width - 1, result))
    {
        set_upper(result, width - 1);
    }
    else
    {
        mask_lower(result, result, width);
    }
    set(iq, result);
}

CSL_FORCE_INLINE void step_bit_extract(mp_int& iq, int64_t ia, size_t width, bool signed_extend, int bit_pos,
                                       temps_span temps) noexcept
{
    int64_t result;
    ld_exp(result, ia, -bit_pos);
    if (signed_extend && test_bit(width - 1, result))
    {
        set_upper(result, width - 1);
    }
    else
    {
        mask_lower(result, result, width);
    }
    set(iq, result);
}

CSL_FORCE_INLINE void step_bit_extract(mp_int& iq, const mp_int& ia, size_t width, bool signed_extend, int bit_pos,
                                       temps_span temps) noexcept
{
    mp_int& result = temps.at(0);
    ld_exp(result, ia, -bit_pos);
    if (signed_extend && test_bit(width - 1, result))
    {
        set_upper(result, width - 1);
    }
    else
    {
        mask_lower(result, result, width);
    }
    set(iq, result);
}

inline void step_biased_round(int64_t& iq, const mp_int& ia, int bit, temps_span temps) noexcept
{
    mp_int& one = temps.at(1);
    one = int64_t(1);
    mp_int& half = temps.at(0);
    ld_exp(half, one, bit - 1);
    half += ia;
    ld_exp(half, half, -bit);
    set(iq, half);
}

inline void step_biased_round(mp_int& iq, int64_t ia, int bit, temps_span temps) noexcept
{
    static constexpr int64_t one{1};
    int64_t half;
    ld_exp(half, one, bit - 1);
    half += ia;
    ld_exp(half, half, -bit);
    set(iq, half);
}

inline void step_biased_round(mp_int& iq, const mp_int& ia, int bit, temps_span temps) noexcept
{
    mp_int& one = temps.at(1);
    one = int64_t(1);
    mp_int& half = temps.at(0);
    ld_exp(half, one, bit - 1);
    half += ia;
    ld_exp(half, half, -bit);
    set(iq, half);
}

inline void step_unbiased_round(int64_t& iq, const mp_int& ia, int bit, temps_span temps) noexcept
{
    mp_int& half = temps.at(0);
    mp_int& frac_mask = temps.at(1);
    mp_int& one = temps.at(2);
    one = int64_t(1);
    ld_exp(half, one, bit - 1);

    mask_lower(frac_mask, ia, bit);
    if (frac_mask == half)
    {
        mp_int& q = half;
        ld_exp(q, ia, -bit);
        if (test_bit(0, q))
        {
            q += 1;
        }
        set(iq, q);
    }
    else
    {
        step_biased_round(iq, ia, bit, temps.next(3));
    }
}

inline void step_unbiased_round(mp_int& iq, int64_t ia, int bit, temps_span temps) noexcept
{
    int64_t half;
    int64_t frac_mask;
    static constexpr int64_t one{1};
    ld_exp(half, one, bit - 1);

    mask_lower(frac_mask, ia, bit);
    if (frac_mask == half)
    {
        int64_t& q = half;
        ld_exp(q, ia, -bit);
        if (test_bit(0, q))
        {
            q += 1;
        }
        set(iq, q);
    }
    else
    {
        step_biased_round(iq, ia, bit, temps.next(3));
    }
}

inline void step_unbiased_round(mp_int& iq, const mp_int& ia, int bit, temps_span temps) noexcept
{
    mp_int& half = temps.at(0);
    mp_int& frac_mask = temps.at(1);
    mp_int& one = temps.at(2);
    one = int64_t(1);
    ld_exp(half, one, bit - 1);

    mask_lower(frac_mask, ia, bit);
    if (frac_mask == half)
    {
        mp_int& q = half;
        ld_exp(q, ia, -bit);
        if (test_bit(0, q))
        {
            q += 1;
        }
        set(iq, q);
    }
    else
    {
        step_biased_round(iq, ia, bit, temps.next(3));
    }
}

inline void step_bit_reverse(int64_t& iq, const mp_int& ia, size_t bit_width) noexcept
{
    set(iq, 0);
    for (size_t i = 0; i < bit_width; ++i)
    {
        if (test_bit(i, ia))
        {
            set_bit(iq, bit_width - 1 - i, false);
        }
    }
}

inline void step_bit_reverse(mp_int& iq, int64_t ia, size_t bit_width) noexcept
{
    set(iq, 0);
    for (size_t i = 0; i < bit_width; ++i)
    {
        if (test_bit(i, ia))
        {
            set_bit(iq, bit_width - 1 - i, false);
        }
    }
}

inline void step_bit_reverse(mp_int& iq, const mp_int& ia, size_t bit_width) noexcept
{
    set(iq, 0);
    for (size_t i = 0; i < bit_width; ++i)
    {
        if (test_bit(i, ia))
        {
            set_bit(iq, bit_width - 1 - i, false);
        }
    }
}

inline void step_sign_bit(int64_t& iq, const mp_int& ia, size_t bit_width) noexcept
{
    set(iq, (test_bit(bit_width - 1, ia) == true) ? 1 : 0);
}

inline void step_sign_bit(mp_int& iq, int64_t ia, size_t bit_width) noexcept
{
    set(iq, (test_bit(bit_width - 1, ia) == true) ? 1 : 0);
}

inline void step_sign_bit(mp_int& iq, const mp_int& ia, size_t bit_width) noexcept
{
    set(iq, (test_bit(bit_width - 1, ia) == true) ? 1 : 0);
}

inline void step_nsign_bit(int64_t& iq, const mp_int& ia, size_t bit_width) noexcept
{
    set(iq, (test_bit(bit_width - 1, ia) == false) ? 1 : 0);
}

inline void step_nsign_bit(mp_int& iq, int64_t ia, size_t bit_width) noexcept
{
    set(iq, (test_bit(bit_width - 1, ia) == false) ? 1 : 0);
}

inline void step_nsign_bit(mp_int& iq, const mp_int& ia, size_t bit_width) noexcept
{
    set(iq, (test_bit(bit_width - 1, ia) == false) ? 1 : 0);
}

inline void step_shift_right(int64_t& iq, const mp_int& ia, size_t amount, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta >>= amount;
    set(iq, ta);
}

inline void step_shift_right(mp_int& iq, int64_t ia, size_t amount, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta >>= amount;
    set(iq, ta);
}

inline void step_shift_right(mp_int& iq, const mp_int& ia, size_t amount, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta >>= amount;
    set(iq, ta);
}

inline void step_shift_left(int64_t& iq, const mp_int& ia, size_t amount, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta <<= amount;
    set(iq, ta);
}

inline void step_shift_left(mp_int& iq, int64_t ia, size_t amount, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta <<= amount;
    set(iq, ta);
}

inline void step_shift_left(mp_int& iq, const mp_int& ia, size_t amount, temps_span temps) noexcept
{
    mp_int& ta = temps.at(0);
    ta = ia;
    ta <<= amount;
    set(iq, ta);
}

inline void step_test_bit(int64_t& iq, const mp_int& ia, size_t bit_position) noexcept
{
    set(iq, test_bit(bit_position, ia) ? 1 : 0);
}

inline void step_test_bit(mp_int& iq, int64_t ia, size_t bit_position) noexcept
{
    set(iq, test_bit(bit_position, ia) ? 1 : 0);
}

inline void step_test_bit(mp_int& iq, const mp_int& ia, size_t bit_position) noexcept
{
    set(iq, test_bit(bit_position, ia) ? 1 : 0);
}

inline void step_set_bit(int64_t& iq, const mp_int& ia, size_t bit_position) noexcept
{
    set_bit(iq, bit_position, ia == 0);
}

inline void step_set_bit(mp_int& iq, int64_t ia, size_t bit_position) noexcept
{
    set_bit(iq, bit_position, ia == 0);
}

inline void step_set_bit(mp_int& iq, const mp_int& ia, size_t bit_position) noexcept
{
    set_bit(iq, bit_position, ia == 0);
}

inline void step_not(int64_t& iq, const mp_int& ia, int bit_width, temps_span temps) noexcept
{
    mp_int& result = temps.at(0);
    not_n(result, ia, bit_width);
    set(iq, result);
}

inline void step_not(mp_int& iq, int64_t ia, int bit_width, temps_span temps) noexcept
{
    int64_t result;
    not_n(result, ia, bit_width);
    set(iq, result);
}

inline void step_not(mp_int& iq, const mp_int& ia, int bit_width, temps_span temps) noexcept
{
    mp_int& result = temps.at(0);
    not_n(result, ia, bit_width);
    set(iq, result);
}

inline void step_not_signed(int64_t& iq, const mp_int& ia, int bit_width, temps_span temps) noexcept
{
    mp_int& result = temps.at(0);
    not_n(result, ia, bit_width);
    set(iq, result);
    if (test_bit(static_cast<size_t>(bit_width) - 1, iq))
    {
        set_upper(iq, static_cast<size_t>(bit_width));
    }
    else
    {
        mask_lower(iq, iq, static_cast<size_t>(bit_width));
    }
}

inline void step_not_signed(mp_int& iq, int64_t ia, int bit_width, temps_span temps) noexcept
{
    int64_t result;
    not_n(result, ia, bit_width);
    set(iq, result);
    if (test_bit(static_cast<size_t>(bit_width) - 1, iq))
    {
        set_upper(iq, static_cast<size_t>(bit_width));
    }
    else
    {
        mask_lower(iq, iq, static_cast<size_t>(bit_width));
    }
}

inline void step_not_signed(mp_int& iq, const mp_int& ia, int bit_width, temps_span temps) noexcept
{
    mp_int& result = temps.at(0);
    not_n(result, ia, bit_width);
    set(iq, result);
    if (test_bit(static_cast<size_t>(bit_width) - 1, iq))
    {
        set_upper(iq, static_cast<size_t>(bit_width));
    }
    else
    {
        mask_lower(iq, iq, static_cast<size_t>(bit_width));
    }
}

inline void step_sequencer(mp_int& state, mp_int& iq, const mp_int& ia, size_t offset, int64_t mod, int64_t cross,
                           temps_span temps) noexcept
{
    (void)offset;
    const bool enable = ia != 0;
    if (enable)
    {
        state += 1;
        state %= mod;
    }
    mp_int& tmp = temps.at(0);
    tmp = cross;
    iq = (state >= tmp) ? 1 : 0;
}

CSL_FORCE_INLINE void step_reduce(int64_t& iq, const mp_int& ia, size_t bit_width, temps_span temps) noexcept
{
    // test sign bit
    mp_int& a = temps.at(0);
    if (test_bit(bit_width - 1, ia) != 0)
    {
        mask_lower(a, ia, bit_width - 1);
        mp_int& one = temps.at(1);
        one = int64_t(1);
        mp_int& b = temps.at(2);
        ld_exp(b, one, (int)(bit_width - 1));
        safe_sub(iq, a, b, temps.next(3));
    }
    else
    {
        mask_lower(a, ia, bit_width - 1);
        set(iq, a);
    }
}

CSL_FORCE_INLINE void step_reduce(mp_int& iq, int64_t ia, size_t bit_width, temps_span temps) noexcept
{
    // test sign bit
    int64_t a;
    if (test_bit(bit_width - 1, ia) != 0)
    {
        mask_lower(a, ia, bit_width - 1);
        static constexpr int64_t one{1};
        int64_t b;
        ld_exp(b, one, (int)(bit_width - 1));
        safe_sub(iq, a, b, temps.next(3));
    }
    else
    {
        mask_lower(a, ia, bit_width - 1);
        set(iq, a);
    }
}

CSL_FORCE_INLINE void step_reduce(mp_int& iq, const mp_int& ia, size_t bit_width, temps_span temps) noexcept
{
    // test sign bit
    mp_int& a = temps.at(0);
    if (test_bit(bit_width - 1, ia) != 0)
    {
        mask_lower(a, ia, bit_width - 1);
        mp_int& one = temps.at(1);
        one = int64_t(1);
        mp_int& b = temps.at(2);
        ld_exp(b, one, (int)(bit_width - 1));
        safe_sub(iq, a, b, temps.next(3));
    }
    else
    {
        mask_lower(a, ia, bit_width - 1);
        set(iq, a);
    }
}

inline void step_counter(int64_t& counter, int64_t& iq, const mp_int& ia, size_t offset, int64_t inc, int64_t mod) noexcept
{
    const bool enable = (ia != 0);
    if (enable)
    {
        counter += inc;
        counter %= mod;
    }
    set(iq, counter);
    iq += offset;
}

inline void step_counter(int64_t& counter, mp_int& iq, int64_t ia, size_t offset, int64_t inc, int64_t mod) noexcept
{
    const bool enable = (ia != 0);
    if (enable)
    {
        counter += inc;
        counter %= mod;
    }
    set(iq, counter);
    iq += offset;
}

inline void step_counter(int64_t& counter, mp_int& iq, const mp_int& ia, size_t offset, int64_t inc, int64_t mod) noexcept
{
    const bool enable = (ia != 0);
    if (enable)
    {
        counter += inc;
        counter %= mod;
    }
    set(iq, counter);
    iq += offset;
}

inline void step_counter(mp_int& counter, int64_t& iq, int64_t ia, size_t offset, int64_t inc, int64_t mod) noexcept
{
    const bool enable = (ia != 0);
    if (enable)
    {
        counter += inc;
        counter %= mod;
    }
    set(iq, counter);
    iq += offset;
}

inline void step_counter(mp_int& counter, int64_t& iq, const mp_int& ia, size_t offset, int64_t inc, int64_t mod) noexcept
{
    const bool enable = (ia != 0);
    if (enable)
    {
        counter += inc;
        counter %= mod;
    }
    set(iq, counter);
    iq += offset;
}

inline void step_counter(mp_int& counter, mp_int& iq, int64_t ia, size_t offset, int64_t inc, int64_t mod) noexcept
{
    const bool enable = (ia != 0);
    if (enable)
    {
        counter += inc;
        counter %= mod;
    }
    set(iq, counter);
    iq += offset;
}

inline void step_counter(mp_int& counter, mp_int& iq, const mp_int& ia, size_t offset, int64_t inc, int64_t mod) noexcept
{
    const bool enable = (ia != 0);
    if (enable)
    {
        counter += inc;
        counter %= mod;
    }
    set(iq, counter);
    iq += offset;
}

inline void step_bit_combine(int64_t& iq, int64_t ia, const mp_int& ib, size_t num_bits, size_t index, size_t bit_pos,
                             size_t next_bit_pos, temps_span temps) noexcept
{
    if (index == 0)
    {
        mp_int& ord = temps.at(0);
        ord = ia;
        ord |= ib;
        set(iq, ord);
    }
    else
    {
        mp_int& shifted_b = temps.at(0);
        mp_int& ibsz = temps.at(1);
        ibsz = ib;
        mp_int& result = temps.at(2);
        ld_exp(shifted_b, ibsz, static_cast<int>(bit_pos));
        if (index < num_bits - 1)
        {
            // avoid masking out sign of last element
            mask_lower(shifted_b, shifted_b, next_bit_pos);
        }
        if (index == 1)
        {
            // mask 1st element during this iteration, as index = 0 can sometimes be skipped
            set(result, ia);
            mask_lower(result, result, bit_pos);
        }
        else
        {
            result = ia;
        }
        result |= shifted_b;
        set(iq, result);
    }
}

inline void step_bit_combine(int64_t& iq, const mp_int& ia, int64_t ib, size_t num_bits, size_t index, size_t bit_pos,
                             size_t next_bit_pos, temps_span temps) noexcept
{
    if (index == 0)
    {
        mp_int& ord = temps.at(0);
        ord = ia;
        ord |= ib;
        set(iq, ord);
    }
    else
    {
        mp_int& shifted_b = temps.at(0);
        mp_int& ibsz = temps.at(1);
        ibsz = ib;
        mp_int& result = temps.at(2);
        ld_exp(shifted_b, ibsz, static_cast<int>(bit_pos));
        if (index < num_bits - 1)
        {
            // avoid masking out sign of last element
            mask_lower(shifted_b, shifted_b, next_bit_pos);
        }
        if (index == 1)
        {
            // mask 1st element during this iteration, as index = 0 can sometimes be skipped
            set(result, ia);
            mask_lower(result, result, bit_pos);
        }
        else
        {
            result = ia;
        }
        result |= shifted_b;
        set(iq, result);
    }
}

inline void step_bit_combine(int64_t& iq, const mp_int& ia, const mp_int& ib, size_t num_bits, size_t index, size_t bit_pos,
                             size_t next_bit_pos, temps_span temps) noexcept
{
    if (index == 0)
    {
        mp_int& ord = temps.at(0);
        ord = ia;
        ord |= ib;
        set(iq, ord);
    }
    else
    {
        mp_int& shifted_b = temps.at(0);
        mp_int& ibsz = temps.at(1);
        ibsz = ib;
        mp_int& result = temps.at(2);
        ld_exp(shifted_b, ibsz, static_cast<int>(bit_pos));
        if (index < num_bits - 1)
        {
            // avoid masking out sign of last element
            mask_lower(shifted_b, shifted_b, next_bit_pos);
        }
        if (index == 1)
        {
            // mask 1st element during this iteration, as index = 0 can sometimes be skipped
            set(result, ia);
            mask_lower(result, result, bit_pos);
        }
        else
        {
            result = ia;
        }
        result |= shifted_b;
        set(iq, result);
    }
}

inline void step_bit_combine(mp_int& iq, int64_t ia, int64_t ib, size_t num_bits, size_t index, size_t bit_pos, size_t next_bit_pos,
                             temps_span temps) noexcept
{
    if (index == 0)
    {
        int64_t ord;
        ord = ia;
        ord |= ib;
        set(iq, ord);
    }
    else
    {
        mp_int& shifted_b = temps.at(0);
        mp_int& ibsz = temps.at(1);
        ibsz = ib;
        mp_int& result = temps.at(2);
        ld_exp(shifted_b, ibsz, static_cast<int>(bit_pos));
        if (index < num_bits - 1)
        {
            // avoid masking out sign of last element
            mask_lower(shifted_b, shifted_b, next_bit_pos);
        }
        if (index == 1)
        {
            // mask 1st element during this iteration, as index = 0 can sometimes be skipped
            set(result, ia);
            mask_lower(result, result, bit_pos);
        }
        else
        {
            result = ia;
        }
        result |= shifted_b;
        set(iq, result);
    }
}

inline void step_bit_combine(mp_int& iq, int64_t ia, const mp_int& ib, size_t num_bits, size_t index, size_t bit_pos,
                             size_t next_bit_pos, temps_span temps) noexcept
{
    if (index == 0)
    {
        mp_int& ord = temps.at(0);
        ord = ia;
        ord |= ib;
        set(iq, ord);
    }
    else
    {
        mp_int& shifted_b = temps.at(0);
        mp_int& ibsz = temps.at(1);
        ibsz = ib;
        mp_int& result = temps.at(2);
        ld_exp(shifted_b, ibsz, static_cast<int>(bit_pos));
        if (index < num_bits - 1)
        {
            // avoid masking out sign of last element
            mask_lower(shifted_b, shifted_b, next_bit_pos);
        }
        if (index == 1)
        {
            // mask 1st element during this iteration, as index = 0 can sometimes be skipped
            set(result, ia);
            mask_lower(result, result, bit_pos);
        }
        else
        {
            result = ia;
        }
        result |= shifted_b;
        set(iq, result);
    }
}

inline void step_bit_combine(mp_int& iq, const mp_int& ia, int64_t ib, size_t num_bits, size_t index, size_t bit_pos,
                             size_t next_bit_pos, temps_span temps) noexcept
{
    if (index == 0)
    {
        mp_int& ord = temps.at(0);
        ord = ia;
        ord |= ib;
        set(iq, ord);
    }
    else
    {
        mp_int& shifted_b = temps.at(0);
        mp_int& ibsz = temps.at(1);
        ibsz = ib;
        mp_int& result = temps.at(2);
        ld_exp(shifted_b, ibsz, static_cast<int>(bit_pos));
        if (index < num_bits - 1)
        {
            // avoid masking out sign of last element
            mask_lower(shifted_b, shifted_b, next_bit_pos);
        }
        if (index == 1)
        {
            // mask 1st element during this iteration, as index = 0 can sometimes be skipped
            set(result, ia);
            mask_lower(result, result, bit_pos);
        }
        else
        {
            result = ia;
        }
        result |= shifted_b;
        set(iq, result);
    }
}

inline void step_bit_combine(mp_int& iq, const mp_int& ia, const mp_int& ib, size_t num_bits, size_t index, size_t bit_pos,
                             size_t next_bit_pos, temps_span temps) noexcept
{
    if (index == 0)
    {
        mp_int& ord = temps.at(0);
        ord = ia;
        ord |= ib;
        set(iq, ord);
    }
    else
    {
        mp_int& shifted_b = temps.at(0);
        mp_int& ibsz = temps.at(1);
        ibsz = ib;
        mp_int& result = temps.at(2);
        ld_exp(shifted_b, ibsz, static_cast<int>(bit_pos));
        if (index < num_bits - 1)
        {
            // avoid masking out sign of last element
            mask_lower(shifted_b, shifted_b, next_bit_pos);
        }
        if (index == 1)
        {
            // mask 1st element during this iteration, as index = 0 can sometimes be skipped
            set(result, ia);
            mask_lower(result, result, bit_pos);
        }
        else
        {
            result = ia;
        }
        result |= shifted_b;
        set(iq, result);
    }
}

inline bool get_lookup_value(const uint64_t* values, const mp_int_info* infos, uint64_t n, uint64_t key, mp_int& out) noexcept
{
    if (key < n)
    {
        csl::fill_mpz_data(out, values, infos, key);
        return true;
    }
    out = 0;
    return false;
}

inline bool get_sparse_lookup_value(const mp_int* values, const unsigned char* exists, uint64_t n, uint64_t key, mp_int& out) noexcept
{
    if (exists[key] && (key > 0) && (key < n))
    {
        out = values[key];
        return true;
    }
    out = 0;
    return false;
}

inline bool get_sorted_lookup_value(const uint64_t* keys, const mp_int* values, uint64_t n, uint64_t key, mp_int& out) noexcept
{
    uint64_t l = 0;
    uint64_t r = n - 1;
    while (l <= r)
    {
        uint64_t curr = l + (r - l) / 2;
        if (keys[curr] == key)
        {
            out = values[curr];
            return true;
        }
        if (keys[curr] < key)
        {
            l = curr + 1;
        }
        else
        {
            r = curr - 1;
        }
    }
    set(out, 0);
    return false;
}

inline void step_lookup(const int64_t* values, uint64_t n, uint64_t offset, mp_int& iq, int64_t ia_in, temps_span temps)
{
    uint64_t ia_u64;
    memcpy(&ia_u64, &ia_in, sizeof(uint64_t));
    uint64_t ia = ia_u64 - offset;
    int64_t result;
    get_lookup_value(values, n, ia, result);
    set(iq, result);
}

inline void step_lookup(const uint64_t* values, const mp_int_info* infos, uint64_t n, uint64_t offset, int64_t& iq, int64_t ia_in,
                        temps_span temps)
{
    uint64_t ia_u64;
    memcpy(&ia_u64, &ia_in, sizeof(uint64_t));
    uint64_t ia = ia_u64 - offset;
    csl::mp_int& result = temps.at(0);
    get_lookup_value(values, infos, n, ia, result);
    set(iq, result);
}

inline void step_lookup(const uint64_t* values, const mp_int_info* infos, uint64_t n, uint64_t offset, mp_int& iq, int64_t ia_in,
                        temps_span temps)
{
    uint64_t ia_u64;
    memcpy(&ia_u64, &ia_in, sizeof(uint64_t));
    uint64_t ia = ia_u64 - offset;
    csl::mp_int& result = temps.at(0);
    get_lookup_value(values, infos, n, ia, result);
    set(iq, result);
}

inline void step_lookup_with_valid(const int64_t* values, uint64_t n, uint64_t offset, int64_t& iq, mp_int& ivalid, int64_t ia_in,
                                   temps_span temps)
{
    uint64_t ia_u64;
    memcpy(&ia_u64, &ia_in, sizeof(uint64_t));
    uint64_t ia = ia_u64 - offset;
    int64_t result;
    bool valid = get_lookup_value(values, n, ia, result);
    set(iq, result);
    set(ivalid, valid ? 1 : 0);
}

inline void step_lookup_with_valid(const int64_t* values, uint64_t n, uint64_t offset, mp_int& iq, int64_t& ivalid, int64_t ia_in,
                                   temps_span temps)
{
    uint64_t ia_u64;
    memcpy(&ia_u64, &ia_in, sizeof(uint64_t));
    uint64_t ia = ia_u64 - offset;
    int64_t result;
    bool valid = get_lookup_value(values, n, ia, result);
    set(iq, result);
    set(ivalid, valid ? 1 : 0);
}

inline void step_lookup_with_valid(const int64_t* values, uint64_t n, uint64_t offset, mp_int& iq, mp_int& ivalid, int64_t ia_in,
                                   temps_span temps)
{
    uint64_t ia_u64;
    memcpy(&ia_u64, &ia_in, sizeof(uint64_t));
    uint64_t ia = ia_u64 - offset;
    int64_t result;
    bool valid = get_lookup_value(values, n, ia, result);
    set(iq, result);
    set(ivalid, valid ? 1 : 0);
}

inline void step_lookup_with_valid(const uint64_t* values, const mp_int_info* infos, uint64_t n, uint64_t offset, int64_t& iq,
                                   int64_t& ivalid, int64_t ia_in, temps_span temps)
{
    uint64_t ia_u64;
    memcpy(&ia_u64, &ia_in, sizeof(uint64_t));
    uint64_t ia = ia_u64 - offset;
    csl::mp_int& result = temps.at(0);
    bool valid = get_lookup_value(values, infos, n, ia, result);
    set(iq, result);
    set(ivalid, valid ? 1 : 0);
}

inline void step_lookup_with_valid(const uint64_t* values, const mp_int_info* infos, uint64_t n, uint64_t offset, int64_t& iq,
                                   mp_int& ivalid, int64_t ia_in, temps_span temps)
{
    uint64_t ia_u64;
    memcpy(&ia_u64, &ia_in, sizeof(uint64_t));
    uint64_t ia = ia_u64 - offset;
    csl::mp_int& result = temps.at(0);
    bool valid = get_lookup_value(values, infos, n, ia, result);
    set(iq, result);
    set(ivalid, valid ? 1 : 0);
}

inline void step_lookup_with_valid(const uint64_t* values, const mp_int_info* infos, uint64_t n, uint64_t offset, mp_int& iq,
                                   int64_t& ivalid, int64_t ia_in, temps_span temps)
{
    uint64_t ia_u64;
    memcpy(&ia_u64, &ia_in, sizeof(uint64_t));
    uint64_t ia = ia_u64 - offset;
    csl::mp_int& result = temps.at(0);
    bool valid = get_lookup_value(values, infos, n, ia, result);
    set(iq, result);
    set(ivalid, valid ? 1 : 0);
}

inline void step_lookup_with_valid(const uint64_t* values, const mp_int_info* infos, uint64_t n, uint64_t offset, mp_int& iq,
                                   mp_int& ivalid, int64_t ia_in, temps_span temps)
{
    uint64_t ia_u64;
    memcpy(&ia_u64, &ia_in, sizeof(uint64_t));
    uint64_t ia = ia_u64 - offset;
    csl::mp_int& result = temps.at(0);
    bool valid = get_lookup_value(values, infos, n, ia, result);
    set(iq, result);
    set(ivalid, valid ? 1 : 0);
}

inline void step_decode(mp_int& iq0, const mp_int& ia, const mp_int& ib, int32_t low, int32_t high, int32_t decode,
                        temps_span temps) noexcept
{
    mp_int& tmp_a = temps.at(0);
    tmp_a = ia;
    ld_exp(tmp_a, tmp_a, -low);
    mask_lower(tmp_a, tmp_a, static_cast<int64_t>(high) - low + 1);
    bool hit = (tmp_a == decode);
    if (hit)
    {
        set(iq0, ib);
    }
    else
    {
        set(iq0, int64_t(0));
    }
}

inline void step_decode(mp_int& iq0, mp_int& iq1, const mp_int& ia, const mp_int& ib, int32_t low, int32_t high, int32_t decode,
                        temps_span temps) noexcept
{
    mp_int& tmp_a = temps.at(0);
    tmp_a = ia;
    ld_exp(tmp_a, tmp_a, -low);
    mask_lower(tmp_a, tmp_a, static_cast<int64_t>(high) - low + 1);
    bool hit = (tmp_a == decode);
    if (hit)
    {
        set(iq0, ib);
        set(iq1, int64_t(1));
    }
    else
    {
        set(iq0, int64_t(0));
        set(iq1, int64_t(0));
    }
}

#ifdef CSL_USE_MPFR

inline void step_fp_acc(mp_int& iacc, int64_t control, int64_t& iq, int64_t ix) noexcept
{
    fp32 x(ix);

    mp_float mp_acc(24);
    if (control == 0)
    {
        // acc = x
        mpfr_set_d(mp_acc.get(), static_cast<double>(x.get()), GMP_RNDN);
    }
    else
    {
        mp_float mp_x(24);
        mpfr_set_d(mp_x.get(), static_cast<double>(x.get()), GMP_RNDN);

        fp32 acc(iacc);
        mpfr_set_d(mp_acc.get(), static_cast<double>(acc.get()), GMP_RNDN);

        // acc = acc + x
        mpfr_add(mp_acc.get(), mp_acc.get(), mp_x.get(), GMP_RNDN);
    }
    float v = static_cast<float>(mpfr_get_d(mp_acc.get(), GMP_RNDN));
    int v_int;
    memcpy(&v_int, &v, sizeof(v));
    iacc = v_int;

    // set output
    iq = to_i64(iacc);
}

#endif // CSL_USE_MPFR
#ifdef CSL_USE_MPFR

inline void step_fp_mult_acc(mp_int& iacc, int64_t control, int64_t& iq, int64_t ix, int64_t iy) noexcept
{
    fp32 x(ix);
    fp32 y(iy);

    // x = x*y
    mp_float mp_x(24);
    mp_float mp_y(24);
    mp_float mp_acc(24);
    mpfr_set_d(mp_x.get(), static_cast<double>(x.get()), GMP_RNDN);
    mpfr_set_d(mp_y.get(), static_cast<double>(y.get()), GMP_RNDN);
    mpfr_mul(mp_x.get(), mp_x.get(), mp_y.get(), GMP_RNDN);

    if (control == 0)
    {
        // acc = (x*y)
        mpfr_set(mp_acc.get(), mp_x.get(), GMP_RNDN);
    }
    else
    {
        fp32 acc(iacc);
        mpfr_set_d(mp_acc.get(), static_cast<double>(acc.get()), GMP_RNDN);
        // acc = acc + (x*y)
        mpfr_add(mp_acc.get(), mp_acc.get(), mp_x.get(), GMP_RNDN);
    }
    float v = static_cast<float>(mpfr_get_d(mp_acc.get(), GMP_RNDN));
    int v_int;
    memcpy(&v_int, &v, sizeof(v));
    iacc = v_int;

    // set output
    iq = to_i64(iacc);
}

#endif // CSL_USE_MPFR
inline void step_loadable_counter(int64_t& state_counter, int64_t& state_mod, int64_t& state_inc, mp_int& iq, int64_t ienable,
                                  int64_t iload, int64_t load_count, int64_t load_mod, int64_t load_inc) noexcept
{
    const bool enable = ienable != 0;
    const bool load = iload != 0;

    if (load)
    {
        state_counter = load_count;
        state_mod = load_mod;
        state_inc = load_inc;
        if (state_inc < 0)
        {
            state_inc += load_mod;
        }
    }
    else if (enable)
    {
        // Modulo zero is undefined - just like divide by zero. The hardware will count as
        // if there is no modulo however, and we will do the same. Currently we don't issue
        // a warning.
        state_counter += state_inc;
        if (state_mod != 0)
        {
            state_counter %= state_mod;
        }
    }
    set(iq, state_counter);
}

inline void step_loadable_counter(mp_int& state_counter, mp_int& state_mod, mp_int& state_inc, int64_t& iq, const mp_int& ienable,
                                  const mp_int& iload, const mp_int& load_count, const mp_int& load_mod,
                                  const mp_int& load_inc) noexcept
{
    const bool enable = ienable != 0;
    const bool load = iload != 0;

    if (load)
    {
        state_counter = load_count;
        state_mod = load_mod;
        state_inc = load_inc;
        if (state_inc < 0)
        {
            state_inc += load_mod;
        }
    }
    else if (enable)
    {
        // Modulo zero is undefined - just like divide by zero. The hardware will count as
        // if there is no modulo however, and we will do the same. Currently we don't issue
        // a warning.
        state_counter += state_inc;
        if (state_mod != 0)
        {
            state_counter %= state_mod;
        }
    }
    set(iq, state_counter);
}

inline void step_loadable_counter(mp_int& state_counter, mp_int& state_mod, mp_int& state_inc, mp_int& iq, const mp_int& ienable,
                                  const mp_int& iload, const mp_int& load_count, const mp_int& load_mod,
                                  const mp_int& load_inc) noexcept
{
    const bool enable = ienable != 0;
    const bool load = iload != 0;

    if (load)
    {
        state_counter = load_count;
        state_mod = load_mod;
        state_inc = load_inc;
        if (state_inc < 0)
        {
            state_inc += load_mod;
        }
    }
    else if (enable)
    {
        // Modulo zero is undefined - just like divide by zero. The hardware will count as
        // if there is no modulo however, and we will do the same. Currently we don't issue
        // a warning.
        state_counter += state_inc;
        if (state_mod != 0)
        {
            state_counter %= state_mod;
        }
    }
    set(iq, state_counter);
}

inline void step_cma_add(const mp_int* const prod_arr, mp_int* const sums_arr, const cma_add_params& params, int64_t sub_ctrl,
                         int64_t neg_ctrl, mp_int& region_sum) noexcept
{
    bool sub_ctrl_value = test_bit(0, sub_ctrl);
    bool neg_ctrl_value = test_bit(0, neg_ctrl);

    for (int k = 0, i = 0; i < params.systolic_region_count; ++i)
    {
        const int mults_in_region = csl::min(params.systolic_region_size, params.n_mults - k);
        region_sum = 0;

        if (sub_ctrl_value)
        {
            region_sum = prod_arr[params.pipeline_depth + 1];
            region_sum -= prod_arr[0];
            k += 2;
        }
        else
        {
            for (int j = 0; j < mults_in_region; ++j)
            {
                region_sum += prod_arr[k * (params.pipeline_depth + 1)];
                k++;
            }
        }

        const int i1 = i + 1;
        if (i1 == params.systolic_region_count)
        {
            sums_arr[i] = region_sum;
        }
        else
        {
            if (neg_ctrl_value)
            {
                sums_arr[i] = sums_arr[i1];
                sums_arr[i] -= region_sum;
            }
            else
            {
                sums_arr[i] = sums_arr[i1];
                sums_arr[i] += region_sum;
            }
        }
    }
}

inline void step_fifo(mp_int* store, const fifo_params& params, mp_int& data, int64_t write_en, int64_t read_en,
                      int64_t flush) noexcept
{
    const int base_index = params.base_index;
    const int read_ptr_index = base_index + 4 + params.depth;
    const int write_ptr_index = base_index + 5 + params.depth;
    const int user_sclr = params.user_sclr;

    int64_t read_ptr_64, write_ptr_64;
    set(read_ptr_64, store[read_ptr_index]);
    set(write_ptr_64, store[write_ptr_index]);
    int read_ptr = static_cast<int>(read_ptr_64);
    int write_ptr = static_cast<int>(write_ptr_64);

    const int twice_depth = 2 * params.depth;
    // (mod 2n) so that full and empty queue states can be disambiguated
    // in both of these cases
    //               mod(write_ptr - read_ptr,   m_depth) == 0
    // but if empty, mod(write_ptr - read_ptr, 2*m_depth) == 0
    // and if full,  mod(write_ptr - read_ptr, 2*m_depth) == m_depth

    int count = ((write_ptr - read_ptr) % twice_depth + twice_depth) % twice_depth;

    // update state independent of input signals
    int write_ptr_delayed;
    int count_delayed;
    bool clearing = user_sclr && flush != 0;

    // update state according to input operands
    if (clearing)
    {
        // Operation during SCLR

        // Flush input is high - clear FIFO state
        int frame_size = 7 + params.depth + params.write_latency; // room for FIFO + circular Buffer + state variables

        for (int i = 0; i < frame_size; ++i)
        {
            // delayed write pipeline -> set zero across the whole frame
            store[base_index + i] = 0;
        }

        // Flush the FIFO by resetting the address counters
        read_ptr = base_index; // Pointers to the start
        write_ptr = base_index;
        write_ptr_delayed = base_index;
        count_delayed = 0; // Counters to zero
        count = 0;

        store[read_ptr_index] = read_ptr; // Reset read pointer
        store[write_ptr_index] = write_ptr; // Reset write pointer
    }
    else
    {
        // Normal operation

        // update state independent of input signals (other than sclr)
        for (int i = params.write_latency; i > 0; --i)
        {
            // delayed write pipeline
            store[write_ptr_index + i] = store[write_ptr_index + i - 1];
        }

        if (write_en != 0)
        {
            if (count == params.depth)
            {
                // Warn on write when full
                warning("FIFO_WRITE_WHILE_FULL");
            }
            if (count < params.depth)
            {
                // enqueue (write enable is high) in circular buffer
                store[base_index + 4 + (write_ptr % params.depth)] = data;
                write_ptr = (write_ptr + 1) % twice_depth;
                ++count;
                store[write_ptr_index] = write_ptr;
            }
        }

        int64_t write_ptr_delayed_64;
        set(write_ptr_delayed_64, store[write_ptr_index + params.write_latency]);
        write_ptr_delayed = static_cast<int>(write_ptr_delayed_64);
        count_delayed = ((write_ptr_delayed - read_ptr) % twice_depth + twice_depth) % twice_depth;

        const int maxCountIdx = base_index + 6 + params.depth + params.write_latency;
        int64_t maxCount;
        set(maxCount, store[maxCountIdx]);
        if (count > maxCount)
        {
            store[maxCountIdx] = count;
        }

        if (read_en != 0)
        {
            // dequeue (read enable is high) from circular buffer
            if (count_delayed > 0)
            {
                read_ptr = (read_ptr + 1) % twice_depth;
                --count_delayed;
                --count;
                store[read_ptr_index] = read_ptr;
            }
            else
            {
                // Warn on read when empty
                warning("FIFO_READACK_VALID_LOW");
            }
        }
    }

    // output register update
    bool valid = count_delayed > 0;
    store[base_index + 0] = valid ? 1 : 0;
    store[base_index + 1] = (count >= params.fill_threshold) ? 1 : 0;
    store[base_index + 2] = (count >= params.full_threshold) ? 1 : 0;

    if (count_delayed > 0)
    {
        // update data output register
        store[base_index + 3] = store[base_index + 4 + (read_ptr % params.depth)];
    }
}

#endif // CSL_USE_GMP

/** Generated steps end */

} // namespace csl
