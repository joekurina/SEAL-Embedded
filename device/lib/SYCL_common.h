#pragma once

#include <cstdint>
#include <cstddef>

namespace sycl_ckks {

constexpr size_t POLY_N = 4096;
constexpr size_t POLY_LOGN = 12;
constexpr size_t LANES = 4;
constexpr size_t NUM_BLOCKS = POLY_N / LANES;
constexpr size_t PIPE_CAPACITY = NUM_BLOCKS;
constexpr int MAX_PIPELINES = 3;

// Barrett reduction: val mod q using precomputed const_ratio (handles signed input)
inline uint32_t barrett_reduce_64(
    int64_t val,
    uint32_t mod_value,
    const uint32_t* const_ratio,
    bool negate_result = false)
{
    uint64_t coeff_abs = (val < 0) ? static_cast<uint64_t>(-val) : static_cast<uint64_t>(val);
    uint32_t sign_mask = static_cast<uint32_t>(val < 0);
    
    uint32_t coeff_abs_vec[2];
    coeff_abs_vec[0] = static_cast<uint32_t>(coeff_abs & 0xFFFFFFFF);
    coeff_abs_vec[1] = static_cast<uint32_t>((coeff_abs >> 32) & 0xFFFFFFFF);
    
    uint32_t right_hw;
    {
        uint64_t res_temp = static_cast<uint64_t>(coeff_abs_vec[0]) * 
                            static_cast<uint64_t>(const_ratio[0]);
        right_hw = static_cast<uint32_t>((res_temp >> 32) & 0xFFFFFFFF);
    }
    
    uint32_t middle_temp[2];
    {
        uint64_t res_temp = static_cast<uint64_t>(coeff_abs_vec[0]) * 
                            static_cast<uint64_t>(const_ratio[1]);
        middle_temp[0] = static_cast<uint32_t>(res_temp & 0xFFFFFFFF);
        middle_temp[1] = static_cast<uint32_t>((res_temp >> 32) & 0xFFFFFFFF);
    }
    
    uint32_t middle_lw = right_hw + middle_temp[0];
    uint32_t middle_lw_carry = static_cast<uint8_t>(middle_lw < right_hw);
    uint32_t middle_hw = middle_temp[1] + middle_lw_carry;
    
    uint32_t middle2_temp[2];
    {
        uint64_t res_temp = static_cast<uint64_t>(coeff_abs_vec[1]) * 
                            static_cast<uint64_t>(const_ratio[0]);
        middle2_temp[0] = static_cast<uint32_t>(res_temp & 0xFFFFFFFF);
        middle2_temp[1] = static_cast<uint32_t>((res_temp >> 32) & 0xFFFFFFFF);
    }
    
    uint32_t middle2_lw = middle_lw + middle2_temp[0];
    uint32_t middle2_lw_carry = static_cast<uint8_t>(middle2_lw < middle_lw);
    uint32_t middle2_hw = middle2_temp[1] + middle2_lw_carry;
    
    uint32_t tmp = coeff_abs_vec[1] * const_ratio[1] + middle_hw + middle2_hw;
    tmp = coeff_abs_vec[0] - tmp * mod_value;
    
    {
        int32_t is_ge_q = static_cast<int32_t>(tmp >= mod_value);
        uint32_t mask = static_cast<uint32_t>(-is_ge_q);
        tmp = tmp - (mod_value & mask);
    }
    
    uint32_t result = ((mod_value - tmp) & (-sign_mask)) + (tmp & (sign_mask - 1));
    
    if (negate_result) {
        int32_t non_zero = static_cast<int32_t>(result != 0);
        uint32_t neg_mask = static_cast<uint32_t>(-non_zero);
        result = (mod_value - result) & neg_mask;
    }
    
    return result;
}

// Barrett reduction for unsigned 64-bit product
inline uint32_t barrett_reduce_u64(
    uint64_t product,
    uint32_t mod_value,
    const uint32_t* const_ratio)
{
    uint32_t product_vec[2];
    product_vec[0] = static_cast<uint32_t>(product & 0xFFFFFFFFu);
    product_vec[1] = static_cast<uint32_t>((product >> 32) & 0xFFFFFFFFu);
    
    uint32_t right_hw;
    {
        uint64_t rt_temp = static_cast<uint64_t>(product_vec[0]) * 
                           static_cast<uint64_t>(const_ratio[0]);
        right_hw = static_cast<uint32_t>((rt_temp >> 32) & 0xFFFFFFFFu);
    }
    
    uint32_t middle_temp[2];
    {
        uint64_t mt_temp = static_cast<uint64_t>(product_vec[0]) * 
                           static_cast<uint64_t>(const_ratio[1]);
        middle_temp[0] = static_cast<uint32_t>(mt_temp & 0xFFFFFFFFu);
        middle_temp[1] = static_cast<uint32_t>((mt_temp >> 32) & 0xFFFFFFFFu);
    }
    
    uint32_t middle_lw = right_hw + middle_temp[0];
    uint32_t middle_lw_carry = static_cast<uint8_t>(middle_lw < right_hw);
    uint32_t middle_hw = middle_temp[1] + middle_lw_carry;
    
    uint32_t middle2_temp[2];
    {
        uint64_t mt2_temp = static_cast<uint64_t>(product_vec[1]) * 
                            static_cast<uint64_t>(const_ratio[0]);
        middle2_temp[0] = static_cast<uint32_t>(mt2_temp & 0xFFFFFFFFu);
        middle2_temp[1] = static_cast<uint32_t>((mt2_temp >> 32) & 0xFFFFFFFFu);
    }
    
    uint32_t middle2_lw = middle_lw + middle2_temp[0];
    uint32_t middle2_lw_carry = static_cast<uint8_t>(middle2_lw < middle_lw);
    uint32_t middle2_hw = middle2_temp[1] + middle2_lw_carry;
    
    uint32_t tmp = product_vec[1] * const_ratio[1] + middle_hw + middle2_hw;
    tmp = product_vec[0] - tmp * mod_value;
    
    int32_t is_ge_q = static_cast<int32_t>(tmp >= mod_value);
    uint32_t mask = static_cast<uint32_t>(-is_ge_q);
    return tmp - (mod_value & mask);
}

inline uint32_t mod_add(uint32_t a, uint32_t b, uint32_t mod_value)
{
    uint32_t sum = a + b;
    int32_t is_ge_q = static_cast<int32_t>(sum >= mod_value);
    uint32_t mask = static_cast<uint32_t>(-is_ge_q);
    return sum - (mod_value & mask);
}

inline uint32_t mod_neg(uint32_t a, uint32_t mod_value)
{
    int32_t non_zero = static_cast<int32_t>(a != 0);
    uint32_t mask = static_cast<uint32_t>(-non_zero);
    return (mod_value - a) & mask;
}

template <typename T, typename Struct4>
inline T lane_get(const Struct4& block, size_t lane)
{
    switch (lane) {
        case 0: return static_cast<T>(block.element0);
        case 1: return static_cast<T>(block.element1);
        case 2: return static_cast<T>(block.element2);
        case 3: return static_cast<T>(block.element3);
        default: return T{};
    }
}

template <typename T, typename Struct4>
inline void lane_set(Struct4& block, size_t lane, T value)
{
    switch (lane) {
        case 0: block.element0 = value; break;
        case 1: block.element1 = value; break;
        case 2: block.element2 = value; break;
        case 3: block.element3 = value; break;
    }
}

template <typename InStruct, typename OutStruct, typename Func>
inline void lane_transform(const InStruct& in, OutStruct& out, Func&& func)
{
    out.element0 = func(in.element0);
    out.element1 = func(in.element1);
    out.element2 = func(in.element2);
    out.element3 = func(in.element3);
}

template <typename InStruct1, typename InStruct2, typename OutStruct, typename Func>
inline void lane_transform2(const InStruct1& in1, const InStruct2& in2, 
                            OutStruct& out, Func&& func)
{
    out.element0 = func(in1.element0, in2.element0);
    out.element1 = func(in1.element1, in2.element1);
    out.element2 = func(in1.element2, in2.element2);
    out.element3 = func(in1.element3, in2.element3);
}

// Modulus selector for RTL NTT core (maps modulus value to hardware selector index)
inline uint8_t get_modulus_selector(uint32_t mod_value)
{
    switch (mod_value) {
        case 134012929u:  return 0;
        case 134111233u:  return 1;
        case 134176769u:  return 2;
        case 1053818881u: return 3;
        case 1054015489u: return 4;
        case 1054212097u: return 5;
        default:          return 0;
    }
}

// NTT primitive root lookup (cryptographic constants for polynomial ring)
inline uint32_t get_ntt_root(size_t n, uint32_t mod_value)
{
    if (n == 4096) {
        switch (mod_value) {
            case 134012929u:  return 7470;
            case 134111233u:  return 3856;
            case 134176769u:  return 24149;
            case 1053818881u: return 503422;
            case 1054015489u: return 16768;
            case 1054212097u: return 7305;
            default:          return 1;
        }
    }
    else if (n == 8192) {
        switch (mod_value) {
            case 1053818881u: return 374229;
            case 1054015489u: return 123363;
            case 1054212097u: return 79941;
            case 1055260673u: return 38869;
            case 1056178177u: return 162146;
            case 1056440321u: return 81884;
            default:          return 1;
        }
    }
    else if (n == 16384) {
        switch (mod_value) {
            case 1053818881u: return 13040;
            case 1054015489u: return 507;
            case 1054212097u: return 1595;
            case 1055260673u: return 68507;
            case 1056178177u: return 3073;
            case 1056440321u: return 6854;
            case 1058209793u: return 44467;
            case 1060175873u: return 16117;
            case 1060700161u: return 27607;
            case 1060765697u: return 222391;
            case 1061093377u: return 105471;
            case 1062469633u: return 310222;
            case 1062535169u: return 2005;
            default:          return 1;
        }
    }
    return 1;
}

}
