#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "the_nwc_4k_ntt_sycl.hpp"
#include <cstdint>

// Data structures for NTT RTL pipes
// These structures package data for efficient transfer between kernels
// Each struct contains 4 data elements to match RTL interface expectations

#undef CSL_PACKED
#define CSL_PACKED( struct_def ) struct_def __attribute__((__packed__, aligned(32)))

// Input data structure for NTT RTL pipeline
// Matches the format expected by the_nwc_4k_ntt_input_t but optimized for pipes
CSL_PACKED(typedef struct
{
    int32_t port_x_in_0;
    int32_t port_x_in_1;
    int32_t port_x_in_2;
    int32_t port_x_in_3;
}) NTT_RTL_Input_Data;

// Output data structure for NTT RTL pipeline
// Matches the format provided by the_nwc_4k_ntt_output_t but optimized for pipes
CSL_PACKED(typedef struct
{
    int32_t port_out_q_0;
    int32_t port_out_q_1;
    int32_t port_out_q_2;
    int32_t port_out_q_3;
}) NTT_RTL_Output_Data;

// Ensure proper struct sizes for efficient pipe transfers
static_assert(sizeof(NTT_RTL_Input_Data) == 32);  // 4 * sizeof(int32_t) + alignment padding
static_assert(sizeof(NTT_RTL_Output_Data) == 32); // 4 * sizeof(int32_t) + alignment padding

// NTT RTL processing capacity - 4K points = 1024 structs of 4 elements each
constexpr size_t NTT_RTL_CAPACITY = 1024;

// SYCL Pipe class definitions for NTT RTL kernels
// These pipes connect the 6-stage RTL pipeline

// Forward declarations for pipe name classes
class NTTAInputPipeName;
class NTTAOutputPipeName;
class NTTBInputPipeName;
class NTTBOutputPipeName;

// Internal NTT A Pipeline Pipes (between the 3 NTT A stages)
using NTTAInputPipe = sycl::ext::intel::pipe<NTTAInputPipeName, NTT_RTL_Input_Data, NTT_RTL_CAPACITY>;
using NTTAOutputPipe = sycl::ext::intel::pipe<NTTAOutputPipeName, NTT_RTL_Output_Data, NTT_RTL_CAPACITY>;

// Internal NTT B Pipeline Pipes (between the 3 NTT B stages)
using NTTBInputPipe = sycl::ext::intel::pipe<NTTBInputPipeName, NTT_RTL_Input_Data, NTT_RTL_CAPACITY>;
using NTTBOutputPipe = sycl::ext::intel::pipe<NTTBOutputPipeName, NTT_RTL_Output_Data, NTT_RTL_CAPACITY>;

// Note: Integration with existing pipeline pipes:
// - NTT A input reads from buffer (secret key data)
// - NTT A output connects to existing NTTToPolyMultNegPipe
// - NTT B input connects to existing ScaleReduceToNTTBPipe
// - NTT B output connects to existing NTTToAddModPipe

// Utility function to map SEAL-Embedded modulus values to RTL modulus selector
inline uint8_t get_rtl_modulus_selector(uint32_t mod_value) {
    switch(mod_value) {
        case 134012929u:  return 0;  // root = 7470
        case 134111233u:  return 1;  // root = 3856
        case 134176769u:  return 2;  // root = 24149
        case 1053818881u: return 3;  // root = 503422
        case 1054015489u: return 4;  // root = 16768
        case 1054212097u: return 5;  // root = 7305
        default:          return 0;  // fallback to first modulus
    }
}

// Utility function to convert uint32_t array to NTT_RTL_Input_Data array
// Packs 4 consecutive uint32_t values into one struct
inline void pack_ntt_input_data(const uint32_t* src, NTT_RTL_Input_Data* dst, size_t num_structs) {
    for (size_t i = 0; i < num_structs; ++i) {
        dst[i].port_x_in_0 = static_cast<int32_t>(src[i * 4 + 0]);
        dst[i].port_x_in_1 = static_cast<int32_t>(src[i * 4 + 1]);
        dst[i].port_x_in_2 = static_cast<int32_t>(src[i * 4 + 2]);
        dst[i].port_x_in_3 = static_cast<int32_t>(src[i * 4 + 3]);
    }
}

// Utility function to convert NTT_RTL_Output_Data array to uint32_t array
// Unpacks structs back into individual uint32_t values
inline void unpack_ntt_output_data(const NTT_RTL_Output_Data* src, uint32_t* dst, size_t num_structs) {
    for (size_t i = 0; i < num_structs; ++i) {
        dst[i * 4 + 0] = static_cast<uint32_t>(src[i].port_out_q_0);
        dst[i * 4 + 1] = static_cast<uint32_t>(src[i].port_out_q_1);
        dst[i * 4 + 2] = static_cast<uint32_t>(src[i].port_out_q_2);
        dst[i * 4 + 3] = static_cast<uint32_t>(src[i].port_out_q_3);
    }
}