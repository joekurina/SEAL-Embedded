#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "the_nwc_4k_ntt_sycl.hpp"
#include <cstdint>

//#undef CSL_PACKED
//#define CSL_PACKED( struct_def ) struct_def __attribute__((__packed__, aligned(32)))

// Input data structure for NTT RTL pipeline
typedef struct
{
    int32_t port_x_in_0;
    int32_t port_x_in_1;
    int32_t port_x_in_2;
    int32_t port_x_in_3;
} NTT_RTL_Input_Data;

// Output data structure for NTT RTL pipeline
typedef struct
{
    int32_t port_out_q_0;
    int32_t port_out_q_1;
    int32_t port_out_q_2;
    int32_t port_out_q_3;
} NTT_RTL_Output_Data;

//static_assert(sizeof(NTT_RTL_Input_Data) == 32);  // 4 * sizeof(int32_t) + alignment padding
//static_assert(sizeof(NTT_RTL_Output_Data) == 32); // 4 * sizeof(int32_t) + alignment padding

// NTT RTL processing capacity - 4K points = 1024 structs of 4 elements each
constexpr size_t NTT_RTL_CAPACITY = 1024;

// SYCL Pipe class definitions for NTT RTL kernels
// These pipes connect the RTL pipeline

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

// Utility function to map SEAL-Embedded modulus values to RTL modulus selector
inline uint8_t get_rtl_modulus_selector(uint32_t mod_value) {
    switch(mod_value) {
        case 134012929:  return 0;  // root = 7470
        case 134111233:  return 1;  // root = 3856
        case 134176769:  return 2;  // root = 24149
        case 1053818881: return 3;  // root = 503422
        case 1054015489: return 4;  // root = 16768
        case 1054212097: return 5;  // root = 7305
        default:          return 0;  // fallback to first modulus
    }
}