#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "the_nwc_4k_ntt_sycl.hpp"
#include <cstdint>


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

// NTT RTL processing capacity - 4K points = 1024 structs of 4 elements each
constexpr size_t NTT_RTL_CAPACITY = 1024;

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