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
// These pipes connect the RTL pipeline. We template on pipeline index P to allow
// multiple independent pipelines (P = 0,1,2, ...).

template <int P>
struct NTT_PIPE_SET
{
    struct NTTAInputPipeName { static constexpr int id = P; };
    struct NTTAModSelectorPipeName { static constexpr int id = P; };
    struct NTTAOutputPipeName { static constexpr int id = P; };
    struct NTTBInputPipeName { static constexpr int id = P; };
    struct NTTBModSelectorPipeName { static constexpr int id = P; };
    struct NTTBOutputPipeName { static constexpr int id = P; };

    // Internal NTT A Pipeline Pipes (between the 3 NTT A stages)
    using NTTAInputPipe = sycl::ext::intel::pipe<NTTAInputPipeName, NTT_RTL_Input_Data, NTT_RTL_CAPACITY>;
    using NTTAModSelectorPipe = sycl::ext::intel::pipe<NTTAModSelectorPipeName, uint8_t, NTT_RTL_CAPACITY>;
    using NTTAOutputPipe = sycl::ext::intel::pipe<NTTAOutputPipeName, NTT_RTL_Output_Data, NTT_RTL_CAPACITY>;

    // Internal NTT B Pipeline Pipes (between the 3 NTT B stages)
    using NTTBInputPipe = sycl::ext::intel::pipe<NTTBInputPipeName, NTT_RTL_Input_Data, NTT_RTL_CAPACITY>;
    using NTTBModSelectorPipe = sycl::ext::intel::pipe<NTTBModSelectorPipeName, uint8_t, NTT_RTL_CAPACITY>;
    using NTTBOutputPipe = sycl::ext::intel::pipe<NTTBOutputPipeName, NTT_RTL_Output_Data, NTT_RTL_CAPACITY>;
};

// Backwards-compatible aliases for the default pipeline instance (P = 0).
using NTTAInputPipe = typename NTT_PIPE_SET<0>::NTTAInputPipe;
using NTTAModSelectorPipe = typename NTT_PIPE_SET<0>::NTTAModSelectorPipe;
using NTTAOutputPipe = typename NTT_PIPE_SET<0>::NTTAOutputPipe;
using NTTBInputPipe = typename NTT_PIPE_SET<0>::NTTBInputPipe;
using NTTBModSelectorPipe = typename NTT_PIPE_SET<0>::NTTBModSelectorPipe;
using NTTBOutputPipe = typename NTT_PIPE_SET<0>::NTTBOutputPipe;