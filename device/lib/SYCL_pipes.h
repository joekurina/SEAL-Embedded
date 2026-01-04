#pragma once

#include "SYCL_data_types.h"
#include "SYCL_common.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sycl_ckks {

template <int P>
struct PipeSet
{
    struct EntryToIFFTPipeID {};
    struct EntryToScaleReducePipeID {};
    struct EntryToNTTAPipeID {};
    struct EntryToPolyMultNegPipeID {};
    struct IFFTToScaleReducePipeID {};
    struct ScaleReduceToNTTBPipeID {};
    struct NTTAToPolyMultNegPipeID {};
    struct NTTBToPolyAddPipeID {};
    struct NTTAToExitPipeID {};
    struct NTTBToExitPipeID {};
    struct PolyAddToExitPipeID {};
    struct NTTAInputPipeID {};
    struct NTTAModSelectorPipeID {};
    struct NTTAOutputPipeID {};
    struct NTTBInputPipeID {};
    struct NTTBModSelectorPipeID {};
    struct NTTBOutputPipeID {};

    using EntryToIFFTPipe = sycl::ext::intel::pipe<EntryToIFFTPipeID, encoding_block, PIPE_CAPACITY>;
    using EntryToScaleReducePipe = sycl::ext::intel::pipe<EntryToScaleReducePipeID, i8x4, PIPE_CAPACITY>;
    using EntryToNTTAPipe = sycl::ext::intel::pipe<EntryToNTTAPipeID, u32x4, PIPE_CAPACITY>;
    using EntryToPolyMultNegPipe = sycl::ext::intel::pipe<EntryToPolyMultNegPipeID, u32x4, PIPE_CAPACITY>;

    using IFFTToScaleReducePipe = sycl::ext::intel::pipe<IFFTToScaleReducePipeID, encoding_block, PIPE_CAPACITY>;
    using ScaleReduceToNTTBPipe = sycl::ext::intel::pipe<ScaleReduceToNTTBPipeID, u32x4, PIPE_CAPACITY>;
    
    using NTTAToPolyMultNegPipe = sycl::ext::intel::pipe<NTTAToPolyMultNegPipeID, u32x4, PIPE_CAPACITY>;
    using NTTBToPolyAddPipe = sycl::ext::intel::pipe<NTTBToPolyAddPipeID, u32x4, PIPE_CAPACITY>;

    using NTTAToExitPipe = sycl::ext::intel::pipe<NTTAToExitPipeID, u32x4, PIPE_CAPACITY>;
    using NTTBToExitPipe = sycl::ext::intel::pipe<NTTBToExitPipeID, u32x4, PIPE_CAPACITY>;
    using PolyAddToExitPipe = sycl::ext::intel::pipe<PolyAddToExitPipeID, u32x4, PIPE_CAPACITY>;

    struct NTTRTLInputData {
        int32_t x0;
        int32_t x1;
        int32_t x2;
        int32_t x3;
    };

    struct NTTRTLOutputData {
        int32_t q0;
        int32_t q1;
        int32_t q2;
        int32_t q3;
    };

    using NTTAInputPipe = sycl::ext::intel::pipe<NTTAInputPipeID, NTTRTLInputData, PIPE_CAPACITY>;
    using NTTAModSelectorPipe = sycl::ext::intel::pipe<NTTAModSelectorPipeID, uint8_t, PIPE_CAPACITY>;
    using NTTAOutputPipe = sycl::ext::intel::pipe<NTTAOutputPipeID, NTTRTLOutputData, PIPE_CAPACITY>;
    using NTTBInputPipe = sycl::ext::intel::pipe<NTTBInputPipeID, NTTRTLInputData, PIPE_CAPACITY>;
    using NTTBModSelectorPipe = sycl::ext::intel::pipe<NTTBModSelectorPipeID, uint8_t, PIPE_CAPACITY>;
    using NTTBOutputPipe = sycl::ext::intel::pipe<NTTBOutputPipeID, NTTRTLOutputData, PIPE_CAPACITY>;
};

}
