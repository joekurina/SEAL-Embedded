#pragma once

#include "SYCL_data_types.h"
#include "SYCL_common.h"
#include "pipe_utils.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sycl_ckks {

// Pre-twist pipeline: Entry -> PreTwist -> IFFT
struct SharedToPreTwistPipeId {};
using SharedToPreTwistPipe = sycl::ext::intel::pipe<SharedToPreTwistPipeId, encoding_block, PIPE_CAPACITY>;

struct PreTwistToIFFTPipeId {};
using PreTwistToIFFTPipe = sycl::ext::intel::pipe<PreTwistToIFFTPipeId, encoding_block, PIPE_CAPACITY>;

// Legacy pipe (kept for compatibility, but no longer used in main pipeline)
struct SharedToIFFTPipeId {};
using SharedToIFFTPipe = sycl::ext::intel::pipe<SharedToIFFTPipeId, encoding_block, PIPE_CAPACITY>;

struct IFFTToScaleReducePipeArrayId {};
using IFFTToScaleReducePipes = fpga_tools::PipeArray<
    IFFTToScaleReducePipeArrayId,
    encoding_block,
    PIPE_CAPACITY,
    NUM_MODULI
>;

struct ErrorToScaleReducePipeArrayId {};
using ErrorToScaleReducePipes = fpga_tools::PipeArray<
    ErrorToScaleReducePipeArrayId,
    i8x4,
    PIPE_CAPACITY,
    NUM_MODULI
>;

template <int P>
struct PipeSet
{
    struct EntryToNTTAPipeID {};
    struct EntryToPolyMultNegPipeID {};
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

    struct SingleIFFTInputPipeID {};
    struct SingleIFFTOutputPipeID {};
    struct SingleErrorToScaleReducePipeID {};

    using SingleIFFTInputPipe = sycl::ext::intel::pipe<SingleIFFTInputPipeID, encoding_block, PIPE_CAPACITY>;
    using SingleIFFTOutputPipe = sycl::ext::intel::pipe<SingleIFFTOutputPipeID, encoding_block, PIPE_CAPACITY>;
    using SingleErrorToScaleReducePipe = sycl::ext::intel::pipe<SingleErrorToScaleReducePipeID, i8x4, PIPE_CAPACITY>;

    using EntryToNTTAPipe = sycl::ext::intel::pipe<EntryToNTTAPipeID, u32x4, PIPE_CAPACITY>;
    using EntryToPolyMultNegPipe = sycl::ext::intel::pipe<EntryToPolyMultNegPipeID, u32x4, PIPE_CAPACITY>;

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
