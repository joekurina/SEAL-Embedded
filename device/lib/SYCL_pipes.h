#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_lanes.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Define pipes for communication between IFFT and ScaleAndConvert kernels
// Capacity is expressed in number of 4-lane structs. For n=4096 scalars, this is 1024 structs.
constexpr size_t PIPE_CAPACITY = 1024;

// Per-pipeline pipe namespace (templated for P = 0,1,2 ...).
template <int P>
struct CKKS_PIPE_SET
{
    // Tag types must be unique per pipeline instance.
    struct IFFTToScaleAndReducePipeID { static constexpr int id = P; };
    struct IFFTErrorToScaleAndReducePipeID { static constexpr int id = P; };
    struct ScaleToReducePipeID { static constexpr int id = P; };
    struct ScaleReduceToNTTBPipeID { static constexpr int id = P; };
    struct NTTToAddModPipeID { static constexpr int id = P; };
    struct NTTToPolyMultNegPipeID { static constexpr int id = P; };
    struct PolyMultNegToPolyAddModPipeID { static constexpr int id = P; };

    using IFFTToScaleAndReducePipe =
        sycl::ext::intel::pipe<IFFTToScaleAndReducePipeID, encoding_buffer_input, PIPE_CAPACITY>;

    using IFFTErrorToScaleAndReducePipe =
        sycl::ext::intel::pipe<IFFTErrorToScaleAndReducePipeID, i8x4_input, PIPE_CAPACITY>;

    using ScaleToReducePipe =
        sycl::ext::intel::pipe<ScaleToReducePipeID, i64x4_input, PIPE_CAPACITY>;

    using ScaleReduceToNTTBPipe =
        sycl::ext::intel::pipe<ScaleReduceToNTTBPipeID, u32x4_input, PIPE_CAPACITY>;

    using NTTToAddModPipe =
        sycl::ext::intel::pipe<NTTToAddModPipeID, u32x4_input, PIPE_CAPACITY>;

    using NTTToPolyMultNegPipe =
        sycl::ext::intel::pipe<NTTToPolyMultNegPipeID, u32x4_input, PIPE_CAPACITY>;

    using PolyMultNegToPolyAddModPipe =
        sycl::ext::intel::pipe<PolyMultNegToPolyAddModPipeID, u32x4_input, PIPE_CAPACITY>;
};

// Backwards-compatible aliases for the default pipeline instance (P = 0).
using IFFTToScaleAndReducePipe = typename CKKS_PIPE_SET<0>::IFFTToScaleAndReducePipe;
using IFFTErrorToScaleAndReducePipe = typename CKKS_PIPE_SET<0>::IFFTErrorToScaleAndReducePipe;
using ScaleToReducePipe = typename CKKS_PIPE_SET<0>::ScaleToReducePipe;
using ScaleReduceToNTTBPipe = typename CKKS_PIPE_SET<0>::ScaleReduceToNTTBPipe;
using NTTToAddModPipe = typename CKKS_PIPE_SET<0>::NTTToAddModPipe;
using NTTToPolyMultNegPipe = typename CKKS_PIPE_SET<0>::NTTToPolyMultNegPipe;
using PolyMultNegToPolyAddModPipe = typename CKKS_PIPE_SET<0>::PolyMultNegToPolyAddModPipe;