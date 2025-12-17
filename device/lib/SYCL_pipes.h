#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_lanes.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Define pipes for communication between IFFT and ScaleAndConvert kernels
// Capacity is expressed in number of 4-lane structs. For n=4096 scalars, this is 1024 structs.
constexpr size_t PIPE_CAPACITY = 1024;

// Pipe from IFFT to ScaleAndConvert kernel for transformed values
using IFFTToScaleAndReducePipe = 
    sycl::ext::intel::pipe<class IFFTToScaleAndReducePipeID, encoding_buffer_input, PIPE_CAPACITY>;

// Pipe to pass error samples from IFFT to ScaleAndConvert kernel
using IFFTErrorToScaleAndReducePipe =
    sycl::ext::intel::pipe<class IFFTErrorToScaleAndReducePipeID, i8x4_input, PIPE_CAPACITY>;

// Pipe from ScaleAndConvertKernel to ReduceSetPTEKernel for plaintext+error values
using ScaleToReducePipe =
    sycl::ext::intel::pipe<class ScaleToReducePipeID, i64x4_input, PIPE_CAPACITY>;

// Pipe from ScaleAndConvertKernel to NTTKernel_1 for plaintext+error values
using ScaleReduceToNTTBPipe =
    sycl::ext::intel::pipe<class ScaleReduceToNTTBPipeID, u32x4_input, PIPE_CAPACITY>;

// Pipe from NTTKernel_B to AddModKernel for NTT(PTE+error)
using NTTToAddModPipe =
    sycl::ext::intel::pipe<class NTTToAddModPipeID, u32x4_input, PIPE_CAPACITY>;

// Pipe from NTTKernel_A to PolyMultNegNTTKernel for NTT(s)
using NTTToPolyMultNegPipe =
    sycl::ext::intel::pipe<class NTTToPolyMultNegPipeID, u32x4_input, PIPE_CAPACITY>;

// Pipe from PolyMultNegNTTKernel to PolyAddModKernel for -(NTT(s)*c1)
using PolyMultNegToPolyAddModPipe =
    sycl::ext::intel::pipe<class PolyMultNegToPolyAddModPipeID, u32x4_input, PIPE_CAPACITY>;