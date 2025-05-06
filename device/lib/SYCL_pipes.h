#pragma once

#include "SYCL_ckks_sym.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Define pipes for communication between IFFT and ScaleAndConvert kernels
constexpr size_t PIPE_CAPACITY = 8192; // Adjust based on your needs 4096

// Pipe from IFFT to ScaleAndConvert kernel for transformed values
using IFFTToScaleAndReducePipe = 
    sycl::ext::intel::pipe<class IFFTToScaleAndReducePipeID, std::complex<double>, PIPE_CAPACITY>;

// Pipe to pass error samples from IFFT to ScaleAndConvert kernel
using IFFTErrorToScaleAndReducePipe =
    sycl::ext::intel::pipe<class IFFTErrorToScaleAndReducePipeID, int8_t, PIPE_CAPACITY>;

// Pipe from ScaleAndConvertKernel to ReduceSetPTEKernel for plaintext+error values
//using ScaleToReducePipe =
//    sycl::ext::intel::pipe<class ScaleToReducePipeID, int64_t, PIPE_CAPACITY>;

// Pipe from ScaleAndReduceKernel to NTTKernel_1
//using ScaleReduceToNTT1Pipe =
//    sycl::ext::intel::pipe<class ScaleReduceToNTT1PipeID, uint32_t, PIPE_CAPACITY>;
