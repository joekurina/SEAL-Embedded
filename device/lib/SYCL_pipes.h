#pragma once

#include "SYCL_ckks_sym.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Define pipes for communication between IFFT and ScaleAndConvert kernels
constexpr size_t PIPE_CAPACITY = 4096; // Adjust based on your needs

// Pipe from IFFT to ScaleAndConvert kernel for transformed values
using IFFTToScaleAndConvertPipe = 
    sycl::ext::intel::pipe<class IFFTToScaleAndConvertPipeID, std::complex<double>, PIPE_CAPACITY>;

// Pipe to pass error samples from IFFT to ScaleAndConvert kernel
using IFFTErrorToScaleAndConvertPipe =
    sycl::ext::intel::pipe<class IFFTErrorToScaleAndConvertPipeID, int8_t, PIPE_CAPACITY>;

// Pipe from ScaleAndConvertKernel to ReduceSetPTEKernel for plaintext+error values
using ScaleToReducePipe =
    sycl::ext::intel::pipe<class ScaleToReducePipeID, int64_t, PIPE_CAPACITY>;