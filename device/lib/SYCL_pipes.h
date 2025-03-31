#pragma once

#include "SYCL_ckks_sym.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Define pipe capacity - adjust based on N
constexpr size_t PIPE_CAPACITY = 1024; // Reduced from 4096 for testing

// Pipes for the Entrance to IFFT kernel
using EntranceToIFFTPipe = 
    sycl::ext::intel::pipe<class EntranceToIFFTPipeID, complex_double, PIPE_CAPACITY>;

// Pipe from Entrance to ScaleAndConvert for error samples
using EntranceToScaleErrorPipe =
    sycl::ext::intel::pipe<class EntranceToScaleErrorPipeID, int8_t, PIPE_CAPACITY>;

// Pipe from IFFT to ScaleAndConvert kernel
using IFFTToScaleAndConvertPipe = 
    sycl::ext::intel::pipe<class IFFTToScaleAndConvertPipeID, complex_double, PIPE_CAPACITY>;

// Pipe from ScaleAndConvert to Exit kernel
using ScaleAndConvertToExitPipe =
    sycl::ext::intel::pipe<class ScaleAndConvertToExitPipeID, int64_t, PIPE_CAPACITY>;