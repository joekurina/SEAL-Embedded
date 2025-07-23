#pragma once

#include "SYCL_ckks_sym.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// FFT Input/Output data structures
typedef struct {
    double port_data_in_0re;
    double port_data_in_0im;
    double port_data_in_1re;
    double port_data_in_1im;
    double port_data_in_2re;
    double port_data_in_2im;
    double port_data_in_3re;
    double port_data_in_3im;
} FFT_Input_Data;

typedef struct {
    double port_data_out_0re;
    double port_data_out_0im;
    double port_data_out_1re;
    double port_data_out_1im;
    double port_data_out_2re;
    double port_data_out_2im;
    double port_data_out_3re;
    double port_data_out_3im;
} FFT_Output_Data;

// Define pipes for communication between kernels
constexpr size_t PIPE_CAPACITY = 4096; // Adjust based on your needs
constexpr size_t FFT_PIPE_CAPACITY = 1024; // For FFT input/output

// Pipe from Entrance to FFT kernel for input data
using EntranceToFFTPipe = 
    sycl::ext::intel::pipe<class EntranceToFFTPipeID, FFT_Input_Data, FFT_PIPE_CAPACITY>;

// Pipe from IFFT to ScaleAndConvert kernel for transformed values
using IFFTToScaleAndReducePipe = 
    sycl::ext::intel::pipe<class IFFTToScaleAndReducePipeID, FFT_Output_Data, FFT_PIPE_CAPACITY>;

// Pipe to pass error samples from IFFT to ScaleAndConvert kernel
using IFFTErrorToScaleAndReducePipe =
    sycl::ext::intel::pipe<class IFFTErrorToScaleAndReducePipeID, int8_t, PIPE_CAPACITY>;

// Pipe from ScaleAndConvertKernel to ReduceSetPTEKernel for plaintext+error values
using ScaleToReducePipe =
    sycl::ext::intel::pipe<class ScaleToReducePipeID, int64_t, PIPE_CAPACITY>;

// Pipe from ScaleAndConvertKernel to NTTKernel_1 for plaintext+error values
using ScaleReduceToNTTBPipe =
    sycl::ext::intel::pipe<class ScaleReduceToNTTBPipeID, uint32_t, PIPE_CAPACITY>;

// Pipe from NTTKernel_B to AddModKernel for NTT(PTE+error)
using NTTToAddModPipe =
    sycl::ext::intel::pipe<class NTTToAddModPipeID, uint32_t, PIPE_CAPACITY>;

// Pipe from NTTKernel_A to PolyMultNegNTTKernel for NTT(s)
using NTTToPolyMultNegPipe =
    sycl::ext::intel::pipe<class NTTToPolyMultNegPipeID, uint32_t, PIPE_CAPACITY>;

// Pipe from PolyMultNegNTTKernel to PolyAddModKernel for -(NTT(s)*c1)
using PolyMultNegToPolyAddModPipe =
    sycl::ext::intel::pipe<class PolyMultNegToPolyAddModPipeID, uint32_t, PIPE_CAPACITY>;