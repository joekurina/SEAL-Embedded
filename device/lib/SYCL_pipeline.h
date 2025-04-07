#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <complex>

using namespace sycl;

// Forward declare kernel names
class IFFTKernel;
class ScaleAndConvertKernel;

// Function declaration for the pipeline
void pipeline(
    queue q,
    size_t n,                           
    size_t logn,                        
    double scale,                       
    buffer<std::complex<double>, 1>& encoding_buf,
    buffer<int8_t, 1>& error_samples_buf,
    buffer<int64_t, 1>& pt_with_error_buf
);