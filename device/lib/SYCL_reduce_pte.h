#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdint>

using namespace sycl;

// Forward declare kernel name
class ReduceSetPTEKernel;

// Function declaration for modular reduction of int64_t values
void reduce_pte(
    queue q, 
    const int64_t *conj_vals_int, 
    size_t n, 
    uint32_t mod_value, 
    const uint32_t* const_ratio, 
    uint32_t *out
);