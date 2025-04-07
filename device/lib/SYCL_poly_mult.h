#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdint>

using namespace sycl;

// Forward declare kernel name
class PolyMultNTTKernel;

// Function declaration for polynomial multiplication in NTT form
void ntt_form_poly_mod_mult(
    queue q, 
    uint32_t *a, 
    const uint32_t *b, 
    size_t n, 
    uint32_t mod_value, 
    const uint32_t* const_ratio
);