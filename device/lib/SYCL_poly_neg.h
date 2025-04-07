#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdint>

using namespace sycl;

// Forward declare kernel name
class PolyNegModKernel;

// Function declaration for polynomial negation modulo q
void poly_negate_mod(
    queue q, 
    uint32_t *p, 
    size_t n, 
    uint32_t mod_value
);