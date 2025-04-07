#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

using namespace sycl;

// Forward declare kernel name
class PolyAddModKernel;

// Function declaration for polynomial addition modulo q
void poly_add_mod(
    queue q,
    uint32_t *p1,
    const uint32_t *p2,
    size_t n,
    uint32_t mod_value
);