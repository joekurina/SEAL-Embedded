#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdint>

using namespace sycl;

// Forward declare kernel names
class NTTKernel_1;
class NTTKernel_2;

// NTT function declarations
void ntt_1(queue q, size_t n, size_t logn, uint32_t mod_value, const uint32_t* const_ratio, uint32_t *vec);
void ntt_2(queue q, size_t n, size_t logn, uint32_t mod_value, const uint32_t* const_ratio, uint32_t *vec);