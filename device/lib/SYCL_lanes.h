#pragma once

#include "SYCL_ckks_sym.h"
#include <cstdint>

// Ensure packed layout matches 4-lane payloads the host will pack/unpack.
#ifndef CSL_PACKED
#ifdef _WIN32
#define CSL_PACKED(struct_def) __pragma(pack(push, 1)) struct_def __pragma(pack(pop))
#else
#define CSL_PACKED(struct_def) struct_def __attribute__((__packed__))
#endif
#endif

CSL_PACKED(typedef struct {
    complex_double element0;
    complex_double element1;
    complex_double element2;
    complex_double element3;
}) encoding_buffer_input;

CSL_PACKED(typedef struct {
    uint32_t element0;
    uint32_t element1;
    uint32_t element2;
    uint32_t element3;
}) u32x4_input;

CSL_PACKED(typedef struct {
    int8_t element0;
    int8_t element1;
    int8_t element2;
    int8_t element3;
}) i8x4_input;

CSL_PACKED(typedef struct {
    int64_t element0;
    int64_t element1;
    int64_t element2;
    int64_t element3;
}) i64x4_input;

static_assert(sizeof(encoding_buffer_input) == sizeof(complex_double) * 4,
              "encoding_buffer_input must pack exactly 4 complex_double values");
static_assert(sizeof(u32x4_input) == sizeof(uint32_t) * 4,
              "u32x4_input must pack exactly 4 uint32_t values");
static_assert(sizeof(i8x4_input) == sizeof(int8_t) * 4,
              "i8x4_input must pack exactly 4 int8_t values");
static_assert(sizeof(i64x4_input) == sizeof(int64_t) * 4,
              "i64x4_input must pack exactly 4 int64_t values");
