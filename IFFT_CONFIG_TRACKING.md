# IFFT Configuration Tracking

This document tracks different RTL IFFT configurations and their test results.

## Configurations Tested

### Config 1: Input Natural / Output Natural
- **RTL Settings**: `br_in = false`, `br_out = false`
- **SYCL Changes**: None (natural indices for twist)
- **Scaling**: `n_inv = 2.0 * kernel_scale / POLY_N`
- **Test Results**: Not yet tested
- **Issue**:

### Config 2: Input Natural / Output Bit-Reversed
- **RTL Settings**: `br_in = false`, `br_out = true`
- **SYCL Changes**: None (natural indices for twist)
- **Scaling**: `n_inv = 2.0 * kernel_scale / POLY_N`
- **Test Results**: Not yet tested
- **Issue**: Not yet tested

### Config 3: Input Bit-Reversed / Output Natural
- **RTL Settings**: `br_in = true`, `br_out = false`
- **SYCL Changes**: Not yet tested
- **Test Results**: Not yet tested
- **Issue**: Not yet tested

### Config 4: Input Bit-Reversed / Output Bit-Reversed
- **RTL Settings**: `br_in = true`, `br_out = true`
- **SYCL Changes**: Not yet tested
- **Test Results**: Not yet tested
- **Issue**: Not yet tested

## Key Observations

1. **Scaling**: Factor of 2 is needed (CKKS uses N/2 slots → N coefficients)
2. **Post-Twist**: Required to convert standard cyclic IFFT to negacyclic IFFT
3. **Uniform Tests**: Always pass because ordering doesn't matter when all values are identical
4. **Non-Uniform Tests**: Fail due to ordering issues

## Files to Modify

| Config | setup_fft.m | SYCL_scale_and_reduce.h | Notes |
|--------|--------------|------------------------|--------|
| 1 | `br_in=false, br_out=false` |  |  |
| 2 | `br_in=false, br_out=true` |  |  |
| 3 | `br_in=true, br_out=false` |  |  |
| 4 | `br_in=true, br_out=true` |  |  |
