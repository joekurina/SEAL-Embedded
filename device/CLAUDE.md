# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SEAL-Embedded is a CKKS homomorphic encryption implementation optimized for embedded devices and FPGA hardware. The project combines C/C++ implementations with Intel SYCL for FPGA acceleration. It provides symmetric and asymmetric encryption modes for CKKS operations including NTT/INTT transforms, FFT operations, and polynomial arithmetic.

## Build System

The project uses CMake with Intel's oneAPI compilers (icx/icpx) and SYCL for FPGA development.

### Common Build Commands

```bash
# Configure for local/emulator build (default)
cmake -B build

# Build library only
cmake -B build -DSE_BUILD_TYPE=Lib
cmake --build build

# Build with tests (default)
cmake -B build -DSE_BUILD_TYPE=Tests  
cmake --build build

# Generate FPGA report files (hardware analysis)
cmake --build build --target fpga_report_file

# Run tests
./build/bin/seal_embedded_tests

# Test all configurations
./scripts/test_all_configs.sh
```

### Build Types and Configurations

- **Debug**: FPGA emulator mode with debugging enabled
- **Release**: FPGA emulator mode optimized
- **Report**: Hardware compilation for FPGA report generation
- **SE_BUILD_TYPE**: Controls whether to build library only ("Lib") or with tests ("Tests")

## Architecture

### Core Components

**Library Structure (`lib/`)**:
- `seal_embedded.h/c`: Main API with setup, encrypt, and cleanup functions
- `ckks_common.h/c`: Common CKKS encryption operations
- `ckks_sym.c/ckks_asym.c`: Symmetric and asymmetric encryption implementations
- `ntt.c/intt.c`: Number Theoretic Transform implementations
- `fft.c`: Fast Fourier Transform for encoding/decoding
- `polymodmult.c`: Polynomial modular multiplication
- `parameters.c`: Parameter management and validation

**SYCL FPGA Kernels (`lib/SYCL_*.h`)**:
- `SYCL_ckks_sym.h`: Main SYCL kernel orchestration
- `SYCL_entrance.h`: Data input kernel (buffer to pipes)
- `SYCL_exit.h`: Data output kernel (pipes to buffers) 
- `SYCL_ntt_a.h/SYCL_ntt_b.h`: Parallel NTT kernels
- `SYCL_ifft.h`: Inverse FFT kernel
- `SYCL_poly_*.h`: Polynomial arithmetic kernels
- `SYCL_pipes.h`: Inter-kernel communication pipes

**Configuration System (`defines.h`, `user_defines.h`)**:
- Extensive compile-time configuration options
- NTT/INTT computation modes (on-the-fly vs precomputed)
- Memory optimization strategies
- Platform-specific adaptations

### Pipeline Architecture

The SYCL implementation uses a pipelined datapath with SYCL pipes for inter-kernel communication:
1. **EntranceKernel**: Reads from host buffers, writes to pipes
2. **Transform Kernels**: NTT, IFFT operations in parallel
3. **Polynomial Kernels**: Addition, multiplication, scaling
4. **ExitKernel**: Collects results from pipes to output buffers

### Configuration Parameters

The system supports various optimization modes configured through defines:
- `SE_NTT_TYPE`: 0=on-the-fly, 1=one-shot, 2=regular, 3=fast
- `SE_INDEX_MAP_TYPE`: Memory allocation strategy for index maps
- `SE_SK_TYPE`: Secret key persistence options
- `SE_DATA_LOAD_TYPE`: How precomputed data is loaded

## Testing

**Test Structure (`test/`)**:
- Unit tests for each component (ntt_tests.c, fft_tests.c, etc.)
- CKKS operation tests (encoding, encryption, arithmetic)
- SYCL-specific tests for kernel validation
- API integration tests

**Benchmarking (`bench/`)**:
- Performance benchmarks for all major operations
- Comparison between different configuration modes

## Development Workflow

1. Modify configuration in `user_defines.h` if needed
2. Use emulator builds for development and testing
3. Generate reports for FPGA resource analysis
4. Run comprehensive test suite before commits
5. Use `scripts/test_all_configs.sh` for validation across configurations

## Target Platforms

- **Local/Emulator**: Development and testing on host CPU
- **Intel Agilex7 FPGA**: Primary hardware target
- **Embedded platforms**: NRF5, Azure Sphere (M4/A7)

## Memory Management

The system supports both malloc-based and fixed memory pool allocation depending on target platform. For embedded targets, precomputed data can be embedded directly in code or loaded from files.