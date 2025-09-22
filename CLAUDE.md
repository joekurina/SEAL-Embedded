# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SEAL-Embedded is a homomorphic encryption library for embedded devices, implementing CKKS-style encoding and encryption with small code and memory footprint. It consists of two main components:

1. **Device Library** (`device/lib`) - Core C/C++ library for embedded encryption
2. **Adapter** (`adapter`) - C++ application that generates keys and interfaces with Microsoft SEAL. This is only used to generate test data.

What we are primarily concerned with is the SYCL pipeline for the symmetric case.

## Architecture

### Core Components

- **Device Library** 
(`device/lib/`): Contains the main encryption/encoding functionality
  - `ckks_sym.c/h` - Reference Host Symmetric encryption implementation
  - `ckks_asym.c/h` - Reference Host Asymmetric encryption implementation
  - `fft.c/h`, `ntt.c/h`, `intt.c/h` - Reference Host Fast Fourier Transform and Number Theoretic Transform implementations
  - `SYCL_ckks_sym.cpp` - Launches the SYCL kernel pipeline on FPGA
  - `SYCL_*.cpp/h` - SYCL/FPGA acceleration kernels for Intel FPGAs
(`device/lib/rtl`): Contains a compiled RTL C model for the NTT transform
  - `the_nwc_4k_ntt_sycl.hpp` - this is the header for the cycle-accurate C model for the NTT transform.
  - `the_nwc_4k_ntt.a` - The compiled cycle-accurate C model for the compiled RTL implementation of the NTT transform.


### Build System

The project uses CMake with Intel oneAPI compilers (icx/icpx) for SYCL support targeting Intel FPGAs (Agilex7).

### Key Configuration Files

- `device/lib/user_defines.h` - Primary configuration for memory allocation strategies, encryption types, and target platforms
- `device/lib/defines.h` - Internal configuration derived from user_defines.h
- `device/CMakeLists.txt` - Device library build configuration with SYCL/FPGA settings
- `adapter/CMakeLists.txt` - Adapter build configuration with Microsoft SEAL dependency

## Common Development Commands

### Building the Adapter
```bash
cd adapter
cmake -S . -B build
cmake --build build -j
./build/bin/se_adapter  # Generate keys and parameters
```

### Building the Device Library (Local/Emulator)
```bash
cd device
cmake -S . -B build -DSE_BUILD_LOCAL=ON
cmake --build build -j
./build/bin/seal_embedded_tests  # Run tests
```

### Building for FPGA Report Generation
```bash
cd device
cmake -S . -B build -DCMAKE_BUILD_TYPE=Report
cmake --build build --target fpga_report_file
```

### Code Formatting

## Development Workflow

## Testing Strategy

- Unit tests in `device/test/` cover individual components (FFT, NTT, modular arithmetic, encoding/encryption)

## Key Configuration Options

## Target Platforms

- **Local Development**: Standard x86/ARM with SE_BUILD_LOCAL=ON
- **Intel FPGA**: Using SYCL kernels with Intel oneAPI compiler

## Branch Strategy

- Current feature branch: `SYCL_BUFFER_SYM_NTT_RTL` (FPGA RTL implementation)

## Important Notes

- SYCL kernels require Intel oneAPI compiler and target Intel FPGAs (specifically Agilex7). The following must be run before any build commands are run.
```bash
. /opt/intel/oneapi/2025.0/oneapi-vars.sh
```

## RTL NTT Integration (COMPLETED)

The software NTT kernels have been successfully replaced with RTL (Register Transfer Level) implementations for hardware acceleration on Intel FPGAs.

### Architecture Overview

**Previous Implementation:**
- 2 monolithic software NTT kernels (`SYCL_ntt_a.h`, `SYCL_ntt_b.h`)
- ~280 lines each with complex Barrett reduction and modular arithmetic
- Pure software implementation

**Current RTL Implementation:**
- 6-stage RTL pipeline with hardware NTT acceleration
- **NTT A Pipeline:** `SYCL_RTL_ntt_a_input.h` → `SYCL_RTL_ntt_a.h` → `SYCL_RTL_ntt_a_output.h`
- **NTT B Pipeline:** `SYCL_RTL_ntt_b_input.h` → `SYCL_RTL_ntt_b.h` → `SYCL_RTL_ntt_b_output.h`

### RTL Components

#### Core RTL Assets (`device/lib/rtl/`)
- `the_nwc_4k_ntt_sycl.hpp` - RTL interface definitions and data structures
- `the_nwc_4k_ntt.a` - Compiled RTL static library (cycle-accurate C model)

#### RTL Data Structures (`SYCL_ntt_rtl_common.h`)
- `NTT_RTL_Input_Data` - 4x int32_t input structure (16 bytes, 32-byte aligned)
- `NTT_RTL_Output_Data` - 4x int32_t output structure (16 bytes, 32-byte aligned)
- SYCL pipes: `NTTAInputPipe`, `NTTAOutputPipe`, `NTTBInputPipe`, `NTTBOutputPipe`

#### Modulus Selection
The RTL supports 6 different modulus values selected via `port_in_c_s`:
- `port_in_c_s = 0` => `Mod_Val = 134012929u` (root = 7470)
- `port_in_c_s = 1` => `Mod_Val = 134111233u` (root = 3856)
- `port_in_c_s = 2` => `Mod_Val = 134176769u` (root = 24149)
- `port_in_c_s = 3` => `Mod_Val = 1053818881u` (root = 503422)
- `port_in_c_s = 4` => `Mod_Val = 1054015489u` (root = 16768)
- `port_in_c_s = 5` => `Mod_Val = 1054212097u` (root = 7305)

### RTL Kernel Pipeline

#### NTT A Pipeline
1. **`RTLNTTKernel_A_Input`**: Reads from input buffer (secret key), converts uint32_t→RTL format, writes to `NTTAInputPipe`
2. **`RTLNTTKernel_A`**: Main RTL kernel calling `the_nwc_4k_ntt()`, reads from `NTTAInputPipe`, writes to `NTTAOutputPipe`
3. **`RTLNTTKernel_A_Output`**: Converts RTL→uint32_t format, writes to existing `NTTToPolyMultNegPipe`

#### NTT B Pipeline
1. **`RTLNTTKernel_B_Input`**: Reads from existing `ScaleReduceToNTTBPipe`, converts to RTL format, writes to `NTTBInputPipe`
2. **`RTLNTTKernel_B`**: Main RTL kernel for second NTT stage, reads from `NTTBInputPipe`, writes to `NTTBOutputPipe`
3. **`RTLNTTKernel_B_Output`**: Converts RTL→uint32_t format, writes to existing `NTTToAddModPipe` and result buffer

### Build Integration

The RTL library has been integrated into the CMake build system:
```cmake
# Link RTL static library
target_link_libraries(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/lib/rtl/the_nwc_4k_ntt.a)
target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/lib/rtl>)
```

### Pipeline Integration

In `SYCL_ckks_sym.cpp`, the original 2 NTT kernel submissions were replaced with 6 RTL kernel submissions:

**Before:**
```cpp
q.submit([&](handler &h) { NTTKernel_A(n, logn, mod_value, root, const_ratio, c0_s_buf, s_save_buf)(h); });
q.submit([&](handler &h) { NTTKernel_B(n, logn, mod_value, root, const_ratio, ntt_pte_buf)(h); });
```

**After:**
```cpp
// RTL NTT A Pipeline
q.submit([&](handler &h) { RTLNTTKernel_A_Input(n, mod_value, c0_s_buf, s_save_buf)(h); });
q.submit([&](handler &h) { RTLNTTKernel_A(mod_value)(h); });
q.submit([&](handler &h) { RTLNTTKernel_A_Output(n)(h); });

// RTL NTT B Pipeline
q.submit([&](handler &h) { RTLNTTKernel_B_Input(n, mod_value)(h); });
q.submit([&](handler &h) { RTLNTTKernel_B(mod_value)(h); });
q.submit([&](handler &h) { RTLNTTKernel_B_Output(n, ntt_pte_buf)(h); });
```

### Key Features

- **Hardware Acceleration**: RTL kernels run directly on FPGA fabric for maximum performance
- **Pipeline Compatibility**: Full integration with existing SYCL pipeline and pipes
- **Emulator Support**: Handles both FPGA emulator and hardware modes
- **Data Conversion**: Efficient packing/unpacking between uint32_t arrays and RTL 4-element structs
- **Automatic Modulus Selection**: Runtime mapping of SEAL-Embedded modulus values to RTL selectors

### Testing and Validation

To validate the RTL integration:
1. Build with `cmake --build build -j` (requires Intel oneAPI environment)
2. Run tests with `./build/bin/seal_embedded_tests`
3. Verify mathematical correctness matches software implementation
4. Generate FPGA reports with `cmake --build build --target fpga_report_file`