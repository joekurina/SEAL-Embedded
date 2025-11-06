# Debug Output Structure

## Overview
The SYCL_ckks_sym.cpp file has been updated to capture intermediate computation results from all kernels and save them to files for debugging and verification purposes.

## Output Directory Structure

Each time `SYCL_combined_encrypt` is called, it creates a new test directory:

```
test_0/
test_1/
test_2/
...
```

The test counter increments automatically for each call to the function.

## Output Files

Within each test directory, the following files are created **for each modulus** used in the computation:

### NTT Kernel A (Secret Key Processing)
- **`NTT_A_INPUT_mod_<modulus>.txt`** - Input to NTT Kernel A (expanded secret key before NTT)
- **`NTT_A_OUTPUT_mod_<modulus>.txt`** - Output from NTT Kernel A (NTT of secret key)

### NTT Kernel B (Plaintext + Error Processing)
- **`NTT_B_INPUT_mod_<modulus>.txt`** - Input to NTT Kernel B (scaled plaintext + error before NTT)
- **`NTT_B_OUTPUT_mod_<modulus>.txt`** - Output from NTT Kernel B (NTT of plaintext + error)

### IFFT Kernel (Encoding Processing)
- **`IFFT_INPUT_mod_<modulus>.txt`** - Input to IFFT Kernel (complex encoded plaintext)
- **`IFFT_OUTPUT_mod_<modulus>.txt`** - Output from IFFT Kernel (transformed complex values)

## File Format

### Integer Buffers (NTT A and B)
Each line contains a single unsigned 32-bit integer value:
```
value_0
value_1
value_2
...
```

### Complex Buffers (IFFT)
Each line contains real and imaginary parts separated by a space:
```
real_0 imag_0
real_1 imag_1
real_2 imag_2
...
```

## Example Directory Structure

For a test with 3 moduli (1053818881, 1054015489, 1054212097), you would see:

```
test_0/
├── NTT_A_INPUT_mod_1053818881.txt
├── NTT_A_OUTPUT_mod_1053818881.txt
├── NTT_B_INPUT_mod_1053818881.txt
├── NTT_B_OUTPUT_mod_1053818881.txt
├── IFFT_INPUT_mod_1053818881.txt
├── IFFT_OUTPUT_mod_1053818881.txt
├── NTT_A_INPUT_mod_1054015489.txt
├── NTT_A_OUTPUT_mod_1054015489.txt
├── NTT_B_INPUT_mod_1054015489.txt
├── NTT_B_OUTPUT_mod_1054015489.txt
├── IFFT_INPUT_mod_1054015489.txt
├── IFFT_OUTPUT_mod_1054015489.txt
├── NTT_A_INPUT_mod_1054212097.txt
├── NTT_A_OUTPUT_mod_1054212097.txt
├── NTT_B_INPUT_mod_1054212097.txt
├── NTT_B_OUTPUT_mod_1054212097.txt
├── IFFT_INPUT_mod_1054212097.txt
└── IFFT_OUTPUT_mod_1054212097.txt
```

## Implementation Details

### Buffer Allocation
The intermediate buffers are allocated as `std::vector` instances in host memory:
```cpp
std::vector<uint32_t> ntt_a_input_vec(n);
std::vector<uint32_t> ntt_a_output_vec(n);
std::vector<uint32_t> ntt_b_input_vec(n);
std::vector<uint32_t> ntt_b_output_vec(n);
std::vector<std::complex<double>> ifft_output_vec(n);
```

These are then wrapped in SYCL buffers and passed to the kernels.

### Kernel Updates
Each kernel has been modified to write its input and output to these debug buffers:
- **IFFTKernel**: Writes output to `ifft_output_buffer`
- **NTTKernel_A**: Writes input to `ntt_a_input_buffer` and output to `ntt_a_output_buffer`
- **NTTKernel_B**: Writes input to `ntt_b_input_buffer` and output to `ntt_b_output_buffer`

### File Writing
After the pipeline completes, the buffers are accessed using `get_host_access(sycl::read_only)` and written to text files for easy inspection.

## Usage Notes

1. **Performance Impact**: Writing these files will slow down execution. This is intended for debugging only.
2. **Disk Space**: Each test with multiple moduli can generate significant data. Monitor disk usage.
3. **Disable for Production**: Comment out or remove the file writing section for production builds.
4. **Thread Safety**: The static `test_counter` is not thread-safe. Use only in single-threaded contexts or add synchronization.

## Cleanup

To remove all test directories:
```bash
rm -rf test_*
```
