# SYCL CKKS Pipeline Progress

FPGA-accelerated CKKS symmetric encryption pipeline for SEAL-Embedded, targeting Intel Agilex7.

**Last Updated**: 2026-01-05

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                      HOST (CPU)                                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   PipelineInputBlock[1024]                                                              │
│   ┌─────────────────────────────────────┐                                               │
│   │ encoding (complex×2 per lane)       │                                               │
│   │ error (i8×4)                        │                                               │
│   │ secret_key[3] (u32×4 per modulus)   │                                               │
│   │ c1[3] (u32×4 per modulus)           │                                               │
│   └─────────────────┬───────────────────┘                                               │
│                     │                                                                   │
│                     ▼                                                                   │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                    FPGA PIPELINE                                        │
│                                                                                         │
│                        ┌──────────────────┐                                             │
│                        │   EntryKernel    │                                             │
│                        │   (single)       │                                             │
│                        └────────┬─────────┘                                             │
│                                 │                                                       │
│           ┌─────────────────────┼─────────────────────┬─────────────────┐               │
│           │                     │                     │                 │               │
│           │ SharedToIFFTPipe    │ ErrorToScaleReduce  │ EntryToNTTA     │ EntryToPolyMult│
│           │ (encoding_block)    │ Pipes (i8×4, ×3)    │ Pipes (×3)      │ Pipes (×3)    │
│           │                     │                     │ (u32×4: s)      │ (u32×4: c1)   │
│           ▼                     │                     │                 │               │
│  ┌────────────────┐             │                     │                 │               │
│  │  IFFTKernel    │             │                     │                 │               │
│  │  (single)      │             │                     │                 │               │
│  └────────┬───────┘             │                     │                 │               │
│           │                     │                     │                 │               │
│           │ IFFTToScaleReducePipes                    │                 │               │
│           │ (encoding_block, fanout to 3)             │                 │               │
│           │                     │                     │                 │               │
│           ▼                     ▼                     │                 │               │
│  ┌─────────────────────────┐ × 3                      │                 │               │
│  │ ScaleAndReduceKernel<P> │                          │                 │               │
│  │ • scale by n_inv        │                          │                 │               │
│  │ • add error             │                          │                 │               │
│  │ • Barrett reduce mod q  │                          │                 │               │
│  └────────────┬────────────┘                          │                 │               │
│               │                                       │                 │               │
│               │ ScaleReduceToNTTBPipe                 │                 │               │
│               │ (u32×4: pte mod q)                    │                 │               │
│               │                                       │                 │               │
│               ▼                                       ▼                 │               │
│  ┌───────────────┐ × 3                   ┌───────────────┐ × 3          │               │
│  │  NTTKernelB<P>│                       │  NTTKernelA<P>│              │               │
│  │  (RTL core)   │                       │  (RTL core)   │              │               │
│  └───────┬───────┘                       └───────┬───────┘              │               │
│          │                                       │                      │               │
│          │ NTTBToPolyAddPipe                     │NTTAToPolyMultNeg     │               │
│          │ (u32×4: NTT(pte))                     │(u32×4: NTT(s))       │               │
│          │                                       │                      │               │
│          │                                       ▼                      ▼               │
│          │                        ┌────────────────────────┐ × 3                        │
│          └───────────────────────►│PolyMultNegAddKernel<P> │◄───────────┘               │
│                                   │ • c0 = -NTT(s)*c1      │                            │
│                                   │      + NTT(pte)        │                            │
│                                   └───────────┬────────────┘                            │
│                                               │                                         │
│                                               │PolyAddToExitPipe                        │
│                                               │(u32×4: c0)                              │
│                                               │                                         │
│                                               ▼                                         │
│                                      ┌──────────────┐ × 3                               │
│                                      │ ExitKernel<P>│                                   │
│                                      └──────┬───────┘                                   │
│                                             │                                           │
├─────────────────────────────────────────────┼───────────────────────────────────────────┤
│                                             ▼                                           │
│                               PerModulusOutputBlock[1024] × 3                           │
│                               ┌─────────────────────┐                                   │
│                               │ c0 (u32×4)          │                                   │
│                               │ ntt_s (optional)    │                                   │
│                               │ ntt_pte (optional)  │                                   │
│                               └─────────────────────┘                                   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

Kernel Count: 17 total
  - EntryKernel:              1
  - IFFTKernel:               1
  - ScaleAndReduceKernel<P>:  3
  - NTTKernelA<P>:            3  (RTL)
  - NTTKernelB<P>:            3  (RTL)
  - PolyMultNegAddKernel<P>:  3
  - ExitKernel<P>:            3
```

### Parameters
| Parameter | Value |
|-----------|-------|
| POLY_N | 4096 |
| POLY_LOGN | 12 |
| LANES | 4 |
| NUM_BLOCKS | 1024 |
| NUM_MODULI | 3 |

---

## API

```c
void SYCL_encrypt(
    size_t n,
    const double* scales,                    // [NUM_MODULI]: scale per modulus
    const uint32_t* mod_values,              // [NUM_MODULI]: modulus values
    const uint32_t* const_ratios,            // [NUM_MODULI * 2]: {cr0_p0, cr1_p0, cr0_p1, cr1_p1, ...}
    const complex_double* encoding_buffer,   // [n]: pre-encoded complex values
    const int8_t* error_samples,             // [n]: CBD error samples
    const uint32_t* const* secret_keys,      // [NUM_MODULI][n]: expanded secret keys
    const uint32_t* const* uniform_polys,    // [NUM_MODULI][n]: uniform random c1 polynomials
    uint32_t** c0_outputs,                   // [NUM_MODULI][n]: output c0 ciphertexts
    uint32_t** c1_outputs,                   // [NUM_MODULI][n]: output c1 (copy of uniform_polys)
    uint32_t** s_save,                       // [NUM_MODULI][n]: optional NTT(s) for testing
    uint32_t** c1_save,                      // [NUM_MODULI][n]: optional c1 copy for testing
    uint32_t** ntt_pte_outputs);             // [NUM_MODULI][n]: optional NTT(pte) for testing
```

### const_ratios Layout
Barrett reduction constants stored as native `uint32_t[2]` pairs:
```
const_ratios[0] = mod0.const_ratio[0]  (low word)
const_ratios[1] = mod0.const_ratio[1]  (high word)
const_ratios[2] = mod1.const_ratio[0]
const_ratios[3] = mod1.const_ratio[1]
const_ratios[4] = mod2.const_ratio[0]
const_ratios[5] = mod2.const_ratio[1]
```

---

## File Structure

```
device/lib/
├── SYCL_common.h              # Constants, Barrett reduction, lane helpers
├── SYCL_data_types.h          # Block types, pack/unpack functions
├── SYCL_pipes.h               # PipeSet<P> template
├── pipe_utils.hpp             # Intel PipeArray utilities
│
├── SYCL_entry.h               # EntryKernel (single unified entry point)
├── SYCL_ifft.h                # IFFTKernel (to be replaced with RTL)
├── SYCL_scale_and_reduce.h    # ScaleAndReduceKernel<P>
├── SYCL_ntt.h                 # NTTKernel<P, Tag>
├── SYCL_poly_mult_neg_add.h   # PolyMultNegAddKernel<P>
├── SYCL_pipeline_exit.h       # ExitKernel<P>
│
├── SYCL_ckks_sym.cpp          # Pipeline orchestration
├── SYCL_ckks_sym.h            # C interface
└── rtl/
    ├── the_nwc_4k_ntt_sycl.hpp
    └── the_nwc_4k_ntt.a
```

---

## Barrett Reduction

### Implementation
Barrett constants use native `uint32_t[2]` format matching SEAL-Embedded's `Modulus.const_ratio`:
- `const_ratio[0]` = low 32 bits of `floor(2^64 / q)`
- `const_ratio[1]` = high 32 bits of `floor(2^64 / q)`

### Core Functions
```cpp
// Signed 64-bit input (for scale+reduce after IFFT)
uint32_t barrett_reduce_64_core(int64_t val, uint32_t mod, uint32_t cr0, uint32_t cr1, bool negate);

// Unsigned 64-bit input (for polynomial multiplication)
uint32_t barrett_reduce_u64_core(uint64_t product, uint32_t mod, uint32_t cr0, uint32_t cr1);
```

### Known Moduli
| Modulus | const_ratio[0] | const_ratio[1] | Bit Width |
|---------|----------------|----------------|-----------|
| 134012929 | 0x0c84dfe5 | 0x00000020 | 27-bit |
| 134111233 | 0x06814e43 | 0x00000020 | 27-bit |
| 134176769 | 0x02802e03 | 0x00000020 | 27-bit |
| 1053818881 | 0x135bf4ba | 0x00000004 | 30-bit |
| 1054015489 | 0x132a2218 | 0x00000004 | 30-bit |
| 1054212097 | 0x12f85437 | 0x00000004 | 30-bit |

---

## Resource Usage (from FPGA report)

| Kernel | ALUTs | FFs | RAMs | DSPs |
|--------|-------|-----|------|------|
| IFFTKernel | 105K | 89K | 277 | 108 |
| NTTKernel x6 | 161K | 184K | 624 | 750 |
| ScaleReduce x3 | 55K | 41K | 15 | 108 |
| PolyMultNegAdd x3 | 15K | 17K | 15 | 72+120 frac |
| Entry/Exit | ~30K | ~85K | 87 | 0 |
| **Total** | **39%** | **24%** | **19%** | **23%** |

---

## Optimizations Completed

### 1. API Consolidation
- Reduced from 4 public functions to 1 (`SYCL_encrypt`)
- Removed redundant `SYCL_combined_encrypt`, `SYCL_combined_encrypt_pipeline`, `encrypt_impl`

### 2. Native const_ratio Format
- Uses `uint32_t[2]` matching SEAL-Embedded's `Modulus.const_ratio` structure
- No runtime packing/unpacking overhead
- Kernel classes store `uint32_t const_ratio[2]` directly

### 3. Single Entry Point
- Unified `EntryKernel` replaces separate `SharedEntryKernel` + `PerModulusEntryKernel<P>`
- Single `PipelineInputBlock` buffer contains all data
- Reduces kernel count from 22 to 17

### 4. Single IFFT Instance
- One IFFT serves all 3 moduli via PipeArray fanout
- Reduces host->device transfer by 45%
- IFFT kernel to be replaced with RTL

### 5. Barrett Constant Propagation
- Hardcoded `const_ratio` values for 6 known moduli
- Switch-based lookup enables FPGA constant propagation

---

## Build Commands

```bash
cd device/build && make -j4
./bin/seal_embedded_tests --sycl
cmake --build . --target fpga_report_file
cmake --build . --target fpga_hardware_file  # hours
```

---

## Test Status

**All tests passing** (9 test cases × 3 moduli = 27 verifications)

Test output verification:
- `pte calculated` matches `pte decrypted` for all moduli
- Decrypted values match original input values within noise tolerance

---

## Next Steps

1. **RTL IFFT Integration** - Replace software IFFT kernel with RTL implementation
2. **Multi-prime Optimization** - Currently uses same modulus for all 3 primes in test mode
3. **Hardware Synthesis** - Full FPGA bitstream generation and on-device testing
