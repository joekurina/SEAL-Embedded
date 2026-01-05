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
All 16 moduli (3 × 27-bit + 13 × 30-bit) are available in `SYCL_common.h`. See the "Available Prime Moduli" section below for full table.

---

## Resource Usage (from FPGA report, 2026-01-05)

### Total Utilization (Intel Agilex7)
| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| ALUTs | 414,771 | 974,400 | **67.9%** |
| FFs | 505,403 | 1,948,800 | **46.7%** |
| RAMs | 1,436 | 7,110 | **25.9%** |
| DSPs | 1,170 | 4,510 | **20.2%** |
| MLABs | 2,026 | 48,720 | **25.9%** |

### Per-Kernel Breakdown
| Kernel | ALUTs | FFs | RAMs | DSPs |
|--------|------:|----:|-----:|-----:|
| IFFTKernel | 146,968 | 142,328 | 360 | 240 |
| NTTKernelA × 3 | 80,346 | 91,821 | 312 | 375 |
| NTTKernelB × 3 | 80,346 | 91,821 | 312 | 375 |
| ScaleAndReduce × 3 | 55,706 | 40,779 | 15 | 108 |
| EntryKernel | 9,931 | 42,010 | 85 | 0 |
| ExitKernel × 3 | 24,144 | 52,911 | 0 | 0 |
| PolyMultNegAdd × 3 | 14,838 | 17,430 | 15 | 72 |

### Key Observations
- IFFT dominates at 147K ALUTs (35% of kernel system) - RTL replacement recommended
- RTL NTT cores efficient at ~27K ALUTs each vs software IFFT's 147K
- 32% ALUTs headroom available for additional moduli or features

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

## Supported Parameters & Moduli

### Maximum Moduli by Polynomial Degree

| N | Max Primes | Prime Bit-Width | Default Scale | Slots |
|---|------------|-----------------|---------------|-------|
| 1024 | 1 | 27-bit | 2^20 | 512 |
| 2048 | 1 | 27-bit | 2^25 | 1024 |
| **4096** | **3** | 27-bit or 30-bit | 2^20 or 2^25 | 2048 |
| **8192** | **6** | 30-bit | 2^25 | 4096 |
| **16384** | **13** | 30-bit | 2^25 | 8192 |

### Available Prime Moduli

**27-bit primes** (q ≡ 1 mod 8192, for N ≤ 4096):
| Index | Modulus | const_ratio[0] | const_ratio[1] |
|-------|---------|----------------|----------------|
| 0 | 134012929 | 0x0c84dfe5 | 0x00000020 |
| 1 | 134111233 | 0x06814e43 | 0x00000020 |
| 2 | 134176769 | 0x02802e03 | 0x00000020 |

**30-bit primes** (q ≡ 1 mod 65536, for N ≥ 4096):
| Index | Modulus | const_ratio[0] | const_ratio[1] | In SYCL |
|-------|---------|----------------|----------------|---------|
| 0 | 1053818881 | 0x135bf4ba | 0x00000004 | ✓ |
| 1 | 1054015489 | 0x132a2218 | 0x00000004 | ✓ |
| 2 | 1054212097 | 0x12f85437 | 0x00000004 | ✓ |
| 3 | 1055260673 | 0x11ef051e | 0x00000004 | ✓ |
| 4 | 1056178177 | 0x11074e88 | 0x00000004 | ✓ |
| 5 | 1056440321 | 0x10c52d4a | 0x00000004 | ✓ |
| 6 | 1058209793 | 0x0f07a84a | 0x00000004 | ✓ |
| 7 | 1060175873 | 0x0d1a6142 | 0x00000004 | ✓ |
| 8 | 1060700161 | 0x0c9725e9 | 0x00000004 | ✓ |
| 9 | 1060765697 | 0x0c86c0d4 | 0x00000004 | ✓ |
| 10 | 1061093377 | 0x0c34cf30 | 0x00000004 | ✓ |
| 11 | 1062469633 | 0x0add3267 | 0x00000004 | ✓ |
| 12 | 1062535169 | 0x0accdb49 | 0x00000004 | ✓ |

### Current SYCL Pipeline Limitations

The current implementation is hardcoded for **N=4096 with 3 moduli**.

To support **4+ moduli** (requires N=8192+):
1. Change `SYCL_NUM_MODULI` in `SYCL_ckks_sym.h`
2. Add Barrett constants to `SYCL_common.h` for indices 3-5 (or more)
3. Add `PipeSet<3>`, `PipeSet<4>`, etc. in `SYCL_pipes.h`
4. Add kernel template instantiations for P=3, 4, 5 in `SYCL_ckks_sym.cpp`
5. Update `SYCL_data_types.h` for larger `PipelineInputBlock`

To support **N=8192**:
1. Change `POLY_N` and `POLY_LOGN` in `SYCL_common.h`
2. Provide RTL NTT core for 8192-point transform (or use software NTT)
3. Update `NUM_BLOCKS` to 2048

---

---

## Investigation: Pipe Resource Scaling

### Current Pipe Configuration
```cpp
constexpr size_t PIPE_CAPACITY = NUM_BLOCKS;  // = POLY_N / LANES = 1024 for N=4096
```

### Resource Impact (from FPGA report)
| Pipe Type | Width | Depth | Count | Total RAMs |
|-----------|-------|-------|-------|------------|
| encoding_block | 512b | 1024 | ~7 | ~182 |
| u32x4 | 128b | 1024 | ~18 | ~126 |
| i8x4 | 32b | 1024 | 3 | ~6 |
| NTTRTLData | 128b | 1024 | 6 | ~42 |

Total pipe RAMs: ~356 (25% of kernel system RAMs)

### Scaling Problem
For N=8192: `NUM_BLOCKS = 2048` → Pipe depths double → RAM usage doubles

### Why Full-Depth Pipes Are Currently Required

**The IFFT Bottleneck:**
```cpp
// SYCL_ifft.h - IFFT reads ALL inputs before producing ANY outputs
complex_double data[POLY_N];  // Local buffer for entire polynomial

for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
    data[blk*LANES...] = SharedToIFFTPipe::read();  // Consume all inputs
}
// ... IFFT computation ...
for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
    IFFTToScaleReducePipes::write(data[blk*LANES...]);  // Produce all outputs
}
```

The IFFT is a global transform that cannot stream - it needs all N inputs before producing any outputs. This creates a "burst" pattern requiring full-depth buffering.

### Strategies to Reduce Pipe Depth

| Strategy | Feasibility | Impact |
|----------|-------------|--------|
| **RTL IFFT with streaming** | High | RTL can pipeline butterfly stages, enabling smaller input buffers |
| **Explicit `min_capacity` hints** | Medium | `[[intel::min_capacity(64)]]` - but scheduler may override |
| **Split polynomial processing** | Low | Breaks IFFT algorithm correctness |
| **Multi-polynomial pipelining** | Medium | Process multiple polynomials concurrently with smaller per-poly buffers |

### Recommended Approach

1. **Short-term**: Accept current pipe depths; focus on RTL IFFT which will fundamentally change the data flow

2. **With RTL IFFT**: The RTL core likely streams data differently:
   - May accept inputs while computing
   - May produce outputs before all inputs consumed
   - Enables `PIPE_CAPACITY = 64-256` instead of `NUM_BLOCKS`

3. **Explicit depth control** (after RTL integration):
   ```cpp
   // Use min_capacity for pipes where producer/consumer rates match
   using MyPipe = sycl::ext::intel::pipe<MyPipeId, u32x4, 64,
       sycl::ext::intel::experimental::min_capacity<64>>;
   ```

### Estimated Savings with RTL IFFT + Reduced Pipes
| Configuration | Pipe RAMs | Savings |
|---------------|-----------|---------|
| Current (N=4096, depth=1024) | ~356 | baseline |
| N=8192, depth=2048 | ~712 | -100% |
| N=8192, depth=256 (with RTL) | ~89 | +75% vs scaled |

---

## Next Steps

1. **RTL IFFT Integration** - Replace software IFFT kernel with RTL implementation (enables pipe optimization)
2. **Multi-prime Optimization** - Currently uses same modulus for all 3 primes in test mode
3. **Hardware Synthesis** - Full FPGA bitstream generation and on-device testing
4. **N=8192 Support** - Extend pipeline to support larger polynomial degrees with 6 moduli

## Completed

- ✓ **Barrett Constants** - All 16 moduli (3 × 27-bit + 13 × 30-bit) now in `SYCL_common.h`
