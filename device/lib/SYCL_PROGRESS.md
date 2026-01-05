# SYCL CKKS Pipeline Progress

FPGA-accelerated CKKS symmetric encryption pipeline for SEAL-Embedded, targeting Intel Agilex7.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                      HOST (CPU)                                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   SharedInputBlock[1024]          PerModulusInputBlock[1024] × 3                        │
│   ┌─────────────────────┐         ┌─────────────────────┐ × 3                           │
│   │ encoding (complex)  │         │ secret_key (u32×4)  │                               │
│   │ error (i8×4)        │         │ c1 (u32×4)          │                               │
│   └─────────┬───────────┘         └─────────┬───────────┘                               │
│             │                               │                                           │
│             ▼                               ▼                                           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                    FPGA PIPELINE                                        │
│                                                                                         │
│  ┌──────────────────┐                              ┌───────────────────┐ × 3            │
│  │ SharedEntryKernel│                              │PerModulusEntry<P> │                │
│  └────────┬─────────┘                              └─────────┬─────────┘                │
│           │                                                  │                          │
│           │ SharedToIFFTPipe                                 ├── EntryToNTTAPipe ──────┐│
│           │ (encoding_block)                                 │   (u32×4: secret_key)   ││
│           │                                                  │                         ││
│           │ ErrorToScaleReducePipes                          └── EntryToPolyMultNeg ──┐││
│           │ (i8×4, fanout to 3)                                  (u32×4: c1)          │││
│           │         │                                                                 │││
│           ▼         │                                                                 │││
│  ┌────────────────┐ │                                                                 │││
│  │  IFFTKernel    │ │                                        ┌───────────────┐ × 3    │││
│  │  (single)      │ │                                        │  NTTKernelA<P>│◄───────┘││
│  └────────┬───────┘ │                                        │  (RTL core)   │         ││
│           │         │                                        └───────┬───────┘         ││
│           │         │                                                │                 ││
│           │ IFFTToScaleReducePipes                                   │NTTAToPolyMultNeg││
│           │ (encoding_block, fanout to 3)                            │(u32×4: NTT(s))  ││
│           │         │                                                │                 ││
│           ▼         ▼                                                ▼                 ││
│  ┌─────────────────────────┐ × 3                     ┌────────────────────────┐ × 3    ││
│  │ ScaleAndReduceKernel<P> │                         │PolyMultNegAddKernel<P> │◄───────┘│
│  │ • scale by n_inv        │                         │ • c0 = -NTT(s)*c1      │         │
│  │ • add error             │                         │      + NTT(pte)        │         │
│  │ • Barrett reduce mod q  │                         └───────────┬────────────┘         │
│  └────────────┬────────────┘                                     │                      │
│               │                                                  │PolyAddToExitPipe     │
│               │ ScaleReduceToNTTBPipe                            │(u32×4: c0)           │
│               │ (u32×4: pte mod q)                               │                      │
│               │                                                  │                      │
│               ▼                                                  │                      │
│  ┌───────────────┐ × 3                                           │                      │
│  │  NTTKernelB<P>│                                               │                      │
│  │  (RTL core)   │                                               │                      │
│  └───────┬───────┘                                               │                      │
│          │                                                       │                      │
│          │ NTTBToPolyAddPipe                                     │                      │
│          │ (u32×4: NTT(pte))                                     │                      │
│          │                                                       │                      │
│          └──────────────────────► PolyMultNegAddKernel ◄─────────┘                      │
│                                                                                         │
│                                          │                                              │
│                                          ▼                                              │
│                                 ┌──────────────┐ × 3                                    │
│                                 │ ExitKernel<P>│                                        │
│                                 └──────┬───────┘                                        │
│                                        │                                                │
├────────────────────────────────────────┼────────────────────────────────────────────────┤
│                                        ▼                                                │
│                          PerModulusOutputBlock[1024] × 3                                │
│                          ┌─────────────────────┐                                        │
│                          │ c0 (u32×4)          │                                        │
│                          │ ntt_s (optional)    │                                        │
│                          │ ntt_pte (optional)  │                                        │
│                          └─────────────────────┘                                        │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

Kernel Count: 22 total
  - SharedEntryKernel:        1
  - IFFTKernel:               1
  - ScaleAndReduceKernel<P>:  3
  - PerModulusEntryKernel<P>: 3
  - NTTKernelA<P>:            3  (RTL)
  - NTTKernelB<P>:            3  (RTL)
  - PolyMultNegAddKernel<P>:  3
  - ExitKernel<P>:            3
```

### Parameters
| Parameter | Value |
|-----------|-------|
| POLY_N | 4096 |
| LANES | 4 |
| NUM_BLOCKS | 1024 |
| NUM_MODULI | 3 |

---

## API

```c
void SYCL_encrypt(
    size_t n, size_t logn,
    const double* scales,
    const uint32_t* mod_values,
    const uint32_t* const_ratios,
    complex_double* encoding_buffer,
    int8_t* error_samples,
    uint32_t* const* expanded_s,
    uint32_t* const* uniform_polys,
    uint32_t** c0_outputs,
    uint32_t** c1_outputs,
    uint32_t** s_save,
    uint32_t** c1_save,
    uint32_t** ntt_pte_outputs);
```

---

## File Structure

```
device/lib/
├── SYCL_common.h              # Constants, Barrett reduction, lane helpers
├── SYCL_data_types.h          # Block types, pack/unpack functions
├── SYCL_pipes.h               # PipeSet<P> template
├── SYCL_twiddles.h            # IFFT twiddle factor LUT
├── pipe_utils.hpp             # Intel PipeArray utilities
│
├── SYCL_shared_entry.h        # SharedEntryKernel
├── SYCL_per_mod_entry.h       # PerModulusEntryKernel<P>
├── SYCL_ifft.h                # IFFTKernel
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

## Resource Usage (from FPGA report)

| Kernel | ALUTs | FFs | RAMs | DSPs |
|--------|-------|-----|------|------|
| IFFTKernel | 105K | 89K | 277 | 108 |
| NTTKernel×6 | 161K | 184K | 624 | 750 |
| ScaleReduce×3 | 55K | 41K | 15 | 108 |
| PolyMultNegAdd×3 | 15K | 17K | 15 | 72+120 frac |
| Entry/Exit | ~30K | ~85K | 87 | 0 |
| **Total** | **39%** | **24%** | **19%** | **23%** |

---

## Optimizations Completed

### IFFT Twiddle LUT
- Precomputed 4096-element twiddle factor table
- O(n log n) → O(n) cos/sin calls
- ~12x reduction in transcendental function calls

### Barrett Reduction Specialization
- Hardcoded `const_ratio = floor(2^64 / q)` for 6 known moduli
- Enables FPGA constant propagation

| Modulus | const_ratio_lo | const_ratio_hi |
|---------|----------------|----------------|
| 134012929 | 0x0c84dfe5 | 0x00000020 |
| 134111233 | 0x06814e43 | 0x00000020 |
| 134176769 | 0x02802e03 | 0x00000020 |
| 1053818881 | 0x135bf4ba | 0x00000004 |
| 1054015489 | 0x132a2218 | 0x00000004 |
| 1054212097 | 0x12f85437 | 0x00000004 |

### Single IFFT Instance
- One IFFT serves all 3 moduli via PipeArray fanout
- Saves ~26% ALM vs 3 separate instances
- Reduces host→device transfer by 45%

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

✅ All tests passing (9 test cases × 3 moduli)
