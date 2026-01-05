# SYCL CKKS Pipeline Progress Document

## Project Overview

FPGA-accelerated CKKS symmetric encryption pipeline for SEAL-Embedded, targeting Intel Agilex7.

---

## Current Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Host (CPU)                                       │
├──────────────────────────────────────────────────────────────────────────┤
│  PipelineInputBlock[1024]  ──────────────────►  PipelineOutputBlock[1024]│
└──────────────────────────────────────────────────────────────────────────┘
                    │                                       ▲
                    ▼                                       │
┌──────────────────────────────────────────────────────────────────────────┐
│                      FPGA Pipeline (per modulus P=0,1,2)                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   EntryKernel ─┬─► IFFTKernel ──► ScaleReduceKernel ──► NTTKernelB ──┐  │
│                │                                                      │  │
│                ├─► NTTKernelA ────────────────────────────────────┐  │  │
│                │                                                   │  │  │
│                └─► (c1) ──────────────────────────────────────┐   │  │  │
│                                                                │   │  │  │
│                              PolyMultNegAddKernel ◄────────────┴───┴──┘  │
│                                       │                                  │
│                                       ▼                                  │
│                                  ExitKernel                              │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Current Parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| POLY_N | 4096 | Polynomial degree |
| POLY_LOGN | 12 | log2(POLY_N) |
| LANES | 4 | Elements per cycle (RTL constraint) |
| NUM_BLOCKS | 1024 | POLY_N / LANES |
| PIPE_CAPACITY | 1024 | Blocks per pipe |
| Pipeline Instances | 3 | One per modulus (P=0,1,2) |

---

## Completed Optimizations

### Phase 1-6: Architecture Refactoring
- [x] Created `SYCL_common.h` - centralized utilities (Barrett reduction, lane accessors)
- [x] Created `SYCL_data_types.h` - mega-struct types (PipelineInputBlock, PipelineOutputBlock)
- [x] Created `SYCL_pipes.h` - unified PipeSet<P> template
- [x] Created `SYCL_ntt.h` - unified NTTKernel<P, Tag> (replaced 6 separate files)
- [x] Created `SYCL_pipeline_entry.h` and `SYCL_pipeline_exit.h`

### Phase 7: Pipeline Orchestration
- [x] Updated `SYCL_ckks_sym.cpp` with mega-struct buffers
- [x] Created pack/unpack functions for host↔device transfer

### Phase 8: Cleanup
- [x] Deleted obsolete files (8 old RTL NTT wrappers, SYCL_lanes.h)

### Phase 9: Performance Optimizations
- [x] NTT kernel: Changed to infinite `while(true)` loop → **fMAX: 283→432 MHz**
- [x] Kernel fusion: Merged PolyMultNeg + PolyAdd → `PolyMultNegAddKernel`
- [x] Removed unused `PolyMultNegToPolyAddPipe`
- [x] IFFT: Added `#pragma unroll 4` (partial unroll for area balance)
- [x] All kernels: Added `[[intel::initiation_interval(1)]]`

---

## Current File Structure

```
device/lib/
├── SYCL_common.h              # Constants, Barrett reduction, lane helpers
├── SYCL_data_types.h          # PipelineInputBlock, PipelineOutputBlock, pack/unpack
├── SYCL_pipes.h               # PipeSet<P> template with all pipe definitions
├── SYCL_pipeline_entry.h      # EntryKernel<P>
├── SYCL_pipeline_exit.h       # ExitKernel<P>
├── SYCL_ntt.h                 # NTTKernel<P, Tag> (unified NTT_A and NTT_B)
├── SYCL_ifft.h                # IFFTKernel<P>
├── SYCL_scale_and_reduce.h    # ScaleAndReduceKernel<P>
├── SYCL_poly_mult_neg_add.h   # PolyMultNegAddKernel<P> (fused)
├── SYCL_ckks_sym.cpp          # Pipeline orchestration, host interface
├── SYCL_ckks_sym.h            # C interface header
└── rtl/
    ├── the_nwc_4k_ntt_sycl.hpp  # RTL NTT header
    └── the_nwc_4k_ntt.a         # RTL NTT compiled IP
```

---

## Current Resource Usage (per kernel instance)

| Kernel | ALM | Registers | DSP | BRAM | fMAX |
|--------|-----|-----------|-----|------|------|
| IFFTKernel | ~13% (with unroll 4) | ~16% | ~11% | ~3% | 480 MHz |
| NTTKernel (RTL) | TBD | TBD | TBD | TBD | **432 MHz** |
| ScaleReduceKernel | Low | Low | Low | Low | 480 MHz |
| PolyMultNegAddKernel | Low | Low | Low | Low | 480 MHz |
| Entry/Exit | Minimal | Minimal | None | Low | 480 MHz |

**Bottleneck**: NTT kernel at 432 MHz (RTL core limitation)

---

## Known Issue: IFFT Duplication

### Current Behavior
Each of the 3 pipeline instances (P=0,1,2) has its own IFFT kernel:
- IFFTKernelTask<0>
- IFFTKernelTask<1>  
- IFFTKernelTask<2>

### Problem
The IFFT operates on **complex values before modular reduction**. The IFFT output is identical for all 3 moduli - only the subsequent ScaleReduce applies modulus-specific operations.

### Current Waste
- 3× IFFT kernels consuming ~39% ALM each (with full unroll) or ~13% each (with unroll 4)
- 3× encoding buffer transfers to device
- 3× identical IFFT computations

### Proposed Optimization
```
                                    ┌─► ScaleReduce(mod0) ─► NTT_B(mod0) ─► ...
EntryKernel ─► IFFTKernel (single) ─┼─► ScaleReduce(mod1) ─► NTT_B(mod1) ─► ...
                                    └─► ScaleReduce(mod2) ─► NTT_B(mod2) ─► ...
```

**Benefits:**
- 1 IFFT instead of 3 → **~26% ALM savings** (with unroll 4)
- 1 encoding buffer transfer instead of 3
- Reduced host↔device bandwidth

---

## Future Optimizations

### High Priority
| Optimization | Impact | Status |
|--------------|--------|--------|
| Single IFFT for all moduli | ~26% ALM, 3x less transfer | Planned |
| IFFT twiddle factor LUT | Eliminate cos/sin compute | Planned |
| Barrett specialization | Hardcode const_ratio for 6 moduli | Planned |

### Medium Priority
| Optimization | Impact | Status |
|--------------|--------|--------|
| IFFT memory banking | Faster butterfly access | Planned |
| Reduce PIPE_CAPACITY | Save BRAM if throughput allows | Planned |
| Double buffering | Overlap transfer with compute | Planned |

### Research / Long-term
| Optimization | Impact | Status |
|--------------|--------|--------|
| n=16384 support | 4x larger polynomials | Feasible (see below) |
| Streaming IFFT | Reduce latency | Complex |

---

## n=16384 Feasibility

### Host Code: ✅ Ready
- All allocations are dynamic (`calloc(n, ...)`)
- `uint16_t` index_map supports up to n=65535
- Test framework has n=16384 commented out, ready to enable

### SYCL Pipeline: ⚠️ Changes Required
```cpp
// SYCL_common.h
constexpr size_t POLY_N = 16384;    // was 4096
constexpr size_t POLY_LOGN = 14;    // was 12
```

### RTL NTT: ❌ New Core Required
- Current: `the_nwc_4k_ntt` (4096-point)
- Needed: `the_nwc_16k_ntt` (16384-point)

### Resource Impact
| Resource | n=4096 | n=16384 | Factor |
|----------|--------|---------|--------|
| IFFT array | 64 KB | 256 KB | 4x |
| Pipe buffers | ~3.5 MB | ~14 MB | 4x |
| NTT stages | 12 | 14 | +2 |

---

## Test Status

All tests passing:
- 9 test cases
- 3 pipeline instances (P=0,1,2)
- SYCL emulator verified

---

## Build Commands

```bash
# Build
cd device/build && make -j4

# Run tests
./bin/seal_embedded_tests --sycl

# Generate FPGA report
cmake --build . --target fpga_report_file

# Full hardware synthesis (hours)
cmake --build . --target fpga_hardware_file
```

---

## Version History

| Date | Change |
|------|--------|
| 2026-01-05 | Kernel fusion (PolyMultNeg + PolyAdd) |
| 2026-01-05 | NTT infinite loop optimization (283→432 MHz) |
| 2026-01-05 | IFFT partial unroll (#pragma unroll 4) |
| 2026-01-03 | Initial SYCL pipeline refactoring complete |
