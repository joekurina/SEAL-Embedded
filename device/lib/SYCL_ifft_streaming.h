#pragma once

#include "SYCL_common.h"
#include "SYCL_pipes.h"
#include "SYCL_data_types.h"
#include "SYCL_ifft_4k_roots.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sycl_ckks {

// Single-path Delay Feedback (SDF) streaming IFFT
// Processes 1 complex sample per cycle through all stages
// Each stage i has a delay buffer of size N/2^(i+1)
// Total latency: N-1 cycles

class StreamingIFFTKernelTask;

class StreamingIFFTKernel {
public:
    StreamingIFFTKernel() {}

    void operator()(sycl::handler& h) const {
        h.single_task<StreamingIFFTKernelTask>([=]() [[intel::kernel_args_restrict]] {

            // Delay buffers for each stage
            // Stage 0: 2048 elements, Stage 1: 1024, ..., Stage 11: 1 element
            complex_double delay0[2048];
            complex_double delay1[1024];
            complex_double delay2[512];
            complex_double delay3[256];
            complex_double delay4[128];
            complex_double delay5[64];
            complex_double delay6[32];
            complex_double delay7[16];
            complex_double delay8[8];
            complex_double delay9[4];
            complex_double delay10[2];
            complex_double delay11[1];

            // Counters for each stage (track position within the stage's cycle)
            int cnt0 = 0, cnt1 = 0, cnt2 = 0, cnt3 = 0;
            int cnt4 = 0, cnt5 = 0, cnt6 = 0, cnt7 = 0;
            int cnt8 = 0, cnt9 = 0, cnt10 = 0, cnt11 = 0;

            // Total iterations: N input + N-1 to flush pipeline
            constexpr int kTotalIterations = POLY_N + POLY_N - 1;

            [[intel::initiation_interval(1)]]
            for (int iter = 0; iter < kTotalIterations; iter++) {
                complex_double x;

                // Read input (or zero for flushing)
                if (iter < POLY_N) {
                    int blk = iter / LANES;
                    int lane = iter % LANES;
                    if (lane == 0) {
                        encoding_block block = SharedToIFFTPipe::read();
                        // Store in registers for subsequent lanes
                        x = block.element0;
                        delay0[0] = block.element1;  // Temporarily use delay0[0]
                        delay0[1] = block.element2;
                        delay0[2] = block.element3;
                    } else if (lane == 1) {
                        x = delay0[0];
                    } else if (lane == 2) {
                        x = delay0[1];
                    } else {
                        x = delay0[2];
                    }
                } else {
                    x = complex_double(0.0, 0.0);
                }

                // Stage 0: delay = 2048, twiddle indices 2048..4095
                {
                    constexpr int delay_size = 2048;
                    constexpr int hh = 2048;
                    int idx = cnt0;
                    complex_double delayed = delay0[idx];
                    
                    if (cnt0 < delay_size) {
                        // First half: just store
                        delay0[idx] = x;
                        x = delayed;  // Output old value (initially garbage, becomes valid later)
                    } else {
                        // Second half: butterfly
                        int j = cnt0 - delay_size;
                        complex_double twiddle(IFFT_4K_TWIDDLE_REAL[hh + j], 
                                               IFFT_4K_TWIDDLE_IMAG[hh + j]);
                        complex_double u = delayed;
                        complex_double v = x;
                        delay0[idx] = u + v;
                        x = (u - v) * twiddle;
                    }
                    cnt0 = (cnt0 + 1) & (2 * delay_size - 1);
                }

                // Stage 1: delay = 1024, twiddle indices 1024..2047
                {
                    constexpr int delay_size = 1024;
                    constexpr int hh = 1024;
                    int idx = cnt1 & (delay_size - 1);
                    complex_double delayed = delay1[idx];
                    
                    if ((cnt1 & delay_size) == 0) {
                        delay1[idx] = x;
                        x = delayed;
                    } else {
                        int j = cnt1 & (delay_size - 1);
                        complex_double twiddle(IFFT_4K_TWIDDLE_REAL[hh + j], 
                                               IFFT_4K_TWIDDLE_IMAG[hh + j]);
                        complex_double u = delayed;
                        complex_double v = x;
                        delay1[idx] = u + v;
                        x = (u - v) * twiddle;
                    }
                    cnt1 = (cnt1 + 1) & (2 * delay_size - 1);
                }

                // Stage 2: delay = 512
                {
                    constexpr int delay_size = 512;
                    constexpr int hh = 512;
                    int idx = cnt2 & (delay_size - 1);
                    complex_double delayed = delay2[idx];
                    
                    if ((cnt2 & delay_size) == 0) {
                        delay2[idx] = x;
                        x = delayed;
                    } else {
                        int j = cnt2 & (delay_size - 1);
                        complex_double twiddle(IFFT_4K_TWIDDLE_REAL[hh + j], 
                                               IFFT_4K_TWIDDLE_IMAG[hh + j]);
                        complex_double u = delayed;
                        complex_double v = x;
                        delay2[idx] = u + v;
                        x = (u - v) * twiddle;
                    }
                    cnt2 = (cnt2 + 1) & (2 * delay_size - 1);
                }

                // Stage 3: delay = 256
                {
                    constexpr int delay_size = 256;
                    constexpr int hh = 256;
                    int idx = cnt3 & (delay_size - 1);
                    complex_double delayed = delay3[idx];
                    
                    if ((cnt3 & delay_size) == 0) {
                        delay3[idx] = x;
                        x = delayed;
                    } else {
                        int j = cnt3 & (delay_size - 1);
                        complex_double twiddle(IFFT_4K_TWIDDLE_REAL[hh + j], 
                                               IFFT_4K_TWIDDLE_IMAG[hh + j]);
                        complex_double u = delayed;
                        complex_double v = x;
                        delay3[idx] = u + v;
                        x = (u - v) * twiddle;
                    }
                    cnt3 = (cnt3 + 1) & (2 * delay_size - 1);
                }

                // Stage 4: delay = 128
                {
                    constexpr int delay_size = 128;
                    constexpr int hh = 128;
                    int idx = cnt4 & (delay_size - 1);
                    complex_double delayed = delay4[idx];
                    
                    if ((cnt4 & delay_size) == 0) {
                        delay4[idx] = x;
                        x = delayed;
                    } else {
                        int j = cnt4 & (delay_size - 1);
                        complex_double twiddle(IFFT_4K_TWIDDLE_REAL[hh + j], 
                                               IFFT_4K_TWIDDLE_IMAG[hh + j]);
                        complex_double u = delayed;
                        complex_double v = x;
                        delay4[idx] = u + v;
                        x = (u - v) * twiddle;
                    }
                    cnt4 = (cnt4 + 1) & (2 * delay_size - 1);
                }

                // Stage 5: delay = 64
                {
                    constexpr int delay_size = 64;
                    constexpr int hh = 64;
                    int idx = cnt5 & (delay_size - 1);
                    complex_double delayed = delay5[idx];
                    
                    if ((cnt5 & delay_size) == 0) {
                        delay5[idx] = x;
                        x = delayed;
                    } else {
                        int j = cnt5 & (delay_size - 1);
                        complex_double twiddle(IFFT_4K_TWIDDLE_REAL[hh + j], 
                                               IFFT_4K_TWIDDLE_IMAG[hh + j]);
                        complex_double u = delayed;
                        complex_double v = x;
                        delay5[idx] = u + v;
                        x = (u - v) * twiddle;
                    }
                    cnt5 = (cnt5 + 1) & (2 * delay_size - 1);
                }

                // Stage 6: delay = 32
                {
                    constexpr int delay_size = 32;
                    constexpr int hh = 32;
                    int idx = cnt6 & (delay_size - 1);
                    complex_double delayed = delay6[idx];
                    
                    if ((cnt6 & delay_size) == 0) {
                        delay6[idx] = x;
                        x = delayed;
                    } else {
                        int j = cnt6 & (delay_size - 1);
                        complex_double twiddle(IFFT_4K_TWIDDLE_REAL[hh + j], 
                                               IFFT_4K_TWIDDLE_IMAG[hh + j]);
                        complex_double u = delayed;
                        complex_double v = x;
                        delay6[idx] = u + v;
                        x = (u - v) * twiddle;
                    }
                    cnt6 = (cnt6 + 1) & (2 * delay_size - 1);
                }

                // Stage 7: delay = 16
                {
                    constexpr int delay_size = 16;
                    constexpr int hh = 16;
                    int idx = cnt7 & (delay_size - 1);
                    complex_double delayed = delay7[idx];
                    
                    if ((cnt7 & delay_size) == 0) {
                        delay7[idx] = x;
                        x = delayed;
                    } else {
                        int j = cnt7 & (delay_size - 1);
                        complex_double twiddle(IFFT_4K_TWIDDLE_REAL[hh + j], 
                                               IFFT_4K_TWIDDLE_IMAG[hh + j]);
                        complex_double u = delayed;
                        complex_double v = x;
                        delay7[idx] = u + v;
                        x = (u - v) * twiddle;
                    }
                    cnt7 = (cnt7 + 1) & (2 * delay_size - 1);
                }

                // Stage 8: delay = 8
                {
                    constexpr int delay_size = 8;
                    constexpr int hh = 8;
                    int idx = cnt8 & (delay_size - 1);
                    complex_double delayed = delay8[idx];
                    
                    if ((cnt8 & delay_size) == 0) {
                        delay8[idx] = x;
                        x = delayed;
                    } else {
                        int j = cnt8 & (delay_size - 1);
                        complex_double twiddle(IFFT_4K_TWIDDLE_REAL[hh + j], 
                                               IFFT_4K_TWIDDLE_IMAG[hh + j]);
                        complex_double u = delayed;
                        complex_double v = x;
                        delay8[idx] = u + v;
                        x = (u - v) * twiddle;
                    }
                    cnt8 = (cnt8 + 1) & (2 * delay_size - 1);
                }

                // Stage 9: delay = 4
                {
                    constexpr int delay_size = 4;
                    constexpr int hh = 4;
                    int idx = cnt9 & (delay_size - 1);
                    complex_double delayed = delay9[idx];
                    
                    if ((cnt9 & delay_size) == 0) {
                        delay9[idx] = x;
                        x = delayed;
                    } else {
                        int j = cnt9 & (delay_size - 1);
                        complex_double twiddle(IFFT_4K_TWIDDLE_REAL[hh + j], 
                                               IFFT_4K_TWIDDLE_IMAG[hh + j]);
                        complex_double u = delayed;
                        complex_double v = x;
                        delay9[idx] = u + v;
                        x = (u - v) * twiddle;
                    }
                    cnt9 = (cnt9 + 1) & (2 * delay_size - 1);
                }

                // Stage 10: delay = 2
                {
                    constexpr int delay_size = 2;
                    constexpr int hh = 2;
                    int idx = cnt10 & (delay_size - 1);
                    complex_double delayed = delay10[idx];
                    
                    if ((cnt10 & delay_size) == 0) {
                        delay10[idx] = x;
                        x = delayed;
                    } else {
                        int j = cnt10 & (delay_size - 1);
                        complex_double twiddle(IFFT_4K_TWIDDLE_REAL[hh + j], 
                                               IFFT_4K_TWIDDLE_IMAG[hh + j]);
                        complex_double u = delayed;
                        complex_double v = x;
                        delay10[idx] = u + v;
                        x = (u - v) * twiddle;
                    }
                    cnt10 = (cnt10 + 1) & (2 * delay_size - 1);
                }

                // Stage 11: delay = 1
                {
                    constexpr int delay_size = 1;
                    constexpr int hh = 1;
                    complex_double delayed = delay11[0];
                    
                    if ((cnt11 & delay_size) == 0) {
                        delay11[0] = x;
                        x = delayed;
                    } else {
                        complex_double twiddle(IFFT_4K_TWIDDLE_REAL[hh], 
                                               IFFT_4K_TWIDDLE_IMAG[hh]);
                        complex_double u = delayed;
                        complex_double v = x;
                        delay11[0] = u + v;
                        x = (u - v) * twiddle;
                    }
                    cnt11 = (cnt11 + 1) & (2 * delay_size - 1);
                }

                // Write output after pipeline latency
                if (iter >= POLY_N - 1) {
                    int out_iter = iter - (POLY_N - 1);
                    int out_lane = out_iter % LANES;
                    
                    if (out_lane == 0) {
                        // Start accumulating output block
                        delay0[0] = x;  // Temporarily store
                    } else if (out_lane == 1) {
                        delay0[1] = x;
                    } else if (out_lane == 2) {
                        delay0[2] = x;
                    } else {
                        // Write complete block
                        encoding_block out;
                        out.element0 = delay0[0];
                        out.element1 = delay0[1];
                        out.element2 = delay0[2];
                        out.element3 = x;
                        IFFTToScaleReducePipes::write(out);
                    }
                }
            }
        });
    }
};

}
