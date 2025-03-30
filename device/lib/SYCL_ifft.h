#pragma once

#include "SYCL_ckks_sym.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// IFFT Kernel functor class
class IFFTKernel {
private:
    size_t n;
    size_t logn;
    mutable sycl::buffer<complex_double, 1> encoding_acc;

public:
    IFFTKernel(size_t n_val, size_t logn_val, sycl::buffer<complex_double, 1>& encoding_buf)
        : n(n_val), logn(logn_val), encoding_acc(encoding_buf) {}
    
    void operator()(sycl::handler& h) const {
        // Get access to the buffer
        auto encoding = encoding_acc.get_access<sycl::access::mode::read_write>(h);
        
        // Capture necessary variables
        size_t kernel_n = n;
        size_t kernel_logn = logn;
        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // Bit-reversal function
            auto bitrev = [](size_t input, size_t numbits) -> size_t {
                size_t t = (((input & 0xaaaa) >> 1) | ((input & 0x5555) << 1));
                t        = (((t & 0xcccc) >> 2) | ((t & 0x3333) << 2));
                t        = (((t & 0xf0f0) >> 4) | ((t & 0x0f0f) << 4));
                t        = (((t & 0xff00) >> 8) | ((t & 0x00ff) << 8));
                return (numbits == 0) ? 0 : (t >> (16 - numbits));
            };
            
            // Root calculation function
            auto calc_root_otf = [](size_t k, size_t m) -> complex_double {
                double angle = 2.0 * M_PI * static_cast<double>(k) / static_cast<double>(m);
                return complex_double(sycl::cos(angle), sycl::sin(angle));
            };
            
            // IFFT implementation
            size_t tt = 1, h = kernel_n / 2;
            
            for (size_t round = 0; round < kernel_logn; round++, tt *= 2, h /= 2) {
                for (size_t j = 0, kstart = 0; j < h; j++, kstart += 2 * tt) {
                    complex_double s;
                    size_t br = bitrev(h + j, kernel_logn);
                    s = std::conj(calc_root_otf(br, kernel_n << 1));
                    
                    for (size_t k = kstart; k < kstart + tt; k++) {
                        complex_double u = encoding[k];
                        complex_double v = encoding[k + tt];
                        encoding[k]      = u + v;
                        encoding[k + tt] = (u - v) * s;
                    }
                }
            }
        });
    }
};