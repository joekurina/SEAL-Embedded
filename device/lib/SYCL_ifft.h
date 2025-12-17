#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_pipes.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

class IFFTKernel {
private:
    size_t n;
    size_t logn;
    mutable sycl::buffer<encoding_buffer_input, 1> encoding_acc;

public:
    IFFTKernel( size_t n_val, size_t logn_val,
                sycl::buffer<encoding_buffer_input, 1>& encoding_buf)
            :   n(n_val), logn(logn_val), 
                encoding_acc(encoding_buf) {}
    
    void operator()(sycl::handler& h) const {
        // Get access to the buffers
        auto encoding_blocks = encoding_acc.get_access<sycl::access::mode::read>(h);

        // Capture kernel variables
        size_t kernel_n = n;
        size_t kernel_logn = logn;
        
        h.single_task<class IFFTKernel>([=]() [[intel::kernel_args_restrict]] {

            // Bit-reversal function 
            auto bitrev = [](size_t input, size_t numbits) -> size_t 
            {
                size_t t = (((input & 0xaaaa) >> 1) | ((input & 0x5555) << 1));
                t        = (((t & 0xcccc) >> 2) | ((t & 0x3333) << 2));
                t        = (((t & 0xf0f0) >> 4) | ((t & 0x0f0f) << 4));
                t        = (((t & 0xff00) >> 8) | ((t & 0x00ff) << 8));
                return (numbits == 0) ? 0 : (t >> (16 - numbits));
            };
            
            // Root calculation function
            auto calc_root_otf = [](size_t k, size_t m) -> std::complex<double> 
            {
                double angle = 2.0 * M_PI * static_cast<double>(k) / static_cast<double>(m);
                return std::complex<double>(sycl::cos(angle), sycl::sin(angle));
            };
            
            // Unpack 4-lane structs into a flat local array for computation
            std::complex<double> encoding[4096];
            for (size_t blk = 0, idx = 0; blk < kernel_n / 4; ++blk) {
                encoding_buffer_input blk_data = encoding_blocks[blk];
                encoding[idx++] = blk_data.element0;
                encoding[idx++] = blk_data.element1;
                encoding[idx++] = blk_data.element2;
                encoding[idx++] = blk_data.element3;
            }

            // IFFT implementation 
            size_t tt = 1, h = kernel_n / 2;
            
            for (size_t i = 0; i < kernel_logn; i++, tt *= 2, h /= 2) 
            {
                for (size_t j = 0, kstart = 0; j < h; j++, kstart += 2 * tt) 
                {
                    std::complex<double> s;
                    size_t br = bitrev(h + j, kernel_logn);
                    s = std::conj(calc_root_otf(br, kernel_n << 1));
                    
                    for (size_t k = kstart; k < kstart + tt; k++) 
                    {
                        std::complex<double> u = encoding[k];
                        std::complex<double> v = encoding[k + tt];
                        encoding[k]      = u + v;
                        encoding[k + tt] = (u - v) * s;
                    }
                }
            } // End of IFFT computation

            // Pass the transformed values to the pipe in 4-lane structs
            for (size_t i = 0; i < kernel_n; i += 4)
            {
                encoding_buffer_input block{};
                block.element0 = encoding[i + 0];
                block.element1 = encoding[i + 1];
                block.element2 = encoding[i + 2];
                block.element3 = encoding[i + 3];
                IFFTToScaleAndReducePipe::write(block);
            }
        }); // End of single_task
    } // End of operator()
}; // End of IFFTKernel class