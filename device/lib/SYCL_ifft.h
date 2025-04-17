#pragma once

#include "SYCL_ckks_sym.h"
#include "SYCL_pipes.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

class IFFTKernel {
private:
    size_t n;
    size_t logn;
    mutable sycl::buffer<std::complex<double>, 1> encoding_acc;
    mutable sycl::buffer<int8_t, 1> error_samples_acc;

public:
    IFFTKernel(size_t n_val, size_t logn_val,
                sycl::buffer<std::complex<double>, 1>& encoding_buf,
                sycl::buffer<int8_t, 1>& error_samples_buf)
        :   n(n_val), logn(logn_val), 
            encoding_acc(encoding_buf), 
            error_samples_acc(error_samples_buf) {}
    
    void operator()(sycl::handler& h) const {
        // Get access to the buffers
        auto encoding = encoding_acc.get_access<sycl::access::mode::read>(h);
        auto error_samples = error_samples_acc.get_access<sycl::access::mode::read>(h);
        
        // Create a stream for debugging
        sycl::stream kernel_dbg_stream(1024 * 4, 256, h); // Debug stream

        // Capture kernel variables
        size_t kernel_n = n;
        size_t kernel_logn = logn;

        
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            kernel_dbg_stream << "IFFTKernel: Starting..." << sycl::endl; // Debug message
            
            // Local array to store input data
            std::complex<double> encoding_local[4096];
            
            // Read data directly from buffer
            for (size_t i = 0; i < kernel_n; i++) {
                encoding_local[i] = encoding[i];
            }

            kernel_dbg_stream << "IFFTKernel: Data loaded." << sycl::endl; // Debug message
            
            // Bit-reversal function 
            auto bitrev = [](size_t input, size_t numbits) -> size_t {
                size_t t = (((input & 0xaaaa) >> 1) | ((input & 0x5555) << 1));
                t        = (((t & 0xcccc) >> 2) | ((t & 0x3333) << 2));
                t        = (((t & 0xf0f0) >> 4) | ((t & 0x0f0f) << 4));
                t        = (((t & 0xff00) >> 8) | ((t & 0x00ff) << 8));
                return (numbits == 0) ? 0 : (t >> (16 - numbits));
            };
            
            // Root calculation function
            auto calc_root_otf = [](size_t k, size_t m) -> std::complex<double> {
                double angle = 2.0 * M_PI * static_cast<double>(k) / static_cast<double>(m);
                return std::complex<double>(sycl::cos(angle), sycl::sin(angle));
            };
            
            // IFFT implementation 
            size_t tt = 1, h = kernel_n / 2;
            
            for (size_t i = 0; i < kernel_logn; i++, tt *= 2, h /= 2) {
                for (size_t j = 0, kstart = 0; j < h; j++, kstart += 2 * tt) {
                    std::complex<double> s;
                    size_t br = bitrev(h + j, kernel_logn);
                    s = std::conj(calc_root_otf(br, kernel_n << 1));
                    
                    for (size_t k = kstart; k < kstart + tt; k++) {
                        std::complex<double> u = encoding_local[k];
                        std::complex<double> v = encoding_local[k + tt];
                        encoding_local[k]      = u + v;
                        encoding_local[k + tt] = (u - v) * s;
                    }
                }
            }
            
            kernel_dbg_stream << "IFFTKernel: Computation finished. Writing to pipes..." << sycl::endl; // Debug message

            // Pass both the transformed values and error samples through pipes
            for (size_t i = 0; i < kernel_n; i++) {
                // Write transformed encoding values to pipe
                IFFTToScaleAndConvertPipe::write(encoding_local[i]);
                
                // Also pass the error samples through to the next kernel
                IFFTErrorToScaleAndConvertPipe::write(error_samples[i]);
            }

            kernel_dbg_stream << "IFFTKernel: Finished writing to pipes." << sycl::endl; // Debug message
        });
    }
};