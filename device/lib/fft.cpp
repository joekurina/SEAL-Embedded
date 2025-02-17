#include "fft.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>

// Ensure M_PI is defined.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// If code calls bitrev(...) from outside, we must define it here.
size_t bitrev(size_t input, size_t numbits)
{
    // typical 16-bit reversal approach:
    size_t t = (((input & 0xaaaa) >> 1) | ((input & 0x5555) << 1));
    t        = (((t & 0xcccc) >> 2) | ((t & 0x3333) << 2));
    t        = (((t & 0xf0f0) >> 4) | ((t & 0x0f0f) << 4));
    t        = (((t & 0xff00) >> 8) | ((t & 0x00ff) << 8));
    return (numbits == 0) ? 0 : (t >> (16 - numbits));
}

namespace {
    // Now supporting a 4096-point FFT.
    constexpr size_t FIXED_N = 4096;
    constexpr size_t FIXED_LOGN = 12;
    
    static inline std::complex<double> calc_root_otf(size_t k, size_t m)
    {
        double angle = 2.0 * M_PI * static_cast<double>(k) / static_cast<double>(m);
        return std::complex<double>(cos(angle), sin(angle));
    }

    //--------------------------------------------------------------------------
    // SYCL BUFFER FFT Implementation
    //
    // This function performs an in-place FFT on the array `vec` of size `n`
    // using `logn` stages. It uses on-the-fly computation of the twiddle factors.
    //--------------------------------------------------------------------------
    void fft(fft_complex* vec, size_t n, size_t logn) {
        [[intel::fpga_register]] size_t h = 1, tt = n / 2;
        
        #pragma unroll
        for (size_t round = 0; round < FIXED_LOGN; round++, h *= 2, tt /= 2) {
            #pragma unroll 4
            for (size_t j = 0, kstart = 0; j < h; j++, kstart += 2 * tt) {
                [[intel::fpga_register]] std::complex<double> s;
                size_t br = bitrev(h + j, logn);
                s = calc_root_otf(br, n << 1);
                
                #pragma unroll 4
                for (size_t k = kstart; k < kstart + tt; k++) {
                    std::complex<double> u = vec[k];
                    std::complex<double> w = vec[k + tt] * s;
                    vec[k]    = u + w;
                    vec[k+tt] = u - w;
                }
            }
        }
    }

    //---------------------------------------------------------------------------
    // SYCL BUFFER IFFT Implementation
    //
    // Performs an in-place IFFT on the array `vec` of size `n` using `logn`
    // stages. The on-the-fly approach is used, and the twiddle factors are
    // conjugated.
    //---------------------------------------------------------------------------
    void ifft(fft_complex* vec, size_t n, size_t logn) {
        [[intel::fpga_register]] size_t tt = 1, h = n / 2;
        
        #pragma unroll
        for (size_t round = 0; round < FIXED_LOGN; round++, tt *= 2, h /= 2) {
            #pragma unroll 4
            for (size_t j = 0, kstart = 0; j < h; j++, kstart += 2 * tt) {
                [[intel::fpga_register]] std::complex<double> s;
                size_t br = bitrev(h + j, logn);
                s = std::conj(calc_root_otf(br, n << 1));
                
                #pragma unroll 4 
                for (size_t k = kstart; k < kstart + tt; k++) {
                    std::complex<double> u = vec[k];
                    std::complex<double> w = vec[k + tt];
                    vec[k]    = u + w;
                    vec[k+tt] = (u - w) * s;
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
// Public interface function: fft_inpl (using buffers)
//-----------------------------------------------------------------------------
//
// Wraps the host array `vec` in a SYCL buffer, then submits a kernel that
// calls `fft` on the entire array. When the buffer is destructed after
// kernel completion, the host memory is updated with the computed FFT.
extern "C" void fft_inpl(fft_complex* vec, size_t n, size_t logn, const fft_complex* roots) {
    // Enforce fixed-size expectations.
    assert(n == FIXED_N);
    assert(logn == FIXED_LOGN);
    (void)roots;  // Precomputed roots are ignored in this implementation.

    // Select a device; here we use the FPGA emulator selector.
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;

    try {
        sycl::queue q{selector};
        auto device = q.get_device();
        std::cout << "Running FFT on device: "
                  << device.get_info<sycl::info::device::name>().c_str()
                  << std::endl;

        // Create a SYCL buffer that wraps the host array.
        sycl::buffer<fft_complex, 1> buf(vec, sycl::range<1>(n));

        // Submit a kernel that calls fft on the entire array.
        q.submit([&](sycl::handler& h) {
            // Create a read-write accessor for the buffer.
            auto data = buf.get_access<sycl::access::mode::read_write>(h);
            h.single_task<class FFTKernelBuffer>([=]() {
                // Call the SYCL BUFFER FFT implementation on the device.
                fft(data.get_pointer(), n, logn);
            });
        });
        q.wait();  // Wait for kernel completion.
        // When the buffer goes out of scope, the host memory `vec` is updated.
    } catch (sycl::exception const& e) {
        std::cerr << "Caught a synchronous SYCL exception in fft_inpl: " 
                  << e.what() << "\n";
        std::exit(1);
    }
}

//-----------------------------------------------------------------------------
// Public interface function: ifft_inpl (using buffers)
//-----------------------------------------------------------------------------
//
// Wraps the host array `vec` in a SYCL buffer, then submits a kernel that
// calls `ifft` on the entire array. When the buffer is destructed after
// kernel completion, the host memory is updated with the computed IFFT.
extern "C" void ifft_inpl(fft_complex* vec, size_t n, size_t logn, const fft_complex* roots) {
    // Enforce fixed-size expectations.
    assert(n == FIXED_N);
    assert(logn == FIXED_LOGN);
    (void)roots;  // Precomputed roots are ignored.

    auto selector = sycl::ext::intel::fpga_emulator_selector_v;

    try {
        sycl::queue q{selector};
        auto device = q.get_device();
        std::cout << "Running IFFT on device: "
                  << device.get_info<sycl::info::device::name>().c_str()
                  << std::endl;

        // Create a buffer that wraps the host array.
        sycl::buffer<fft_complex, 1> buf(vec, sycl::range<1>(n));

        // Submit a kernel that calls ifft on the entire array.
        q.submit([&](sycl::handler& h) {
            auto data = buf.get_access<sycl::access::mode::read_write>(h);
            h.single_task<class IFFTKernelBuffer>([=]() {
                ifft(data.get_pointer(), n, logn);
            });
        });
        q.wait();
    } catch (sycl::exception const& e) {
        std::cerr << "Caught a synchronous SYCL exception in ifft_inpl: " 
                  << e.what() << "\n";
        std::exit(1);
    }
}