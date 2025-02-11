#include "fft.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cassert>
#include <cmath>
#include <complex>
#include <complex.h>

//#include "lib_fft_rtl.hpp"
//#include "lib_ifft_rtl.hpp"

//#include "exception_handler.hpp"
#include <stdint.h>

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

    // Conversion helpers between C99 _Complex double and std::complex<double>
    static inline std::complex<double> from_c(fft_complex c) {
        return std::complex<double>(__real__ c, __imag__ c);
    }

    static inline fft_complex to_c(const std::complex<double>& z) {
        fft_complex tmp = 0;
        __real__ tmp = z.real();
        __imag__ tmp = z.imag();
        return tmp;
    }

    /**
     * Define helper functions to represent the imaginary unit
     * and compute complex conjugate for C style complex numbers.
     */
    static inline _Complex double my_I() 
    {
        typedef union { double d[2]; _Complex double c; } conv;
        conv tmp = { { 0.0, 1.0 } };
        return tmp.c;
    }

    static inline _Complex double my_conj(_Complex double z) 
    {
        return __real__(z) - __imag__(z) * my_I();
    }

    // Basic root calculation using our imaginary unit helper.
    static _Complex double calc_root_otf(size_t k, size_t m)
    {
        double angle = 2.0 * M_PI * (double)k / (double)m;
        return cos(angle) + my_I() * sin(angle);
    }

    //--------------------------------------------------------------------------
    // RTL FFT Implementation
    //
    // This function performs an in-place FFT on the array `vec` of size `n`
    // using `logn` stages. It uses on-the-fly computation of the twiddle factors.
    //--------------------------------------------------------------------------
    void fft_rtl(fft_complex* vec, size_t n, size_t logn) {
        size_t h = 1, tt = n / 2;
        for (size_t round = 0; round < logn; round++, h *= 2, tt /= 2) {
            for (size_t j = 0, kstart = 0; j < h; j++, kstart += 2 * tt) {
                _Complex double s;
                size_t br = bitrev(h + j, logn);
                s = calc_root_otf(br, n << 1);
                for (size_t k = kstart; k < kstart + tt; k++) {
                    _Complex double u = vec[k];
                    _Complex double w = vec[k + tt] * s;
                    vec[k]    = u + w;
                    vec[k+tt] = u - w;
                }
            }
        }
    }

    //---------------------------------------------------------------------------
    // RTL IFFT Implementation
    //
    // Performs an in-place IFFT on the array `vec` of size `n` using `logn`
    // stages. The on-the-fly approach is used, and the twiddle factors are
    // conjugated.
    //---------------------------------------------------------------------------
    void ifft_rtl(fft_complex* vec, size_t n, size_t logn) {
        size_t tt = 1, h = n / 2;
        for (size_t round = 0; round < logn; round++, tt *= 2, h /= 2) {
            for (size_t j = 0, kstart = 0; j < h; j++, kstart += 2 * tt) {
                _Complex double s;
                size_t br = bitrev(h + j, logn);
                s = my_conj(calc_root_otf(br, n << 1));
                for (size_t k = kstart; k < kstart + tt; k++) {
                    _Complex double u = vec[k];
                    _Complex double w = vec[k + tt];
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
// calls `fft_rtl` on the entire array. When the buffer is destructed after
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

        // Submit a kernel that calls fft_rtl on the entire array.
        q.submit([&](sycl::handler& h) {
            // Create a read-write accessor for the buffer.
            auto data = buf.get_access<sycl::access::mode::read_write>(h);
            h.single_task<class FFTKernelBuffer>([=]() {
                // Call the RTL FFT implementation on the device.
                fft_rtl(data.get_pointer(), n, logn);
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
// calls `ifft_rtl` on the entire array. When the buffer is destructed after
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

        // Submit a kernel that calls ifft_rtl on the entire array.
        q.submit([&](sycl::handler& h) {
            auto data = buf.get_access<sycl::access::mode::read_write>(h);
            h.single_task<class IFFTKernelBuffer>([=]() {
                ifft_rtl(data.get_pointer(), n, logn);
            });
        });
        q.wait();
    } catch (sycl::exception const& e) {
        std::cerr << "Caught a synchronous SYCL exception in ifft_inpl: " 
                  << e.what() << "\n";
        std::exit(1);
    }
}