#include "fft.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cassert>
#include <cmath>
#include <complex>
#include <complex.h>

#include "lib_fft_rtl.hpp"
#include "lib_ifft_rtl.hpp"

#include "exception_handler.hpp"
#include <stdint.h>

// Ensure M_PI is defined.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
}

// Forward declarations for the RTL functions that operate on an entire array.
extern "C" void rtl_fft(fft_complex* vec);
extern "C" void rtl_ifft(fft_complex* vec);

//-----------------------------------------------------------------------------
// Public interface function: fft_inpl (using USM)
//-----------------------------------------------------------------------------
//
// This function copies the input data into USM shared memory,
// launches a kernel that calls rtl_fft on the entire array,
// and then copies the results back into the original array.
extern "C" void fft_inpl(fft_complex* vec, size_t n, size_t logn, const fft_complex* roots) {
    // This implementation assumes a fixed FFT size.
    assert(n == FIXED_N);
    assert(logn == FIXED_LOGN);
    (void)roots;  // precomputed roots are ignored

    // Choose a device. Here we use the FPGA emulator selector.
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;

    try {
        sycl::queue q{selector};

        // Allocate USM shared memory for n fft_complex elements.
        fft_complex* usm_data = sycl::malloc_shared<fft_complex>(n, q);
        if (usm_data == nullptr) {
            std::cerr << "Failed to allocate USM shared memory.\n";
            std::exit(1);
        }

        // Copy the input data from vec into the USM memory.
        for (size_t i = 0; i < n; ++i) {
            usm_data[i] = vec[i];
        }

        // Submit a single-task kernel that performs the FFT on the entire array.
        q.submit([&](sycl::handler& h) {
            h.single_task<class FFTKernelUSM>([=]() {
                // rtl_fft processes the entire array in place.
                rtl_fft(usm_data);
            });
        });
        q.wait();  // Wait for the kernel to finish.

        // Copy the results from USM memory back to the host array.
        for (size_t i = 0; i < n; ++i) {
            vec[i] = usm_data[i];
        }

        // Free the USM memory.
        sycl::free(usm_data, q);
    } catch (sycl::exception const& e) {
        std::cerr << "Caught a synchronous SYCL exception in fft_inpl: " << e.what() << "\n";
        std::exit(1);
    }
}

//-----------------------------------------------------------------------------
// Public interface function: ifft_inpl (using USM)
//-----------------------------------------------------------------------------
//
// This function copies the input data into USM shared memory,
// launches a kernel that calls rtl_ifft on the entire array,
// and then copies the results back into the original array.
extern "C" void ifft_inpl(fft_complex* vec, size_t n, size_t logn, const fft_complex* roots) {
    // This implementation assumes a fixed FFT size.
    assert(n == FIXED_N);
    assert(logn == FIXED_LOGN);
    (void)roots;  // precomputed roots are ignored

    auto selector = sycl::ext::intel::fpga_emulator_selector_v;

    try {
        sycl::queue q{selector};

        // Allocate USM shared memory for n fft_complex elements.
        fft_complex* usm_data = sycl::malloc_shared<fft_complex>(n, q);
        if (usm_data == nullptr) {
            std::cerr << "Failed to allocate USM shared memory.\n";
            std::exit(1);
        }

        // Copy the input data from vec into the USM memory.
        for (size_t i = 0; i < n; ++i) {
            usm_data[i] = vec[i];
        }

        // Submit a single-task kernel that performs the IFFT on the entire array.
        q.submit([&](sycl::handler& h) {
            h.single_task<class IFFTKernelUSM>([=]() {
                // rtl_ifft processes the entire array in place.
                rtl_ifft(usm_data);
            });
        });
        q.wait();  // Wait for the kernel to finish.

        // Copy the results from USM memory back to the host array.
        for (size_t i = 0; i < n; ++i) {
            vec[i] = usm_data[i];
        }

        // Free the USM memory.
        sycl::free(usm_data, q);
    } catch (sycl::exception const& e) {
        std::cerr << "Caught a synchronous SYCL exception in ifft_inpl: " << e.what() << "\n";
        std::exit(1);
    }
}