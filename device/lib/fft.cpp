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
// Public interface function: fft_inpl (using buffers)
//-----------------------------------------------------------------------------
//
// This function creates a SYCL buffer from the input array, submits a kernel
// that calls rtl_fft on the entire array via a pointer obtained from a read-write
// accessor, and then (after kernel completion) the buffer’s destructor updates the
// host memory with the computed FFT in place.
extern "C" void fft_inpl(fft_complex* vec, size_t n, size_t logn, const fft_complex* roots) {
    // Enforce fixed size expectations.
    assert(n == FIXED_N);
    assert(logn == FIXED_LOGN);
    (void)roots;  // precomputed roots are ignored in this implementation

    // Choose a device; here we use the FPGA emulator selector.
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;

    try {
        sycl::queue q{ selector };

        // Create a SYCL buffer wrapping the host memory.
        sycl::buffer<fft_complex, 1> buf(vec, sycl::range<1>(n));

        // Submit a kernel that will call rtl_fft on the entire array.
        q.submit([&](sycl::handler& h) {
            // Create an accessor with read-write permissions.
            auto data = buf.get_access<sycl::access::mode::read_write>(h);
            h.single_task<class FFTKernelBuffer>([=]() {
                // Obtain a pointer to the data from the accessor and pass it to rtl_fft.
                // (This assumes that your implementation of get_pointer() returns a valid device pointer.)
                rtl_fft(data.get_pointer());
            });
        });
        q.wait();  // Wait for kernel completion.
        // When the buffer goes out of scope, the host memory (vec) is updated with the results.
    } catch (sycl::exception const& e) {
        std::cerr << "Caught a synchronous SYCL exception in fft_inpl: " << e.what() << "\n";
        std::exit(1);
    }
}

//-----------------------------------------------------------------------------
// Public interface function: ifft_inpl (using buffers)
//-----------------------------------------------------------------------------
//
// This function creates a SYCL buffer from the input array, submits a kernel
// that calls rtl_ifft on the entire array, and then (upon kernel completion)
// the host memory is updated with the computed IFFT in place.
extern "C" void ifft_inpl(fft_complex* vec, size_t n, size_t logn, const fft_complex* roots) {
    // Enforce fixed size expectations.
    assert(n == FIXED_N);
    assert(logn == FIXED_LOGN);
    (void)roots;  // precomputed roots are ignored

    auto selector = sycl::ext::intel::fpga_emulator_selector_v;

    try {
        sycl::queue q{ selector };

        // Create a buffer that wraps the host memory.
        sycl::buffer<fft_complex, 1> buf(vec, sycl::range<1>(n));

        // Submit a kernel that calls rtl_ifft on the entire array.
        q.submit([&](sycl::handler& h) {
            auto data = buf.get_access<sycl::access::mode::read_write>(h);
            h.single_task<class IFFTKernelBuffer>([=]() {
                rtl_ifft(data.get_pointer());
            });
        });
        q.wait();
        // The buffer's destructor ensures that vec is updated with the computed data.
    } catch (sycl::exception const& e) {
        std::cerr << "Caught a synchronous SYCL exception in ifft_inpl: " << e.what() << "\n";
        std::exit(1);
    }
}