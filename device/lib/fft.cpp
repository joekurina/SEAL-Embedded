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

//-----------------------------------------------------------------------------
// Forward-declare the kernel name to reduce name mangling.
//-----------------------------------------------------------------------------
class KernelFFT;
class KernelIFFT;

//-----------------------------------------------------------------------------
// Define SYCL pipe types for streaming FFT data between host and kernel.
//-----------------------------------------------------------------------------
class IDPipeFFTIn;
using InputPipeFFT = sycl::ext::intel::experimental::pipe<IDPipeFFTIn, std::complex<double>>;

class IDPipeFFTOut;
using OutputPipeFFT = sycl::ext::intel::experimental::pipe<IDPipeFFTOut, std::complex<double>>;

class IDPipeIFFTIn;
using InputPipeIFFT = sycl::ext::intel::experimental::pipe<IDPipeIFFTIn, std::complex<double>>;

class IDPipeIFFTOut;
using OutputPipeIFFT = sycl::ext::intel::experimental::pipe<IDPipeIFFTOut, std::complex<double>>;

//-----------------------------------------------------------------------------
// Kernel functor that implements the in-place FFT using SYCL pipes.
// This kernel reads FIXED_N elements from the input pipe, computes an FFT
// (using a decimation-in-time Cooley–Tukey algorithm), and writes the result
// to the output pipe. (The result is in bit-reversed order.)
//
// Note: This implementation ignores the precomputed roots (the twiddle
// factors are computed on the fly).
//-----------------------------------------------------------------------------
template <typename PipeIn, typename PipeOut>
struct FFTKernelFunctor {
    // use a streaming pipelined invocation interface to minimize hardware
    // overhead
    auto get(sycl::ext::oneapi::experimental::properties_tag) {
        return sycl::ext::oneapi::experimental::properties{
            sycl::ext::intel::experimental::streaming_interface_accept_downstream_stall, 
            sycl::ext::intel::experimental::pipelined<1>
        };
    }
    void operator()() const {
        std::complex<double> fft_input = PipeIn::read();
        std::complex<double> fft_output = rtl_fft(fft_input);
        PipeOut::write(fft_output);
    }
};

//-----------------------------------------------------------------------------
// Kernel functor that implements the in-place IFFT using SYCL pipes.
//-----------------------------------------------------------------------------
template <typename PipeIn, typename PipeOut>
struct IFFTKernelFunctor {
    // use a streaming pipelined invocation interface to minimize hardware
    // overhead
    auto get(sycl::ext::oneapi::experimental::properties_tag) {
        return sycl::ext::oneapi::experimental::properties{
            sycl::ext::intel::experimental::streaming_interface_accept_downstream_stall, 
            sycl::ext::intel::experimental::pipelined<1>
        };
    }
    void operator()() const {
        std::complex<double> ifft_input = PipeIn::read();
        std::complex<double> ifft_output = rtl_ifft(ifft_input);
        PipeOut::write(ifft_output);
    }
};

//-----------------------------------------------------------------------------
// Public interface function: fft_inpl
//
// This function conforms to the original signature and uses SYCL to perform
// the FFT. It creates its own SYCL queue, streams the input via a pipe to a
// kernel that performs the FFT, and then reads the results back.
//-----------------------------------------------------------------------------
extern "C" void fft_inpl(fft_complex* vec, size_t n, size_t logn, const fft_complex* roots) {
    std::complex<double> result_fft;
    // For this SYCL-based implementation, we require n == FIXED_N and logn == FIXED_LOGN.
    assert(n == FIXED_N);
    assert(logn == FIXED_LOGN);
    (void)roots;
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;

    try {
        sycl::queue q{ selector };
        auto device = q.get_device();
        //std::cout << "Running on device: "
        //        << device.get_info<sycl::info::device::name>().c_str()
        //        << std::endl;

        for (size_t i = 0; i < n; ++i) {
            std::complex<double> value = from_c(vec[i]);
            InputPipeFFT::write(q, value);
        }

        q.single_task<KernelFFT>(FFTKernelFunctor<InputPipeFFT,OutputPipeFFT>{}).wait();

        for (size_t i = 0; i < n; ++i) {
            result_fft = OutputPipeFFT::read(q);
            vec[i] = to_c(value);
        }

    } catch (sycl::exception const& e) {
        std::cerr << "Caught a synchronous SYCL exception: " << e.what() << std::endl;
        std::exit(1);
    }
}

//-----------------------------------------------------------------------------
// Public interface function: ifft_inpl
//
// This function conforms to the original signature and uses SYCL to perform
// the IFFT. It creates its own SYCL queue, streams the input via a pipe to a
// kernel that performs the FFT, and then reads the results back.
//-----------------------------------------------------------------------------
void ifft_inpl(fft_complex* vec, size_t n, size_t logn, const fft_complex* roots)
{
    std::complex<double> result_ifft;
    // For this SYCL-based implementation, we require n == FIXED_N and logn == FIXED_LOGN.
    assert(n == FIXED_N);
    assert(logn == FIXED_LOGN);
    (void)roots;
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;

    try {
        sycl::queue q{ selector };
        auto device = q.get_device();
        //std::cout << "Running on device: "
        //        << device.get_info<sycl::info::device::name>().c_str()
        //        << std::endl;

        for (size_t i = 0; i < n; ++i) {
            std::complex<double> value = from_c(vec[i]);
            InputPipeIFFT::write(q, value);
        }

        q.single_task<KernelIFFT>(IFFTKernelFunctor<InputPipeIFFT,OutputPipeIFFT>{}).wait();

        // Loops reads the output pipe into the result vector.
        // Result vector is then converted to a C-type and stored in vec.
        for (size_t i = 0; i < n; ++i) {
            result_ifft = OutputPipeIFFT::read(q);
            vec[i] = to_c(result_ifft);
        }

    } catch (sycl::exception const& e) {
        std::cerr << "Caught a synchronous SYCL exception: " << e.what() << std::endl;
        std::exit(1);
    }
}