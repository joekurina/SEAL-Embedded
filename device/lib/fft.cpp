#include "fft.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cassert>
#include <cmath>
#include <complex>
#include <complex.h>

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
// Forward-declare the kernel name to reduce name mangling.
//-----------------------------------------------------------------------------
class KernelFFT;
class KernelIFFT;

//-----------------------------------------------------------------------------
// Kernel functor that implements the in-place FFT using SYCL pipes.
// This kernel reads FIXED_N elements from the input pipe, computes an FFT
// (using a decimation-in-time Cooley–Tukey algorithm), and writes the result
// to the output pipe. (The result is in bit-reversed order.)
//
// Note: This implementation ignores the precomputed roots (the twiddle
// factors are computed on the fly).
//-----------------------------------------------------------------------------
struct FFTKernelFunctor {
  void operator()() const {
    std::complex<double> data[FIXED_N];

    // Read FIXED_N values from the input pipe.
    for (size_t i = 0; i < FIXED_N; ++i) {
      data[i] = InputPipeFFT::read();
    }

    // Perform in-place FFT.
    for (size_t stage = 0; stage < FIXED_LOGN; stage++) {
      size_t m = 1 << (stage + 1);  // group size for this stage
      for (size_t k = 0; k < FIXED_N; k += m) {
        for (size_t j = 0; j < m / 2; j++) {
          double angle = -2.0 * M_PI * j / m;
          std::complex<double> twiddle(std::cos(angle), std::sin(angle));
          std::complex<double> t = twiddle * data[k + j + m / 2];
          std::complex<double> u = data[k + j];
          data[k + j] = u + t;
          data[k + j + m / 2] = u - t;
        }
      }
    }

    // Write the FFT result to the output pipe.
    for (size_t i = 0; i < FIXED_N; ++i) {
      OutputPipeFFT::write(data[i]);
    }
  }
};

//-----------------------------------------------------------------------------
// Kernel functor that implements the in-place IFFT using SYCL pipes.
//-----------------------------------------------------------------------------
struct IFFTKernelFunctor {
  void operator()() const {
    std::complex<double> data[FIXED_N];

    // Read FIXED_N values from the IFFT input pipe.
    for (size_t i = 0; i < FIXED_N; ++i) {
      data[i] = InputPipeIFFT::read();
    }

    // Perform in-place IFFT.
    for (size_t stage = 0; stage < FIXED_LOGN; stage++) {
      size_t m = 1 << (stage + 1);  // group size for this stage
      for (size_t k = 0; k < FIXED_N; k += m) {
        for (size_t j = 0; j < m / 2; j++) {
          double angle = 2.0 * M_PI * j / m;
          std::complex<double> twiddle(std::cos(angle), std::sin(angle));
          std::complex<double> t = twiddle * data[k + j + m / 2];
          std::complex<double> u = data[k + j];
          data[k + j] = u + t;
          data[k + j + m / 2] = u - t;
        }
      }
    }

    // Normalize the IFFT result.
    for (size_t i = 0; i < FIXED_N; ++i) {
      data[i] /= FIXED_N;
    }

    // Write the IFFT result to the output pipe.
    for (size_t i = 0; i < FIXED_N; ++i) {
      OutputPipeIFFT::write(data[i]);
    }
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
  // For this SYCL-based implementation, we require n == 4096 and logn == 12.
  assert(n == FIXED_N);
  assert(logn == FIXED_LOGN);
  (void)roots;  // The roots parameter is ignored in this implementation.

  // Create a SYCL queue internally.
  sycl::queue q{ sycl::default_selector{} };

  // Write input data into the input pipe.
  // Convert each fft_complex value to std::complex<double>.
  for (size_t i = 0; i < n; ++i) {
    std::complex<double> value = from_c(vec[i]);
    InputPipeFFT::write(q, value);
  }

  // Launch the FFT kernel as a single_task.
  q.single_task<KernelFFT>(FFTKernelFunctor{}).wait();

  // Read the FFT result from the output pipe.
  // Convert each std::complex<double> back to fft_complex.
  for (size_t i = 0; i < n; ++i) {
    std::complex<double> value = OutputPipeFFT::read(q);
    vec[i] = to_c(value);
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
  // For this SYCL-based implementation, we require n == 4096 and logn == 12.
  assert(n == FIXED_N);
  assert(logn == FIXED_LOGN);
  (void)roots;  // The roots parameter is ignored in this implementation.

  // Create a SYCL queue internally.
  sycl::queue q{ sycl::default_selector{} };

  // Write input data into the IFFT input pipe.
  for (size_t i = 0; i < n; ++i) {
    std::complex<double> value = from_c(vec[i]);
    InputPipeIFFT::write(q, value);
  }

  // Launch the IFFT kernel as a single_task.
  q.single_task<KernelIFFT>(IFFTKernelFunctor{}).wait();

  // Read the IFFT result from the IFFT output pipe.
  for (size_t i = 0; i < n; ++i) {
    std::complex<double> value = OutputPipeIFFT::read(q);
    vec[i] = to_c(value);
  }
}