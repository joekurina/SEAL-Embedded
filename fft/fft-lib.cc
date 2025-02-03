// fft-lib.cc
// Implementation of FFT functions for the FFT library.
// Bridges between C99 _Complex double and C++ std::complex<double>.
#include "fft_lib.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cassert>
#include <cmath>
#include <complex>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Bit reversal helper function.
// Typical 16-bit reversal approach.
size_t bitrev(size_t input, size_t numbits)
{
    size_t t = (((input & 0xaaaa) >> 1) | ((input & 0x5555) << 1));
    t        = (((t & 0xcccc) >> 2) | ((t & 0x3333) << 2));
    t        = (((t & 0xf0f0) >> 4) | ((t & 0x0f0f) << 4));
    t        = (((t & 0xff00) >> 8) | ((t & 0x00ff) << 8));
    return (numbits == 0) ? 0 : (t >> (16 - numbits));
}

// If <complex.h> isn’t included in C++ mode, define creal and cimag.
#ifndef creal
#  define creal(c) __real__(c)
#endif
#ifndef cimag
#  define cimag(c) __imag__(c)
#endif

// Convert from C++ std::complex<double> to C99 _Complex double.
static inline _Complex double to_c99(const std::complex<double>& z)
{
    _Complex double tmp = 0;  // initialize to 0+0i
    __real__ tmp = z.real();
    __imag__ tmp = z.imag();
    return tmp;
}

// Convert from C99 _Complex double to C++ std::complex<double>.
static inline std::complex<double> from_c99(_Complex double c)
{
    return std::complex<double>(creal(c), cimag(c));
}


namespace {
  // For this example we support only an 8-point FFT.
  constexpr size_t FIXED_N = 8;
  constexpr size_t FIXED_LOGN = 3;
}

//-----------------------------------------------------------------------------
// Define SYCL pipe types for streaming FFT data between host and kernel.
//-----------------------------------------------------------------------------
class IDPipeFFTIn;
using InputPipeFFT = sycl::ext::intel::experimental::pipe<IDPipeFFTIn, std::complex<double>>;

class IDPipeFFTOut;
using OutputPipeFFT = sycl::ext::intel::experimental::pipe<IDPipeFFTOut, std::complex<double>>;


//-----------------------------------------------------------------------------
// Forward-declare the kernel name to reduce name mangling.
//-----------------------------------------------------------------------------
class KernelFFT;

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
}

// Compute the k-th root of unity on the fly.
static std::complex<double> calc_root_otf(size_t k, size_t m)
{
    double angle = 2.0 * M_PI * static_cast<double>(k) / static_cast<double>(m);
    return std::complex<double>(std::cos(angle), std::sin(angle));
}

// Compute FFT roots for forward FFT.
void calc_fft_roots(size_t n, size_t logn, fft_complex* roots)
{
    assert(n >= 4 && roots);

    size_t m = (n << 1);
    for (size_t i = 0; i < n; i++)
    {
        // Optionally, perform bit reversal on the index.
        size_t br = bitrev(i, logn);
        std::complex<double> z = calc_root_otf(br, m);
        roots[i] = to_c99(z);
    }
}

// Compute FFT roots for inverse FFT.
void calc_ifft_roots(size_t n, size_t logn, fft_complex* ifft_roots)
{
    assert(n >= 4 && ifft_roots);

    size_t m = (n << 1);
    for (size_t i = 0; i < n; i++)
    {
        // Calculate the conjugate of the forward FFT root (or adjust bit-reversal as needed).
        std::complex<double> z = calc_root_otf((i - 1), m);
        z = std::conj(z);
        ifft_roots[i] = to_c99(z);
    }
}

// In-place inverse FFT implementation.
void ifft_inpl(fft_complex* vec, size_t n, size_t logn, const fft_complex* roots)
{
    // 1) Copy the input array from _Complex double to std::complex<double>.
    std::vector<std::complex<double>> data(n);
    for (size_t i = 0; i < n; i++)
        data[i] = from_c99(vec[i]);

    // 2) Copy roots if provided.
    std::vector<std::complex<double>> r(n);
    if (roots)
    {
        for (size_t i = 0; i < n; i++)
            r[i] = from_c99(roots[i]);
    }

    // Reset root index for each call.
    size_t root_idx = 1;

    // 3) Perform the inverse FFT (IFFT) using the “Harvey butterfly” approach.
    size_t tt = 1, h = n / 2;
    for (size_t round = 0; round < logn; round++, tt *= 2, h /= 2)
    {
        for (size_t j = 0, kstart = 0; j < h; j++, kstart += 2 * tt)
        {
            std::complex<double> s;
            if (roots)
            {
                // Use the precomputed roots.
                s = r[root_idx++];
            }
            else
            {
                // Compute on the fly.
                size_t br = bitrev(h + j, logn);
                s = std::conj(calc_root_otf(br, n << 1));
            }

            for (size_t k = kstart; k < kstart + tt; k++)
            {
                auto u = data[k];
                auto w = data[k + tt];
                data[k]    = u + w;
                data[k+tt] = (u - w) * s;
            }
        }
    }

    // 4) Copy the result back to the caller’s array.
    for (size_t i = 0; i < n; i++)
        vec[i] = to_c99(data[i]);
}

//-----------------------------------------------------------------------------
// Public interface function: fft_inpl
//
// This function conforms to the original signature and uses SYCL to perform
// the FFT. It creates its own SYCL queue, streams the input via a pipe to a
// kernel that performs the FFT, and then reads the results back.
//-----------------------------------------------------------------------------
extern "C" void fft_inpl(fft_complex* vec, size_t n, size_t logn, const fft_complex* roots) {
  // For this SYCL-based implementation, we require n == 8 and logn == 3.
  assert(n == FIXED_N);
  assert(logn == FIXED_LOGN);
  (void)roots;  // The roots parameter is ignored in this implementation.

  // Create a SYCL queue internally.
  sycl::queue q{ sycl::default_selector{} };

  // Write input data into the input pipe.
  // Convert each fft_complex value to std::complex<double>.
  for (size_t i = 0; i < n; ++i) {
    std::complex<double> value = from_c99(vec[i]);
    InputPipeFFT::write(q, value);
  }

  // Launch the FFT kernel as a single_task.
  q.single_task<KernelFFT>(FFTKernelFunctor{}).wait();

  // Read the FFT result from the output pipe.
  // Convert each std::complex<double> back to fft_complex.
  for (size_t i = 0; i < n; ++i) {
    std::complex<double> value = OutputPipeFFT::read(q);
    vec[i] = to_c99(value);
  }
}

#ifdef __cplusplus
} // extern "C"
#endif
