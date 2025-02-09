#include "lib_fft_rtl.hpp"

// This C++ model is only used during emulation, so it should functionally
// match the RTL in lib_fft_rtl.v.

SYCL_EXTERNAL extern "C" std::complex<double> rtl_fft(std::complex<double> fft_input)
{
    // This function is a model of the RTL FFT block. It is used during emulation
    // to provide a cycle-accurate model of the RTL. It should match the RTL
    // implementation in lib_fft_rtl.v.

    // This is a simple pass-through model that just returns the input.
    return fft_input;
}