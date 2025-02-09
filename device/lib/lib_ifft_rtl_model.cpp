#include "lib_ifft_rtl.hpp"

// This C++ model is only used during emulation, so it should functionally
// match the RTL in lib_ifft_rtl.v.

SYCL_EXTERNAL extern "C" std::complex<double> rtl_ifft(std::complex<double> ifft_input)
{
    // This function is a model of the RTL IFFT block. It is used during emulation
    // to provide a cycle-accurate model of the RTL. It should match the RTL
    // implementation in lib_ifft_rtl.v.

    // This is a simple pass-through model that just returns the input.
    return ifft_input;
}