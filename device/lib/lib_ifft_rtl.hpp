#include <sycl/sycl.hpp>
#include <complex>

SYCL_EXTERNAL extern "C" std::complex<double> rtl_ifft(std::complex<double> ifft_input);