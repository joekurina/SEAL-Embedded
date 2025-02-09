#include <sycl/sycl.hpp>
#include <complex>

SYCL_EXTERNAL extern "C" std::complex<double> rtl_fft(std::complex<double> fft_input);