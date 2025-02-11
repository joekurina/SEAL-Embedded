#include <sycl/sycl.hpp>
#include "complex.h"

using fft_complex = double _Complex;

SYCL_EXTERNAL extern "C" fft_complex rtl_fft( fft_input);