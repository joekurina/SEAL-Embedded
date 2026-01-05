// ------------------------------------------------------------------------- 
// High Level Design Compiler for Intel(R) FPGAs Version 24.1 (Release Build #1f326d321c)
// Software model created on 2026-01-05 14:38:50
// Generation mode: Bit Accurate
// ------------------------------------------------------------------------- 
#ifndef SOFTWARE_MODEL_WRAPPER_THE_FFT_H_
#define SOFTWARE_MODEL_WRAPPER_THE_FFT_H_
#include "fft_example_DUT.h"

#ifndef NO_SYCL
#include <sycl/sycl.hpp>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
    #define CSL_PACKED( struct_def ) __pragma( pack(push, 1) ) struct_def __pragma( pack(pop) )
#else
    #define CSL_PACKED( struct_def ) struct_def __attribute__((__packed__))
#endif

CSL_PACKED(typedef struct
{
    int8_t port_v_in_s;
    int8_t port_channel_in_s;
    double port_data_in_0re;
    double port_data_in_0im;
    double port_data_in_1re;
    double port_data_in_1im;
    double port_data_in_2re;
    double port_data_in_2im;
    double port_data_in_3re;
    double port_data_in_3im;

}) the_fft_input_t;

CSL_PACKED(typedef struct
{
    int8_t port_v_out_s;
    double port_data_out_0re;
    double port_data_out_0im;
    double port_data_out_1re;
    double port_data_out_1im;
    double port_data_out_2re;
    double port_data_out_2im;
    double port_data_out_3re;
    double port_data_out_3im;

}) the_fft_output_t;

#ifdef FPGA_EMULATOR
#ifdef NO_SYCL
the_fft_output_t the_fft(fft_example_DUT* instance, the_fft_input_t input);
#else
SYCL_EXTERNAL the_fft_output_t the_fft(fft_example_DUT* instance, the_fft_input_t input);
#endif
#else
SYCL_EXTERNAL the_fft_output_t the_fft(the_fft_input_t input);
#endif

#ifdef __cplusplus
}
#endif 

#endif // SOFTWARE_MODEL_WRAPPER_THE_FFT_H_
