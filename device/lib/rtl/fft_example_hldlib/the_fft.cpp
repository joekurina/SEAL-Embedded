// ------------------------------------------------------------------------- 
// High Level Design Compiler for Altera(R) FPGAs Version 25.1 (Release Build #6a12354d2f)
// Software model created on 2025-07-08 08:39:19
// Generation mode: Bit Accurate
// ------------------------------------------------------------------------- 
#ifndef CSL_USE_GMP
    #define CSL_USE_GMP
#endif
#include "the_fft_sycl.hpp"
#include "fft_example_DUT.h"
#include <stdio.h>
#include <assert.h>

namespace csl
{
#ifdef WRITE_STM_FILES
void error(const char* msg)   { printf("Error: %s\n", msg); }
void warning(const char* msg) { printf("Warning: %s\n", msg); }
void info(const char* msg)    { printf("Info: %s\n", msg); }
#else
void error(const char* msg)   { }
void warning(const char* msg) { }
void info(const char* msg)    { }
#endif
}

extern "C" fft_example_DUT* the_fft_new_instance()
{
    uint8_t* instance_data = (uint8_t*)malloc(sizeof(fft_example_DUT));
    fft_example_DUT* instance = new (instance_data) fft_example_DUT();
    instance->reset();
    return instance;
}

extern "C" void the_fft_delete_instance(fft_example_DUT* instance)
{
    instance->~fft_example_DUT();
    free(instance);
}

extern "C" the_fft_output_t the_fft(fft_example_DUT* instance, the_fft_input_t input)
{
    assert(instance != nullptr && "Invalid emulator instance!");
    fft_example_DUT::io_struct_ChannelIn input0;
    input0.port_v_in_s = input.port_v_in_s;
    input0.port_channel_in_s = input.port_channel_in_s;
    input0.port_data_in_0re = input.port_data_in_0re;
    input0.port_data_in_0im = input.port_data_in_0im;
    input0.port_data_in_1re = input.port_data_in_1re;
    input0.port_data_in_1im = input.port_data_in_1im;
    input0.port_data_in_2re = input.port_data_in_2re;
    input0.port_data_in_2im = input.port_data_in_2im;
    input0.port_data_in_3re = input.port_data_in_3re;
    input0.port_data_in_3im = input.port_data_in_3im;
    instance->write(input0);

    fft_example_DUT::io_struct_ChannelOut output0;
    instance->read(output0);

    the_fft_output_t result;
    result.port_v_out_s = output0.port_v_out_s;
    result.port_data_out_0re = output0.port_data_out_0re;
    result.port_data_out_0im = output0.port_data_out_0im;
    result.port_data_out_1re = output0.port_data_out_1re;
    result.port_data_out_1im = output0.port_data_out_1im;
    result.port_data_out_2re = output0.port_data_out_2re;
    result.port_data_out_2im = output0.port_data_out_2im;
    result.port_data_out_3re = output0.port_data_out_3re;
    result.port_data_out_3im = output0.port_data_out_3im;
    return result;
}
