// ------------------------------------------------------------------------- 
// High Level Design Compiler for Altera(R) FPGAs Version 25.1 (Release Build #6a12354d2f)
// Software model created on 2025-07-08 08:39:19
// Generation mode: Bit Accurate
// ------------------------------------------------------------------------- 
#if defined(CSL_USE_PRAGMA_ONCE)
#pragma once
#endif

#ifndef SOFTWARE_MODEL_FFT_EXAMPLE_DUTATB_H_
#define SOFTWARE_MODEL_FFT_EXAMPLE_DUTATB_H_

#include "support/csl_io.h"
#include "fft_example_DUT.h"

class fft_example_DUTATB
{
public:
    void read_from_stm(csl::stimulus_file& stm, fft_example_DUT::io_struct_ChannelIn& valueOut)
    {
        stm.get(valueOut.port_v_in_s, 1);
        stm.get(valueOut.port_channel_in_s, 8);
        stm.get(valueOut.port_data_in_0re);
        stm.get(valueOut.port_data_in_0im);
        stm.get(valueOut.port_data_in_1re);
        stm.get(valueOut.port_data_in_1im);
        stm.get(valueOut.port_data_in_2re);
        stm.get(valueOut.port_data_in_2im);
        stm.get(valueOut.port_data_in_3re);
        stm.get(valueOut.port_data_in_3im);
    }

    bool run()
    {
        bool success = true;
        fft_example_DUT* model_instance = new fft_example_DUT();
        model_instance->reset();
        model_instance->open_stimulus_files();
        csl::info("[fft_example_DUT] Opening input stimulus files...");
        csl::stimulus_file io_struct_ChannelIn_stm_in0("../fft_example_DUT_ChannelIn_vunroll_cunroll_x.stm", csl::stimulus_file::StimulusFormat::SIGNED);
        success &= io_struct_ChannelIn_stm_in0.is_open();
        csl::info("[fft_example_DUT] Simulating...");
        while(io_struct_ChannelIn_stm_in0.next_line())
        {
            fft_example_DUT::io_struct_ChannelIn io_struct_ChannelIn0;
            read_from_stm(io_struct_ChannelIn_stm_in0, io_struct_ChannelIn0);
            model_instance->write(io_struct_ChannelIn0);
            fft_example_DUT::io_struct_ChannelOut io_struct_ChannelOut0;
            model_instance->read(io_struct_ChannelOut0);
        }
        delete model_instance;
        if (success)
        {
            csl::info("[fft_example_DUT] Simulation has completed.");
            return true;
        }
        csl::info("[fft_example_DUT] Simulation failure! Stimulus file IO errors occurred.");
        return false;
    }

    bool compare()
    {
        bool success = true;
        success &= csl::compare_stm_files("fft_example_DUT_ChannelOut_vunroll_cunroll_x.stm", "../fft_example_DUT_ChannelOut_vunroll_cunroll_x.stm");
        return success;
    }
};

#endif // SOFTWARE_MODEL_FFT_EXAMPLE_DUTATB_H_

