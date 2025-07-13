// ------------------------------------------------------------------------- 
// High Level Design Compiler for Altera(R) FPGAs Version 25.1 (Release Build #6a12354d2f)
// Software model created on 2025-07-08 08:39:19
// Generation mode: Bit Accurate
// ------------------------------------------------------------------------- 
#include "the_fft_sycl.hpp"
#include "support/csl.h"
#include "support/csl_io.h"

class fft_example_DUTWrapperATB
{
public:
    void read_from_stm_ChannelIn(csl::stimulus_file& stm, the_fft_input_t& valueOut)
    {
        int8_t tmp_valueoutport_v_in_s;
        stm.get(tmp_valueoutport_v_in_s, 1);
        valueOut.port_v_in_s = tmp_valueoutport_v_in_s;
        int8_t tmp_valueoutport_channel_in_s;
        stm.get(tmp_valueoutport_channel_in_s, 8);
        valueOut.port_channel_in_s = tmp_valueoutport_channel_in_s;
        double tmp_valueoutport_data_in_0re;
        stm.get(tmp_valueoutport_data_in_0re);
        valueOut.port_data_in_0re = tmp_valueoutport_data_in_0re;
        double tmp_valueoutport_data_in_0im;
        stm.get(tmp_valueoutport_data_in_0im);
        valueOut.port_data_in_0im = tmp_valueoutport_data_in_0im;
        double tmp_valueoutport_data_in_1re;
        stm.get(tmp_valueoutport_data_in_1re);
        valueOut.port_data_in_1re = tmp_valueoutport_data_in_1re;
        double tmp_valueoutport_data_in_1im;
        stm.get(tmp_valueoutport_data_in_1im);
        valueOut.port_data_in_1im = tmp_valueoutport_data_in_1im;
        double tmp_valueoutport_data_in_2re;
        stm.get(tmp_valueoutport_data_in_2re);
        valueOut.port_data_in_2re = tmp_valueoutport_data_in_2re;
        double tmp_valueoutport_data_in_2im;
        stm.get(tmp_valueoutport_data_in_2im);
        valueOut.port_data_in_2im = tmp_valueoutport_data_in_2im;
        double tmp_valueoutport_data_in_3re;
        stm.get(tmp_valueoutport_data_in_3re);
        valueOut.port_data_in_3re = tmp_valueoutport_data_in_3re;
        double tmp_valueoutport_data_in_3im;
        stm.get(tmp_valueoutport_data_in_3im);
        valueOut.port_data_in_3im = tmp_valueoutport_data_in_3im;
    }

    void read_from_stm_ChannelOut(csl::stimulus_file& stm, the_fft_output_t& valueOut)
    {
        int8_t tmp_valueoutport_v_out_s;
        stm.get(tmp_valueoutport_v_out_s, 1);
        valueOut.port_v_out_s = tmp_valueoutport_v_out_s;
        stm.skip(1);
        double tmp_valueoutport_data_out_0re;
        stm.get(tmp_valueoutport_data_out_0re);
        valueOut.port_data_out_0re = tmp_valueoutport_data_out_0re;
        double tmp_valueoutport_data_out_0im;
        stm.get(tmp_valueoutport_data_out_0im);
        valueOut.port_data_out_0im = tmp_valueoutport_data_out_0im;
        double tmp_valueoutport_data_out_1re;
        stm.get(tmp_valueoutport_data_out_1re);
        valueOut.port_data_out_1re = tmp_valueoutport_data_out_1re;
        double tmp_valueoutport_data_out_1im;
        stm.get(tmp_valueoutport_data_out_1im);
        valueOut.port_data_out_1im = tmp_valueoutport_data_out_1im;
        double tmp_valueoutport_data_out_2re;
        stm.get(tmp_valueoutport_data_out_2re);
        valueOut.port_data_out_2re = tmp_valueoutport_data_out_2re;
        double tmp_valueoutport_data_out_2im;
        stm.get(tmp_valueoutport_data_out_2im);
        valueOut.port_data_out_2im = tmp_valueoutport_data_out_2im;
        double tmp_valueoutport_data_out_3re;
        stm.get(tmp_valueoutport_data_out_3re);
        valueOut.port_data_out_3re = tmp_valueoutport_data_out_3re;
        double tmp_valueoutport_data_out_3im;
        stm.get(tmp_valueoutport_data_out_3im);
        valueOut.port_data_out_3im = tmp_valueoutport_data_out_3im;
    }

    bool run()
    {
        bool success = true;
        fft_example_DUT* instance = the_fft_new_instance();
        instance->open_stimulus_files();
        the_fft_input_t input;
        the_fft_output_t output;
        csl::info("[fft_example_DUT] Opening stimulus files...");
        csl::stimulus_file io_struct_ChannelIn_stm_in0("../fft_example_DUT_ChannelIn_vunroll_cunroll_x.stm", csl::stimulus_file::StimulusFormat::SIGNED);
        success &= io_struct_ChannelIn_stm_in0.is_open();
        csl::info("[fft_example_DUT] Simulating...");
        while(io_struct_ChannelIn_stm_in0.next_line())
        {
            read_from_stm_ChannelIn(io_struct_ChannelIn_stm_in0, input);
            output = the_fft(instance, input);
        }
        the_fft_delete_instance(instance);
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
        if (success)
        {
            csl::info("[fft_example_DUT] Success! Software model matches Simulink simulation.");
        }
        else
        {
            csl::info("[fft_example_DUT] Error! Software model does not match Simulink simulation.");
        }
        return success;
    }
};


int main(int argc, char** argv)
{
    fft_example_DUTWrapperATB wrapper_atb;
    if (wrapper_atb.run())
    {
        return wrapper_atb.compare() ? 0 : 1;
    }
    return 1;
}
