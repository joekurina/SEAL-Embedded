// ------------------------------------------------------------------------- 
// High Level Design Compiler for Altera(R) FPGAs Version 25.1 (Release Build #6a12354d2f)
// Software model created on 2025-07-08 08:39:19
// Generation mode: Bit Accurate
// ------------------------------------------------------------------------- 
#pragma once

#ifndef SOFTWARE_MODEL_FLT_FFT_EXAMPLE_DUT_ADDBLOCK_TYPESFLOATIEEE_52_11_TYPESFLOATIEEE_52_11_TYPESFLOA0000J6OF0CD16OK0CP06HJ0U_H_
#define SOFTWARE_MODEL_FLT_FFT_EXAMPLE_DUT_ADDBLOCK_TYPESFLOATIEEE_52_11_TYPESFLOATIEEE_52_11_TYPESFLOA0000J6OF0CD16OK0CP06HJ0U_H_

#include "support/csl.h"
#ifdef WRITE_STM_FILES
#include "support/csl_io.h"
#endif

class flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u
{
public:
    // IO struct for "in_0"
    struct io_struct_in_0
    {
        double port_in_0 = 0;
    };

    // IO struct for "in_1"
    struct io_struct_in_1
    {
        double port_in_1 = 0;
    };

    // IO struct for "out_primWireOut"
    struct io_struct_out_primWireOut
    {
        double port_out_primwireout = 0;
    };

public:
    // Read functions read the values of an output node from the model.
    // Delay correction is applied if the model is bit-accurate
    // but not cycle accurate to simulate latency from delay balancing.
    // DSP Builder applies this same adjustment for bit-accurate
    // non-cycle accurate simulation in Simulink.
    // 
    // Read functions will append output to stimulus files if compiling
    // with WRITE_STM_FILES defined.
    // 
    // Read functions should be called once per cycle per instance.
    // Subsequent calls in the same cycle will copy output values to the
    // provided struct, but will not write stimulus or update internal state.
    void read(io_struct_out_primWireOut& output)
    {
        bool needs_to_execute = (m_io_cycle[2] == m_update_cycle);
        if (needs_to_execute)
        {
            m_io_cycle[2]++;
            execute();
        }

        mask_lower(output.port_out_primwireout, m_w[PORT_OUT_PRIMWIREOUT2], 64, m_temps);
    }

    // Write functions write the values of an input node to the model
    // and initiates any internal simulation that depends on that 
    // input and any other inputs previously provided for the
    // current cycle. When all inputs nodes have been provided,
    // the next cycle will begin automatically.
    // 
    // Write functions will append output to stimulus files if compiling
    // with WRITE_STM_FILES defined.
    // 
    // Write functions should be called once per cycle per instance.
    // Subsequent calls in the same cycle will do nothing.
    void write(const io_struct_in_0& input)
    {
        bool needs_to_execute = (m_io_cycle[0] == m_update_cycle);
        if (needs_to_execute)
        {
            mask_lower(m_w[PORT_IN_00], input.port_in_0, 64, m_temps);
            m_io_cycle[0]++;
            execute();
        }
    }

    void write(const io_struct_in_1& input)
    {
        bool needs_to_execute = (m_io_cycle[1] == m_update_cycle);
        if (needs_to_execute)
        {
            mask_lower(m_w[PORT_IN_11], input.port_in_1, 64, m_temps);
            m_io_cycle[1]++;
            execute();
        }
    }

    // Resets internal simulation state to default values and opens stimulus files.
    void reset()
    {
        static constexpr int64_t native_reset_range_values[] = { 0ll };
        static constexpr uint32_t native_reset_range_value_indices[] = { 0 };
        static constexpr size_t native_reset_range_indices[] = { 0 };
        static constexpr size_t native_reset_range_sizes[] = { 172 };
        for (size_t i = 0; i < 1; ++i)
        {
            csl::fill_n(&m_n[native_reset_range_indices[i]], native_reset_range_sizes[i], native_reset_range_values[native_reset_range_value_indices[i]]);
        }
        static constexpr uint64_t wide_reset_range_values[] = { 0 };
        static constexpr csl::mp_int_info wide_reset_range_infos[] = { { 0, 1, 0 } };
        static constexpr uint32_t wide_reset_range_value_indices[] = { 0 };
        static constexpr size_t wide_reset_range_indices[] = { 0 };
        static constexpr size_t wide_reset_range_sizes[] = { 19 };
        for (size_t i = 0; i < 1; ++i)
        {
            for (size_t j = 0; j < wide_reset_range_sizes[i]; ++j)
            {
                csl::fill_mpz_data(m_w[wide_reset_range_indices[i] + j],
                    wide_reset_range_values, wide_reset_range_infos, wide_reset_range_value_indices[i]);
            }
        }
        csl::fill_n(m_io_cycle, 3, -1);
        csl::fill_n(m_segment_cycle, 4, -1);
        m_update_cycle = -1;
    }

    // Opens all stimulus files associated with this class and its children.
    // Files will be flushed and closed on destruction or by calling close_stimulus_files().
    void open_stimulus_files()
    {
#ifdef WRITE_STM_FILES
#endif
    }

    // Closes and flushes all stimulus files associated with this class and its children.
    // Must call open_stimulus_files() before writing more stimulus data to re-open the files.
    void close_stimulus_files()
    {
#ifdef WRITE_STM_FILES
#endif
    }

    // If the current cycle of this system is equal to the
    // input cycle, sets the state of all output IO structs to read,
    // allowing the simulation state to advance to the next cycle.
    // This allows skipping the reading of unused output structs without
    // blocking the simulation. This function is primarily intended for
    // internal usage within model hierarchies.
    void flush_outputs(int64_t cycle)
    {
        if (m_update_cycle == cycle)
        {
            m_io_cycle[2] = m_update_cycle + 1;
            execute();
        }
    }

private:
    // Segments are chunks of execution that depend on unique sets of inputs
    // These functions are invoked automatically as inputs are provided to the model.
    void execute_segment_0_fragment_0()
    {
        csl::set(m_n[71], 57ll); // step_const
        csl::set(m_n[131], 0ll); // step_const
        csl::set(m_n[1], 1ll); // step_const
        csl::set(m_n[0], 0ll); // step_const
        csl::set(m_n[64], 0ll); // step_const
        csl::set(m_n[48], 54ll); // step_const
        csl::set(m_n[24], 0ll); // step_const
        csl::set(m_n[43], 0ll); // step_const
        csl::set(m_n[52], 0ll); // step_const
        csl::set(m_n[61], 1ll); // step_const
        csl::set(m_n[60], 0ll); // step_const
        csl::set(m_n[55], 0ll); // step_const
        csl::set(m_n[139], 0ll); // step_const
        csl::set(m_n[134], -1ll); // step_const
        csl::set(m_n[145], 0ll); // step_const
        csl::set(m_n[151], 0ll); // step_const
        csl::set(m_n[157], 0ll); // step_const
        csl::set(m_n[163], 0ll); // step_const
        csl::set(m_n[22], 2047ll); // step_const
        csl::set(m_n[8], 2047ll); // step_const
        csl::set(m_n[10], 0ll); // step_const
        csl::set(m_n[23], 0ll); // step_const
        csl::set(m_n[9], 0ll); // step_const
        csl::set(m_n[127], 2047ll); // step_const
        csl::set(m_n[128], 2047ll); // step_const
        csl::set(m_n[84], 8ll); // step_const
        csl::set(m_n[75], 1ll); // step_const
        csl::set(m_n[129], 0ll); // step_const
        csl::set(m_n[91], 1ll); // step_const
        csl::set(m_n[88], 2047ll); // step_const
        csl::set(m_n[95], 0ll); // step_const
        csl::set(m_n[123], 1ll); // step_const
        csl::set(m_n[124], 0ll); // step_const
        csl::set(m_n[125], 0ll); // step_const
    }

    void execute_segment_0()
    {
        execute_segment_0_fragment_0();
    }

    void execute_segment_1_fragment_0()
    {
        csl::step_reduce(m_segment_temps_w[0], m_w[PORT_IN_11], 64, m_temps);
        csl::step_bit_extract(m_n[7], m_segment_temps_w[0], 1, true, 63, m_temps);
        csl::mask_lower(m_segment_temps_w[0], m_w[PORT_IN_11], 64, m_temps);
        csl::step_bit_extract(m_n[6], m_segment_temps_w[0], 11, true, 52, m_temps);
        csl::mask_lower(m_segment_temps_w[0], m_w[PORT_IN_11], 64, m_temps);
        csl::step_bit_extract(m_n[5], m_segment_temps_w[0], 52, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[5], 52, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[6], 11, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[7], 1, m_temps);
        csl::step_bit_combine(m_w[4], m_segment_temps_n[0], m_segment_temps_n[1], 3, 1, 52, 63, m_temps);
        csl::step_bit_combine(m_w[4], m_w[4], m_segment_temps_n[2], 3, 2, 63, 0, m_temps);
        csl::mask_lower(m_segment_temps_w[0], m_w[PORT_IN_11], 64, m_temps);
        csl::step_bit_extract(m_n[3], m_segment_temps_w[0], 63, true, 0, m_temps);
    }

    void execute_segment_1()
    {
        execute_segment_1_fragment_0();
    }

    void execute_segment_2_fragment_0()
    {
        csl::step_reduce(m_segment_temps_w[0], m_w[PORT_IN_00], 64, m_temps);
        csl::step_bit_extract(m_w[5], m_segment_temps_w[0], 64, false, 0, m_temps);
        csl::mask_lower(m_segment_temps_w[0], m_w[5], 64, m_temps);
        csl::step_bit_extract(m_w[6], m_segment_temps_w[0], 64, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_w[0], m_w[6], 64, m_temps);
        csl::step_bit_extract(m_w[7], m_segment_temps_w[0], 64, true, 0, m_temps);
        csl::mask_lower(m_segment_temps_w[0], m_w[PORT_IN_00], 64, m_temps);
        csl::step_bit_extract(m_n[2], m_segment_temps_w[0], 63, true, 0, m_temps);
    }

    void execute_segment_2()
    {
        execute_segment_2_fragment_0();
    }

    void execute_segment_3_fragment_0()
    {
        csl::mask_lower(m_segment_temps_n[0], m_n[2], 63, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[3], 63, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[0], 1, m_temps);
        csl::step_add(m_segment_temps_w[0], m_segment_temps_n[1], m_segment_temps_n[3], m_temps);
        csl::step_sub(m_w[3], m_segment_temps_n[0], m_segment_temps_w[0], m_temps);
        csl::step_nsign_bit(m_n[4], m_w[3], 65);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[4], 1, m_temps);
        csl::step_reduce(m_segment_temps_w[1], m_w[7], 64, m_temps);
        csl::step_reduce(m_segment_temps_w[2], m_w[4], 64, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_36[2] = { 1, 2 };
        csl::set(m_w[9], m_segment_temps_w[csl::checked_array_value(mux_lookup_36, m_segment_temps_n[4])]);
        csl::step_reduce(m_segment_temps_w[3], m_w[9], 64, m_temps);
        csl::step_bit_extract(m_n[41], m_segment_temps_w[3], 1, true, 63, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[4], 1, m_temps);
        csl::step_reduce(m_segment_temps_w[1], m_w[4], 64, m_temps);
        csl::step_reduce(m_segment_temps_w[2], m_w[7], 64, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_44[2] = { 1, 2 };
        csl::set(m_w[8], m_segment_temps_w[csl::checked_array_value(mux_lookup_44, m_segment_temps_n[4])]);
        csl::step_reduce(m_segment_temps_w[3], m_w[8], 64, m_temps);
        csl::step_bit_extract(m_n[40], m_segment_temps_w[3], 1, true, 63, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[40], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[41], 1, m_temps);
        csl::step_xor(m_n[42], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[42], m_n[42], 1);
        csl::mask_lower(m_segment_temps_n[0], m_w[9], 63, m_temps);
        csl::step_bit_extract(m_n[37], m_segment_temps_n[0], 11, true, 52, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_w[8], 63, m_temps);
        csl::step_bit_extract(m_n[36], m_segment_temps_n[0], 11, true, 52, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[36], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[37], 11, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_sub(m_n[47], m_segment_temps_n[0], m_segment_temps_n[1], m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[47], 12, m_temps);
        csl::step_bit_extract(m_n[49], m_segment_temps_n[0], 12, false, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[48], 6, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[49], 12, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[0], 1, m_temps);
        csl::step_add(m_segment_temps_n[4], m_segment_temps_n[1], m_segment_temps_n[3], m_temps);
        csl::step_sub(m_n[51], m_segment_temps_n[0], m_segment_temps_n[4], m_temps);
        csl::step_sign_bit(m_n[50], m_n[51], 14);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[50], 1, m_temps);
        csl::step_not_signed(m_n[53], m_segment_temps_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_w[9], 63, m_temps);
        csl::step_bit_extract(m_n[25], m_segment_temps_n[0], 11, true, 52, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[25], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[24], 11, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 11, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 11);
        csl::step_reducing_and(m_n[27], m_segment_temps_n[3], 11, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[27], 1, m_temps);
        csl::step_not_signed(m_n[45], m_segment_temps_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_w[9], 52, m_temps);
        csl::step_bit_extract(m_n[39], m_segment_temps_n[0], 52, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[27], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[39], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[43], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_92[2] = { 2, 3 };
        csl::set(m_n[44], m_segment_temps_n[csl::checked_array_value(mux_lookup_92, m_segment_temps_n[4])]);
        csl::step_reduce(m_segment_temps_n[0], m_n[44], 52, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[45], 1, m_temps);
        csl::step_bit_combine(m_n[46], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 52, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[52], 54, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[46], 53, m_temps);
        csl::step_bit_combine(m_w[10], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 54, 0, m_temps);
        csl::mask_lower(m_segment_temps_w[3], m_w[10], 107, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[47], 12, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_ld_exp(m_w[11], m_segment_temps_w[3], m_segment_temps_n[1], true, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_w[4], m_w[11], 107, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[53], 1, m_temps);
        csl::step_and(m_w[12], m_segment_temps_w[4], m_segment_temps_n[2], 107, m_temps);
        csl::step_reduce(m_w[12], m_w[12], 107, m_temps);
        csl::mask_lower(m_segment_temps_w[3], m_w[12], 107, m_temps);
        csl::step_bit_extract(m_n[63], m_segment_temps_w[3], 55, true, 52, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[63], 55, m_temps);
    }

    void execute_segment_3_fragment_1()
    {
        csl::step_reduce(m_segment_temps_n[1], m_n[64], 1, m_temps);
        csl::step_bit_combine(m_n[65], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 55, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[65], 56, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[42], 1, m_temps);
        csl::step_xor(m_n[66], m_segment_temps_n[1], m_segment_temps_n[2], 56, m_temps);
        csl::step_reduce(m_n[66], m_n[66], 56, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_w[8], 52, m_temps);
        csl::step_bit_extract(m_n[38], m_segment_temps_n[0], 52, true, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_w[12], 52, m_temps);
        csl::step_bit_extract(m_n[54], m_segment_temps_n[0], 52, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[54], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[55], 52, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 52, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 52);
        csl::step_reducing_and(m_n[56], m_segment_temps_n[3], 52, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[56], 1, m_temps);
        csl::step_not_signed(m_n[57], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[57], 1, m_temps);
        csl::step_not_signed(m_n[58], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[42], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[58], 1, m_temps);
        csl::step_and(m_n[59], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[59], m_n[59], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[59], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[60], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[38], 52, m_temps);
        csl::step_reduce(m_segment_temps_n[3], m_n[61], 2, m_temps);
        csl::step_bit_combine(m_n[62], m_segment_temps_n[0], m_segment_temps_n[1], 4, 1, 1, 2, m_temps);
        csl::step_bit_combine(m_n[62], m_n[62], m_segment_temps_n[2], 4, 2, 2, 54, m_temps);
        csl::step_bit_combine(m_n[62], m_n[62], m_segment_temps_n[3], 4, 3, 54, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[62], 56, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[66], 56, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_add(m_n[67], m_segment_temps_n[0], m_segment_temps_n[1], m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[67], 56, m_temps);
        csl::step_bit_extract(m_n[68], m_segment_temps_n[0], 56, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[57], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[68], 56, m_temps);
        csl::step_bit_combine(m_n[69], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 1, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[69], 57, m_temps);
        csl::step_bit_extract(m_n[132], m_segment_temps_n[0], 32, true, 25, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[132], 32, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[131], 32, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 32, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 32);
        csl::step_reducing_and(m_n[133], m_segment_temps_n[3], 32, false, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[69], 25, m_temps);
        csl::step_bit_extract(m_n[135], m_segment_temps_n[0], 25, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[134], 7, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[135], 25, m_temps);
        csl::step_bit_combine(m_n[136], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 7, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[69], 57, m_temps);
        csl::step_bit_extract(m_n[137], m_segment_temps_n[0], 32, true, 25, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[133], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[137], 32, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[136], 32, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_174[2] = { 2, 3 };
        csl::set(m_n[138], m_segment_temps_n[csl::checked_array_value(mux_lookup_174, m_segment_temps_n[4])]);
        csl::mask_lower(m_segment_temps_n[0], m_n[138], 32, m_temps);
        csl::step_bit_extract(m_n[140], m_segment_temps_n[0], 16, true, 16, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[140], 16, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[139], 16, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 16, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 16);
        csl::step_reducing_and(m_n[141], m_segment_temps_n[3], 16, false, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[138], 16, m_temps);
        csl::step_bit_extract(m_n[142], m_segment_temps_n[0], 16, true, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[138], 32, m_temps);
        csl::step_bit_extract(m_n[143], m_segment_temps_n[0], 16, true, 16, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[141], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[143], 16, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[142], 16, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_192[2] = { 2, 3 };
        csl::set(m_n[144], m_segment_temps_n[csl::checked_array_value(mux_lookup_192, m_segment_temps_n[4])]);
        csl::mask_lower(m_segment_temps_n[0], m_n[144], 16, m_temps);
        csl::step_bit_extract(m_n[146], m_segment_temps_n[0], 8, true, 8, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[146], 8, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[145], 8, m_temps);
    }

    void execute_segment_3_fragment_2()
    {
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 8, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 8);
        csl::step_reducing_and(m_n[147], m_segment_temps_n[3], 8, false, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[144], 8, m_temps);
        csl::step_bit_extract(m_n[148], m_segment_temps_n[0], 8, true, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[144], 16, m_temps);
        csl::step_bit_extract(m_n[149], m_segment_temps_n[0], 8, true, 8, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[147], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[149], 8, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[148], 8, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_210[2] = { 2, 3 };
        csl::set(m_n[150], m_segment_temps_n[csl::checked_array_value(mux_lookup_210, m_segment_temps_n[4])]);
        csl::mask_lower(m_segment_temps_n[0], m_n[150], 8, m_temps);
        csl::step_bit_extract(m_n[152], m_segment_temps_n[0], 4, true, 4, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[152], 4, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[151], 4, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 4, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 4);
        csl::step_reducing_and(m_n[153], m_segment_temps_n[3], 4, false, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[150], 4, m_temps);
        csl::step_bit_extract(m_n[154], m_segment_temps_n[0], 4, true, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[150], 8, m_temps);
        csl::step_bit_extract(m_n[155], m_segment_temps_n[0], 4, true, 4, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[153], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[155], 4, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[154], 4, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_228[2] = { 2, 3 };
        csl::set(m_n[156], m_segment_temps_n[csl::checked_array_value(mux_lookup_228, m_segment_temps_n[4])]);
        csl::mask_lower(m_segment_temps_n[0], m_n[156], 4, m_temps);
        csl::step_bit_extract(m_n[158], m_segment_temps_n[0], 2, true, 2, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[158], 2, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[157], 2, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 2, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 2);
        csl::step_reducing_and(m_n[159], m_segment_temps_n[3], 2, false, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[156], 2, m_temps);
        csl::step_bit_extract(m_n[160], m_segment_temps_n[0], 2, true, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[156], 4, m_temps);
        csl::step_bit_extract(m_n[161], m_segment_temps_n[0], 2, true, 2, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[159], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[161], 2, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[160], 2, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_246[2] = { 2, 3 };
        csl::set(m_n[162], m_segment_temps_n[csl::checked_array_value(mux_lookup_246, m_segment_temps_n[4])]);
        csl::mask_lower(m_segment_temps_n[0], m_n[162], 2, m_temps);
        csl::step_bit_extract(m_n[164], m_segment_temps_n[0], 1, true, 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[164], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[163], 1, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 1);
        csl::step_reducing_and(m_n[165], m_segment_temps_n[3], 1, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[165], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[159], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[153], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[3], m_n[147], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[4], m_n[141], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[5], m_n[133], 1, m_temps);
        csl::step_bit_combine(m_n[166], m_segment_temps_n[0], m_segment_temps_n[1], 6, 1, 1, 2, m_temps);
        csl::step_bit_combine(m_n[166], m_n[166], m_segment_temps_n[2], 6, 2, 2, 3, m_temps);
        csl::step_bit_combine(m_n[166], m_n[166], m_segment_temps_n[3], 6, 3, 3, 4, m_temps);
        csl::step_bit_combine(m_n[166], m_n[166], m_segment_temps_n[4], 6, 4, 4, 5, m_temps);
        csl::step_bit_combine(m_n[166], m_n[166], m_segment_temps_n[5], 6, 5, 5, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[166], 6, m_temps);
        csl::step_bit_extract(m_n[70], m_segment_temps_n[0], 6, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[70], 6, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[71], 6, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 6, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 6);
        csl::step_reducing_and(m_n[72], m_segment_temps_n[3], 6, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[72], 1, m_temps);
        csl::step_not_signed(m_n[112], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[25], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[22], 11, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 11, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 11);
        csl::step_reducing_and(m_n[28], m_segment_temps_n[3], 11, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[28], 1, m_temps);
        csl::step_not_signed(m_n[33], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[27], 1, m_temps);
        csl::step_not_signed(m_n[34], m_segment_temps_n[1], 1, m_temps);
    }

    void execute_segment_3_fragment_3()
    {
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[34], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[33], 1, m_temps);
        csl::step_and(m_n[35], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[35], m_n[35], 1);
        csl::mask_lower(m_segment_temps_n[0], m_w[8], 63, m_temps);
        csl::step_bit_extract(m_n[11], m_segment_temps_n[0], 11, true, 52, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[11], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[8], 11, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 11, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 11);
        csl::step_reducing_and(m_n[14], m_segment_temps_n[3], 11, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[14], 1, m_temps);
        csl::step_not_signed(m_n[19], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[11], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[10], 11, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 11, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 11);
        csl::step_reducing_and(m_n[13], m_segment_temps_n[3], 11, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[13], 1, m_temps);
        csl::step_not_signed(m_n[20], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[20], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[19], 1, m_temps);
        csl::step_and(m_n[21], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[21], m_n[21], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[21], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[35], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[40], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_n[112], 1, m_temps);
        csl::step_and(m_n[113], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[113], m_n[113], 1);
        csl::step_and(m_n[113], m_n[113], m_segment_temps_n[3], 1, m_temps);
        csl::mask(m_n[113], m_n[113], 1);
        csl::step_and(m_n[113], m_n[113], m_segment_temps_n[4], 1, m_temps);
        csl::mask(m_n[113], m_n[113], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[13], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[27], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[40], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_n[41], 1, m_temps);
        csl::step_and(m_n[117], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[117], m_n[117], 1);
        csl::step_and(m_n[117], m_n[117], m_segment_temps_n[3], 1, m_temps);
        csl::mask(m_n[117], m_n[117], 1);
        csl::step_and(m_n[117], m_n[117], m_segment_temps_n[4], 1, m_temps);
        csl::mask(m_n[117], m_n[117], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[27], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[21], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[40], 1, m_temps);
        csl::step_and(m_n[118], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[118], m_n[118], 1);
        csl::step_and(m_n[118], m_n[118], m_segment_temps_n[3], 1, m_temps);
        csl::mask(m_n[118], m_n[118], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[118], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[117], 1, m_temps);
        csl::step_or(m_n[119], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[119], m_n[119], 1);
        csl::mask_lower(m_segment_temps_n[0], m_w[9], 52, m_temps);
        csl::step_bit_extract(m_n[26], m_segment_temps_n[0], 52, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[23], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[26], 52, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 52, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 52);
        csl::step_reducing_and(m_n[29], m_segment_temps_n[3], 52, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[28], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[29], 1, m_temps);
        csl::step_and(m_n[31], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[31], m_n[31], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[41], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[31], 1, m_temps);
        csl::step_and(m_n[114], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[114], m_n[114], 1);
        csl::mask_lower(m_segment_temps_n[0], m_w[8], 52, m_temps);
        csl::step_bit_extract(m_n[12], m_segment_temps_n[0], 52, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[9], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[12], 52, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 52, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 52);
        csl::step_reducing_and(m_n[15], m_segment_temps_n[3], 52, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[14], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[15], 1, m_temps);
        csl::step_and(m_n[17], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[17], m_n[17], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[40], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[17], 1, m_temps);
        csl::step_and(m_n[115], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[115], m_n[115], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[115], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[114], 1, m_temps);
        csl::step_or(m_n[116], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[116], m_n[116], 1);
    }

    void execute_segment_3_fragment_4()
    {
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[116], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[119], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[113], 1, m_temps);
        csl::step_or(m_n[120], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[120], m_n[120], 1);
        csl::step_or(m_n[120], m_n[120], m_segment_temps_n[3], 1, m_temps);
        csl::mask(m_n[120], m_n[120], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[29], 1, m_temps);
        csl::step_not_signed(m_n[30], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[28], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[30], 1, m_temps);
        csl::step_and(m_n[32], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[32], m_n[32], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[15], 1, m_temps);
        csl::step_not_signed(m_n[16], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[14], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[16], 1, m_temps);
        csl::step_and(m_n[18], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[18], m_n[18], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[18], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[32], 1, m_temps);
        csl::step_or(m_n[107], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[107], m_n[107], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[17], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[31], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[42], 1, m_temps);
        csl::step_and(m_n[108], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[108], m_n[108], 1);
        csl::step_and(m_n[108], m_n[108], m_segment_temps_n[3], 1, m_temps);
        csl::mask(m_n[108], m_n[108], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[108], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[107], 1, m_temps);
        csl::step_or(m_n[109], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[109], m_n[109], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[109], 1, m_temps);
        csl::step_not_signed(m_n[121], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[121], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[120], 1, m_temps);
        csl::step_and(m_n[122], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[122], m_n[122], 1);
        csl::mask_lower(m_segment_temps_n[0], m_n[69], 57, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[70], 6, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_ld_exp(m_n[73], m_segment_temps_n[0], m_segment_temps_n[1], false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[73], 5, m_temps);
        csl::step_bit_extract(m_n[82], m_segment_temps_n[0], 1, true, 4, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[73], 4, m_temps);
        csl::step_bit_extract(m_n[81], m_segment_temps_n[0], 1, true, 3, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[73], 3, m_temps);
        csl::step_bit_extract(m_n[80], m_segment_temps_n[0], 1, true, 2, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[73], 2, m_temps);
        csl::step_bit_extract(m_n[79], m_segment_temps_n[0], 1, true, 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[73], 1, m_temps);
        csl::step_bit_extract(m_n[78], m_segment_temps_n[0], 1, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[78], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[79], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[80], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[3], m_n[81], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[4], m_n[82], 1, m_temps);
        csl::step_bit_combine(m_n[83], m_segment_temps_n[0], m_segment_temps_n[1], 5, 1, 1, 2, m_temps);
        csl::step_bit_combine(m_n[83], m_n[83], m_segment_temps_n[2], 5, 2, 2, 3, m_temps);
        csl::step_bit_combine(m_n[83], m_n[83], m_segment_temps_n[3], 5, 3, 3, 4, m_temps);
        csl::step_bit_combine(m_n[83], m_n[83], m_segment_temps_n[4], 5, 4, 4, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[83], 5, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[84], 5, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 5, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 5);
        csl::step_reducing_and(m_n[85], m_segment_temps_n[3], 5, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[85], 1, m_temps);
        csl::step_not_signed(m_n[86], m_segment_temps_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[36], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[75], 11, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_add(m_n[76], m_segment_temps_n[0], m_segment_temps_n[1], m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[76], 12, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[70], 6, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_sub(m_n[77], m_segment_temps_n[0], m_segment_temps_n[1], m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[73], 57, m_temps);
        csl::step_bit_extract(m_n[74], m_segment_temps_n[0], 56, true, 1, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[74], 55, m_temps);
        csl::step_bit_extract(m_n[87], m_segment_temps_n[0], 53, true, 2, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[87], 53, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[77], 13, m_temps);
    }

    void execute_segment_3_fragment_5()
    {
        csl::step_bit_combine(m_w[13], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 53, 0, m_temps);
        csl::mask_lower(m_segment_temps_w[3], m_w[13], 66, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[86], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_add(m_w[14], m_segment_temps_w[3], m_segment_temps_n[1], m_temps);
        csl::mask_lower(m_segment_temps_w[3], m_w[14], 64, m_temps);
        csl::step_bit_extract(m_n[100], m_segment_temps_w[3], 11, true, 53, m_temps);
        csl::mask_lower(m_segment_temps_w[3], m_w[14], 66, m_temps);
        csl::step_bit_extract(m_n[92], m_segment_temps_w[3], 2, true, 64, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[92], 2, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[91], 2, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 2, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 2);
        csl::step_reducing_and(m_n[93], m_segment_temps_n[3], 2, false, m_temps);
        csl::mask_lower(m_segment_temps_w[3], m_w[14], 66, m_temps);
        csl::step_bit_extract(m_n[89], m_segment_temps_w[3], 13, true, 53, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[89], 13, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[88], 13, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 13, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 13);
        csl::step_reducing_and(m_n[90], m_segment_temps_n[3], 13, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[90], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[93], 1, m_temps);
        csl::step_or(m_n[94], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[94], m_n[94], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[21], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[35], 1, m_temps);
        csl::step_and(m_n[101], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[101], m_n[101], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[101], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[94], 1, m_temps);
        csl::step_and(m_n[104], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[104], m_n[104], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[42], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[17], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[31], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[3], m_n[18], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[4], m_n[32], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[5], m_n[104], 1, m_temps);
        csl::step_bit_combine(m_n[105], m_segment_temps_n[0], m_segment_temps_n[1], 6, 1, 1, 2, m_temps);
        csl::step_bit_combine(m_n[105], m_n[105], m_segment_temps_n[2], 6, 2, 2, 3, m_temps);
        csl::step_bit_combine(m_n[105], m_n[105], m_segment_temps_n[3], 6, 3, 3, 4, m_temps);
        csl::step_bit_combine(m_n[105], m_n[105], m_segment_temps_n[4], 6, 4, 4, 5, m_temps);
        csl::step_bit_combine(m_n[105], m_n[105], m_segment_temps_n[5], 6, 5, 5, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[105], 6, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[1], 1, m_temps);
        static constexpr int64_t lut_init_data_498[] = {
            0ll, 0ll, 1ll, 1ll, 1ll, 1ll, 1ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 
            0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 
            1ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 
            0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll
        };
        csl::step_lookup(lut_init_data_498, 64, 0, m_n[106], m_segment_temps_n[0], m_temps);
        csl::step_reduce(m_segment_temps_w[3], m_w[14], 66, m_temps);
        csl::step_bit_extract(m_n[97], m_segment_temps_w[3], 1, true, 65, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[89], 13, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[95], 13, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 13, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 13);
        csl::step_reducing_and(m_n[96], m_segment_temps_n[3], 13, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[96], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[97], 1, m_temps);
        csl::step_or(m_n[98], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[98], m_n[98], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[13], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[27], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[101], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[3], m_n[98], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[4], m_n[72], 1, m_temps);
        csl::step_bit_combine(m_n[102], m_segment_temps_n[0], m_segment_temps_n[1], 5, 1, 1, 2, m_temps);
        csl::step_bit_combine(m_n[102], m_n[102], m_segment_temps_n[2], 5, 2, 2, 3, m_temps);
        csl::step_bit_combine(m_n[102], m_n[102], m_segment_temps_n[3], 5, 3, 3, 4, m_temps);
        csl::step_bit_combine(m_n[102], m_n[102], m_segment_temps_n[4], 5, 4, 4, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[102], 5, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[1], 1, m_temps);
        static constexpr int64_t lut_init_data_519[] = {
            0ll, 0ll, 0ll, 1ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 1ll, 1ll, 0ll, 0ll, 0ll, 
            0ll, 0ll, 0ll, 1ll, 1ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 1ll, 1ll, 0ll, 0ll, 0ll
        };
        csl::step_lookup(lut_init_data_519, 32, 0, m_n[103], m_segment_temps_n[0], m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[103], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[106], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[109], 1, m_temps);
        csl::step_bit_combine(m_n[110], m_segment_temps_n[0], m_segment_temps_n[1], 3, 1, 1, 2, m_temps);
        csl::step_bit_combine(m_n[110], m_n[110], m_segment_temps_n[2], 3, 2, 2, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[110], 3, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[1], 1, m_temps);
        static constexpr int64_t lut_init_data_526[] = {
            1ll, 0ll, 2ll, 2ll, 3ll, 3ll, 3ll, 3ll
        };
        csl::step_lookup(lut_init_data_526, 8, 0, m_n[111], m_segment_temps_n[0], m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[111], 2, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[129], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[100], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_n[128], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[5], m_n[127], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[6], m_segment_temps_n[1], 2, m_temps);
        static constexpr size_t mux_lookup_534[4] = { 2, 3, 4, 5 };
        csl::set(m_n[130], m_segment_temps_n[csl::checked_array_value(mux_lookup_534, m_segment_temps_n[6])]);
        csl::mask_lower(m_segment_temps_n[0], m_w[14], 53, m_temps);
        csl::step_bit_extract(m_n[99], m_segment_temps_n[0], 52, true, 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
    }

    void execute_segment_3_fragment_6()
    {
        csl::mask_lower(m_segment_temps_n[1], m_n[111], 2, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[125], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[99], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_n[124], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[5], m_n[123], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[6], m_segment_temps_n[1], 2, m_temps);
        static constexpr size_t mux_lookup_547[4] = { 2, 3, 4, 5 };
        csl::set(m_n[126], m_segment_temps_n[csl::checked_array_value(mux_lookup_547, m_segment_temps_n[6])]);
        csl::step_reduce(m_segment_temps_n[0], m_n[126], 52, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[130], 11, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[122], 1, m_temps);
        csl::step_bit_combine(m_w[15], m_segment_temps_n[0], m_segment_temps_n[1], 3, 1, 52, 63, m_temps);
        csl::step_bit_combine(m_w[15], m_w[15], m_segment_temps_n[2], 3, 2, 63, 0, m_temps);
        csl::step_reduce(m_segment_temps_w[3], m_w[15], 64, m_temps);
        csl::step_bit_extract(m_w[0], m_segment_temps_w[3], 64, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_w[3], m_w[0], 64, m_temps);
        m_w[PORT_OUT_PRIMWIREOUT2] = m_segment_temps_w[3];
    }

    void execute_segment_3()
    {
        execute_segment_3_fragment_0();
        execute_segment_3_fragment_1();
        execute_segment_3_fragment_2();
        execute_segment_3_fragment_3();
        execute_segment_3_fragment_4();
        execute_segment_3_fragment_5();
        execute_segment_3_fragment_6();
    }

    void execute_segment_update()
    {
    }


    // Progresses the internal simulation state based on the currently
    // available input values
    void execute()
    {
        if (m_segment_cycle[0] == m_update_cycle)
        {
            execute_segment_0();
            ++m_segment_cycle[0];
        }
        if ((m_segment_cycle[1] == m_update_cycle) && (m_io_cycle[1] > m_update_cycle))
        {
            execute_segment_1();
            ++m_segment_cycle[1];
        }
        if ((m_segment_cycle[2] == m_update_cycle) && (m_io_cycle[0] > m_update_cycle))
        {
            execute_segment_2();
            ++m_segment_cycle[2];
        }
        if ((m_segment_cycle[3] == m_update_cycle) && (m_io_cycle[1] > m_update_cycle) && (m_io_cycle[0] > m_update_cycle))
        {
            execute_segment_3();
            ++m_segment_cycle[3];
        }
        const bool all_io_ready = (m_io_cycle[0] > m_update_cycle) && (m_io_cycle[1] > m_update_cycle) && (m_io_cycle[2] > m_update_cycle);
        if (all_io_ready && (m_segment_cycle[0] > m_update_cycle) && (m_segment_cycle[1] > m_update_cycle) && (m_segment_cycle[2] > m_update_cycle) && (m_segment_cycle[3] > m_update_cycle))
        {
            execute_segment_update();
            ++m_update_cycle;
        }
    }

    static constexpr size_t PORT_IN_00 = 1;
    static constexpr size_t PORT_IN_11 = 2;
    static constexpr size_t PORT_OUT_PRIMWIREOUT2 = 18;

    int64_t m_io_cycle[3];
    int64_t m_segment_cycle[4];
    int64_t m_update_cycle;

    int64_t m_segment_temps_n[7] = { 0 };
    csl::mp_int m_segment_temps_w[5];
    csl::mp_int_temps m_temps;
    int64_t m_n[172];
    csl::mp_int m_w[19];
};

#endif // SOFTWARE_MODEL_FLT_FFT_EXAMPLE_DUT_ADDBLOCK_TYPESFLOATIEEE_52_11_TYPESFLOATIEEE_52_11_TYPESFLOA0000J6OF0CD16OK0CP06HJ0U_H_