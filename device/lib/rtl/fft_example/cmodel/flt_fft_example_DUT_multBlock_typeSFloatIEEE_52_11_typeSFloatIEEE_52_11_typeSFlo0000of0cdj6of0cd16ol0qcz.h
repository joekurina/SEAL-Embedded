// ------------------------------------------------------------------------- 
// High Level Design Compiler for Altera(R) FPGAs Version 25.1 (Release Build #6a12354d2f)
// Software model created on 2025-07-08 08:39:19
// Generation mode: Bit Accurate
// ------------------------------------------------------------------------- 
#pragma once

#ifndef SOFTWARE_MODEL_FLT_FFT_EXAMPLE_DUT_MULTBLOCK_TYPESFLOATIEEE_52_11_TYPESFLOATIEEE_52_11_TYPESFLO0000OF0CDJ6OF0CD16OL0QCZ_H_
#define SOFTWARE_MODEL_FLT_FFT_EXAMPLE_DUT_MULTBLOCK_TYPESFLOATIEEE_52_11_TYPESFLOATIEEE_52_11_TYPESFLO0000OF0CDJ6OF0CD16OL0QCZ_H_

#include "support/csl.h"
#ifdef WRITE_STM_FILES
#include "support/csl_io.h"
#endif

class flt_fft_example_DUT_multBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFlo0000of0cdj6of0cd16ol0qcz
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
        static constexpr size_t native_reset_range_sizes[] = { 127 };
        for (size_t i = 0; i < 1; ++i)
        {
            csl::fill_n(&m_n[native_reset_range_indices[i]], native_reset_range_sizes[i], native_reset_range_values[native_reset_range_value_indices[i]]);
        }
        static constexpr uint64_t wide_reset_range_values[] = { 0 };
        static constexpr csl::mp_int_info wide_reset_range_infos[] = { { 0, 1, 0 } };
        static constexpr uint32_t wide_reset_range_value_indices[] = { 0 };
        static constexpr size_t wide_reset_range_indices[] = { 0 };
        static constexpr size_t wide_reset_range_sizes[] = { 15 };
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
        csl::set(m_n[7], 0ll); // step_const
        csl::set(m_n[1], 1ll); // step_const
        csl::set(m_n[6], 2047ll); // step_const
        csl::set(m_n[22], 0ll); // step_const
        csl::set(m_n[21], 0ll); // step_const
        csl::set(m_n[20], 2047ll); // step_const
        csl::set(m_n[8], 0ll); // step_const
        csl::set(m_n[91], 2047ll); // step_const
        csl::set(m_n[92], 2047ll); // step_const
        csl::set(m_n[0], 0ll); // step_const
        csl::set(m_n[38], 1ll); // step_const
        csl::set(m_n[35], 1ll); // step_const
        csl::set(m_n[117], 0ll); // step_const
        csl::set(m_n[113], 0ll); // step_const
        csl::set(m_n[106], 0ll); // step_const
        csl::set(m_n[102], 0ll); // step_const
        csl::set(m_n[61], 0ll); // step_const
        csl::set(m_n[58], 2ll); // step_const
        csl::set(m_n[53], 0ll); // step_const
        csl::set(m_n[41], 1023ll); // step_const
        csl::set(m_n[94], 0ll); // step_const
        csl::set(m_n[68], 2047ll); // step_const
        csl::set(m_n[87], 1ll); // step_const
        csl::set(m_n[88], 0ll); // step_const
        csl::set(m_n[89], 0ll); // step_const
    }

    void execute_segment_0()
    {
        execute_segment_0_fragment_0();
    }

    void execute_segment_1_fragment_0()
    {
        csl::mask_lower(m_segment_temps_w[0], m_w[PORT_IN_00], 64, m_temps);
        csl::step_bit_extract(m_n[10], m_segment_temps_w[0], 52, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[7], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[10], 52, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 52, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 52);
        csl::step_reducing_and(m_n[13], m_segment_temps_n[3], 52, false, m_temps);
        csl::mask_lower(m_segment_temps_w[0], m_w[PORT_IN_00], 64, m_temps);
        csl::step_bit_extract(m_n[9], m_segment_temps_w[0], 11, true, 52, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[9], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[6], 11, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 11, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 11);
        csl::step_reducing_and(m_n[12], m_segment_temps_n[3], 11, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[12], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[13], 1, m_temps);
        csl::step_and(m_n[15], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[15], m_n[15], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[9], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[8], 11, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 11, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 11);
        csl::step_reducing_and(m_n[11], m_segment_temps_n[3], 11, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[13], 1, m_temps);
        csl::step_not_signed(m_n[14], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[12], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[14], 1, m_temps);
        csl::step_and(m_n[16], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[16], m_n[16], 1);
        csl::step_reduce(m_segment_temps_w[0], m_w[PORT_IN_00], 64, m_temps);
        csl::step_bit_extract(m_n[4], m_segment_temps_w[0], 1, true, 63, m_temps);
        csl::mask_lower(m_segment_temps_w[0], m_w[PORT_IN_00], 64, m_temps);
        csl::step_bit_extract(m_n[34], m_segment_temps_w[0], 52, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[34], 52, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[35], 1, m_temps);
        csl::step_bit_combine(m_n[36], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 52, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[36], 53, m_temps);
        csl::step_bit_extract(m_n[98], m_segment_temps_n[0], 27, true, 26, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[36], 26, m_temps);
        csl::step_bit_extract(m_n[114], m_segment_temps_n[0], 26, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[113], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[114], 26, m_temps);
        csl::step_bit_combine(m_n[115], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 1, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[115], 27, m_temps);
        csl::step_bit_extract(m_n[116], m_segment_temps_n[0], 27, false, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[36], 26, m_temps);
        csl::step_bit_extract(m_n[107], m_segment_temps_n[0], 26, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[106], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[107], 26, m_temps);
        csl::step_bit_combine(m_n[108], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 1, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[108], 27, m_temps);
        csl::step_bit_extract(m_n[109], m_segment_temps_n[0], 27, false, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[36], 53, m_temps);
        csl::step_bit_extract(m_n[101], m_segment_temps_n[0], 27, true, 26, m_temps);
        csl::mask_lower(m_segment_temps_w[0], m_w[PORT_IN_00], 64, m_temps);
        csl::step_bit_extract(m_n[2], m_segment_temps_w[0], 11, true, 52, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[12], 1, m_temps);
        csl::step_not_signed(m_n[17], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[11], 1, m_temps);
        csl::step_not_signed(m_n[18], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[18], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[17], 1, m_temps);
        csl::step_and(m_n[19], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[19], m_n[19], 1);
    }

    void execute_segment_1()
    {
        execute_segment_1_fragment_0();
    }

    void execute_segment_2_fragment_0()
    {
        csl::mask_lower(m_segment_temps_w[0], m_w[PORT_IN_11], 64, m_temps);
        csl::step_bit_extract(m_n[23], m_segment_temps_w[0], 11, true, 52, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[23], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[22], 11, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 11, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 11);
        csl::step_reducing_and(m_n[25], m_segment_temps_n[3], 11, false, m_temps);
        csl::mask_lower(m_segment_temps_w[0], m_w[PORT_IN_11], 64, m_temps);
        csl::step_bit_extract(m_n[24], m_segment_temps_w[0], 52, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[21], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[24], 52, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 52, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 52);
        csl::step_reducing_and(m_n[27], m_segment_temps_n[3], 52, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[23], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[20], 11, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 11, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 11);
        csl::step_reducing_and(m_n[26], m_segment_temps_n[3], 11, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[26], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[27], 1, m_temps);
        csl::step_and(m_n[29], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[29], m_n[29], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[27], 1, m_temps);
        csl::step_not_signed(m_n[28], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[26], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[28], 1, m_temps);
        csl::step_and(m_n[30], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[30], m_n[30], 1);
        csl::step_reduce(m_segment_temps_w[0], m_w[PORT_IN_11], 64, m_temps);
        csl::step_bit_extract(m_n[5], m_segment_temps_w[0], 1, true, 63, m_temps);
        csl::mask_lower(m_segment_temps_w[0], m_w[PORT_IN_11], 64, m_temps);
        csl::step_bit_extract(m_n[37], m_segment_temps_w[0], 52, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[37], 52, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[38], 1, m_temps);
        csl::step_bit_combine(m_n[39], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 52, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[39], 53, m_temps);
        csl::step_bit_extract(m_n[99], m_segment_temps_n[0], 27, true, 26, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[39], 26, m_temps);
        csl::step_bit_extract(m_n[118], m_segment_temps_n[0], 26, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[117], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[118], 26, m_temps);
        csl::step_bit_combine(m_n[119], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 1, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[119], 27, m_temps);
        csl::step_bit_extract(m_n[120], m_segment_temps_n[0], 27, false, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[39], 53, m_temps);
        csl::step_bit_extract(m_n[110], m_segment_temps_n[0], 27, true, 26, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[39], 26, m_temps);
        csl::step_bit_extract(m_n[103], m_segment_temps_n[0], 26, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[102], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[103], 26, m_temps);
        csl::step_bit_combine(m_n[104], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 1, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[104], 27, m_temps);
        csl::step_bit_extract(m_n[105], m_segment_temps_n[0], 27, false, 0, m_temps);
        csl::mask_lower(m_segment_temps_w[0], m_w[PORT_IN_11], 64, m_temps);
        csl::step_bit_extract(m_n[3], m_segment_temps_w[0], 11, true, 52, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[26], 1, m_temps);
        csl::step_not_signed(m_n[31], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[25], 1, m_temps);
        csl::step_not_signed(m_n[32], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[32], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[31], 1, m_temps);
        csl::step_and(m_n[33], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[33], m_n[33], 1);
    }

    void execute_segment_2()
    {
        execute_segment_2_fragment_0();
    }

    void execute_segment_3_fragment_0()
    {
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[25], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[15], 1, m_temps);
        csl::step_and(m_n[81], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[81], m_n[81], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[11], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[29], 1, m_temps);
        csl::step_and(m_n[82], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[82], m_n[82], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[82], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[81], 1, m_temps);
        csl::step_or(m_n[83], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[83], m_n[83], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[16], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[30], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[83], 1, m_temps);
        csl::step_or(m_n[84], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[84], m_n[84], 1);
        csl::step_or(m_n[84], m_n[84], m_segment_temps_n[3], 1, m_temps);
        csl::mask(m_n[84], m_n[84], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[84], 1, m_temps);
        csl::step_not_signed(m_n[96], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[4], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[5], 1, m_temps);
        csl::step_xor(m_n[43], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[43], m_n[43], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[43], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[96], 1, m_temps);
        csl::step_and(m_n[97], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[97], m_n[97], 1);
        csl::mask_lower(m_segment_temps_n[0], m_n[98], 27, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[99], 27, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_mul(m_n[100], m_segment_temps_n[0], m_segment_temps_n[1], m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[116], 27, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[120], 27, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_mul(m_n[121], m_segment_temps_n[0], m_segment_temps_n[1], m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[121], 54, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[100], 54, m_temps);
        csl::step_bit_combine(m_w[7], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 54, 0, m_temps);
        csl::mask_lower(m_segment_temps_w[0], m_w[7], 108, m_temps);
        csl::step_bit_extract(m_w[8], m_segment_temps_w[0], 81, true, 27, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[101], 27, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[105], 27, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_n[109], 27, m_temps);
        csl::mask_lower(m_segment_temps_n[5], m_n[110], 27, m_temps);
        csl::step_mul(m_segment_temps_n[6], m_segment_temps_n[2], m_segment_temps_n[3], m_temps);
        csl::step_mul(m_segment_temps_n[7], m_segment_temps_n[4], m_segment_temps_n[5], m_temps);
        csl::step_addsub(m_segment_temps_n[1], m_n[111], m_segment_temps_n[6], m_segment_temps_n[7], m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[111], 55, m_temps);
        csl::step_bit_extract(m_n[112], m_segment_temps_n[0], 55, true, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[112], 55, m_temps);
        csl::mask_lower(m_segment_temps_w[1], m_w[8], 81, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_add(m_w[9], m_segment_temps_n[0], m_segment_temps_w[1], m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_w[7], 27, m_temps);
        csl::step_bit_extract(m_n[122], m_segment_temps_n[0], 27, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[122], 27, m_temps);
        csl::step_reduce(m_segment_temps_w[1], m_w[9], 82, m_temps);
        csl::step_bit_combine(m_w[10], m_segment_temps_n[0], m_segment_temps_w[1], 2, 1, 27, 0, m_temps);
        csl::mask_lower(m_segment_temps_w[0], m_w[10], 108, m_temps);
        csl::step_bit_extract(m_w[11], m_segment_temps_w[0], 106, true, 2, m_temps);
        csl::step_reduce(m_segment_temps_w[0], m_w[11], 106, m_temps);
        csl::step_bit_extract(m_w[3], m_segment_temps_w[0], 106, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_w[0], m_w[3], 106, m_temps);
        csl::step_bit_extract(m_n[44], m_segment_temps_w[0], 1, true, 105, m_temps);
        csl::mask_lower(m_segment_temps_w[0], m_w[3], 105, m_temps);
        csl::step_bit_extract(m_n[46], m_segment_temps_w[0], 53, true, 52, m_temps);
        csl::mask_lower(m_segment_temps_w[0], m_w[3], 104, m_temps);
        csl::step_bit_extract(m_n[47], m_segment_temps_w[0], 53, true, 51, m_temps);
        csl::step_reduce(m_segment_temps_w[0], m_w[3], 106, m_temps);
        csl::step_bit_extract(m_n[45], m_segment_temps_w[0], 1, true, 105, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[45], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[47], 53, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[46], 53, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_206[2] = { 2, 3 };
        csl::set(m_n[48], m_segment_temps_n[csl::checked_array_value(mux_lookup_206, m_segment_temps_n[4])]);
        csl::mask_lower(m_segment_temps_n[0], m_n[48], 2, m_temps);
        csl::step_bit_extract(m_n[56], m_segment_temps_n[0], 2, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_w[3], 52, m_temps);
    }

    void execute_segment_3_fragment_1()
    {
        csl::step_bit_extract(m_n[50], m_segment_temps_n[0], 1, true, 51, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[44], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[0], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[50], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_217[2] = { 2, 3 };
        csl::set(m_n[51], m_segment_temps_n[csl::checked_array_value(mux_lookup_217, m_segment_temps_n[4])]);
        csl::mask_lower(m_segment_temps_n[0], m_w[3], 51, m_temps);
        csl::step_bit_extract(m_n[49], m_segment_temps_n[0], 51, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[49], 51, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[51], 1, m_temps);
        csl::step_bit_combine(m_n[52], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 51, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[52], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[53], 52, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 52, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 52);
        csl::step_reducing_and(m_n[54], m_segment_temps_n[3], 52, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[54], 1, m_temps);
        csl::step_not_signed(m_n[55], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[55], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[56], 2, m_temps);
        csl::step_bit_combine(m_n[57], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 1, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[57], 3, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[58], 3, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 3, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 3);
        csl::step_reducing_and(m_n[59], m_segment_temps_n[3], 3, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[59], 1, m_temps);
        csl::step_not_signed(m_n[60], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[60], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[61], 52, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[44], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[3], m_n[0], 1, m_temps);
        csl::step_bit_combine(m_n[62], m_segment_temps_n[0], m_segment_temps_n[1], 4, 1, 1, 53, m_temps);
        csl::step_bit_combine(m_n[62], m_n[62], m_segment_temps_n[2], 4, 2, 53, 54, m_temps);
        csl::step_bit_combine(m_n[62], m_n[62], m_segment_temps_n[3], 4, 3, 54, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[2], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[3], 11, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_add(m_n[40], m_segment_temps_n[0], m_segment_temps_n[1], m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[40], 12, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[41], 13, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_sub(m_n[42], m_segment_temps_n[0], m_segment_temps_n[1], m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[48], 53, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[42], 14, m_temps);
        csl::step_bit_combine(m_w[4], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 53, 0, m_temps);
        csl::step_reduce(m_segment_temps_w[0], m_w[4], 67, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[62], 55, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_add(m_w[5], m_segment_temps_w[0], m_segment_temps_n[1], m_temps);
        csl::step_reduce(m_segment_temps_w[0], m_w[5], 68, m_temps);
        csl::step_bit_extract(m_n[64], m_segment_temps_w[0], 15, true, 53, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[64], 11, m_temps);
        csl::step_bit_extract(m_n[65], m_segment_temps_n[0], 11, true, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[65], 11, m_temps);
        csl::step_bit_extract(m_n[93], m_segment_temps_n[0], 11, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[64], 15, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[68], 11, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[0], 1, m_temps);
        csl::step_add(m_segment_temps_n[4], m_segment_temps_n[1], m_segment_temps_n[3], m_temps);
        csl::step_sub(m_n[70], m_segment_temps_n[0], m_segment_temps_n[4], m_temps);
        csl::step_nsign_bit(m_n[69], m_n[70], 17);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[19], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[33], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[69], 1, m_temps);
        csl::step_and(m_n[79], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[79], m_n[79], 1);
        csl::step_and(m_n[79], m_n[79], m_segment_temps_n[3], 1, m_temps);
        csl::mask(m_n[79], m_n[79], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[33], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[15], 1, m_temps);
        csl::step_and(m_n[78], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[78], m_n[78], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[19], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[29], 1, m_temps);
        csl::step_and(m_n[77], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[77], m_n[77], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[15], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[29], 1, m_temps);
        csl::step_and(m_n[76], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[76], m_n[76], 1);
    }

    void execute_segment_3_fragment_2()
    {
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[76], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[77], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[78], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_n[79], 1, m_temps);
        csl::step_or(m_n[80], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[80], m_n[80], 1);
        csl::step_or(m_n[80], m_n[80], m_segment_temps_n[3], 1, m_temps);
        csl::mask(m_n[80], m_n[80], 1);
        csl::step_or(m_n[80], m_n[80], m_segment_temps_n[4], 1, m_temps);
        csl::mask(m_n[80], m_n[80], 1);
        csl::mask_lower(m_segment_temps_n[0], m_n[0], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[64], 15, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[0], 1, m_temps);
        csl::step_add(m_segment_temps_n[4], m_segment_temps_n[1], m_segment_temps_n[3], m_temps);
        csl::step_sub(m_n[67], m_segment_temps_n[0], m_segment_temps_n[4], m_temps);
        csl::step_nsign_bit(m_n[66], m_n[67], 17);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[19], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[33], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[66], 1, m_temps);
        csl::step_and(m_n[74], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[74], m_n[74], 1);
        csl::step_and(m_n[74], m_n[74], m_segment_temps_n[3], 1, m_temps);
        csl::mask(m_n[74], m_n[74], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[25], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[19], 1, m_temps);
        csl::step_and(m_n[73], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[73], m_n[73], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[11], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[33], 1, m_temps);
        csl::step_and(m_n[72], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[72], m_n[72], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[11], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[25], 1, m_temps);
        csl::step_and(m_n[71], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[71], m_n[71], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[71], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[72], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[73], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_n[74], 1, m_temps);
        csl::step_or(m_n[75], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[75], m_n[75], 1);
        csl::step_or(m_n[75], m_n[75], m_segment_temps_n[3], 1, m_temps);
        csl::mask(m_n[75], m_n[75], 1);
        csl::step_or(m_n[75], m_n[75], m_segment_temps_n[4], 1, m_temps);
        csl::mask(m_n[75], m_n[75], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[75], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[80], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[84], 1, m_temps);
        csl::step_bit_combine(m_n[85], m_segment_temps_n[0], m_segment_temps_n[1], 3, 1, 1, 2, m_temps);
        csl::step_bit_combine(m_n[85], m_n[85], m_segment_temps_n[2], 3, 2, 2, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[85], 3, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[1], 1, m_temps);
        static constexpr int64_t lut_init_data_361[] = {
            1ll, 0ll, 2ll, 0ll, 3ll, 0ll, 0ll, 0ll
        };
        csl::step_lookup(lut_init_data_361, 8, 0, m_n[86], m_segment_temps_n[0], m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[86], 2, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[94], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[93], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_n[92], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[5], m_n[91], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[6], m_segment_temps_n[1], 2, m_temps);
        static constexpr size_t mux_lookup_369[4] = { 2, 3, 4, 5 };
        csl::set(m_n[95], m_segment_temps_n[csl::checked_array_value(mux_lookup_369, m_segment_temps_n[6])]);
        csl::mask_lower(m_segment_temps_n[0], m_w[5], 53, m_temps);
        csl::step_bit_extract(m_n[63], m_segment_temps_n[0], 52, true, 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[86], 2, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[89], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[63], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_n[88], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[5], m_n[87], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[6], m_segment_temps_n[1], 2, m_temps);
        static constexpr size_t mux_lookup_382[4] = { 2, 3, 4, 5 };
        csl::set(m_n[90], m_segment_temps_n[csl::checked_array_value(mux_lookup_382, m_segment_temps_n[6])]);
        csl::step_reduce(m_segment_temps_n[0], m_n[90], 52, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[95], 11, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[97], 1, m_temps);
        csl::step_bit_combine(m_w[6], m_segment_temps_n[0], m_segment_temps_n[1], 3, 1, 52, 63, m_temps);
        csl::step_bit_combine(m_w[6], m_w[6], m_segment_temps_n[2], 3, 2, 63, 0, m_temps);
        csl::step_reduce(m_segment_temps_w[0], m_w[6], 64, m_temps);
        csl::step_bit_extract(m_w[0], m_segment_temps_w[0], 64, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_w[0], m_w[0], 64, m_temps);
        m_w[PORT_OUT_PRIMWIREOUT2] = m_segment_temps_w[0];
    }

    void execute_segment_3()
    {
        execute_segment_3_fragment_0();
        execute_segment_3_fragment_1();
        execute_segment_3_fragment_2();
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
        if ((m_segment_cycle[1] == m_update_cycle) && (m_io_cycle[0] > m_update_cycle))
        {
            execute_segment_1();
            ++m_segment_cycle[1];
        }
        if ((m_segment_cycle[2] == m_update_cycle) && (m_io_cycle[1] > m_update_cycle))
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
    static constexpr size_t PORT_OUT_PRIMWIREOUT2 = 14;

    int64_t m_io_cycle[3];
    int64_t m_segment_cycle[4];
    int64_t m_update_cycle;

    int64_t m_segment_temps_n[8] = { 0 };
    csl::mp_int m_segment_temps_w[2];
    csl::mp_int_temps m_temps;
    int64_t m_n[127];
    csl::mp_int m_w[15];
};

#endif // SOFTWARE_MODEL_FLT_FFT_EXAMPLE_DUT_MULTBLOCK_TYPESFLOATIEEE_52_11_TYPESFLOATIEEE_52_11_TYPESFLO0000OF0CDJ6OF0CD16OL0QCZ_H_