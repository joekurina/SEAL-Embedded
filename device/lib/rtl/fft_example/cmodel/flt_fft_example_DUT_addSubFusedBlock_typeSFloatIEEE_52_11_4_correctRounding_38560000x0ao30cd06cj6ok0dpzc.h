// ------------------------------------------------------------------------- 
// High Level Design Compiler for Altera(R) FPGAs Version 25.1 (Release Build #6a12354d2f)
// Software model created on 2025-07-08 08:39:19
// Generation mode: Bit Accurate
// ------------------------------------------------------------------------- 
#pragma once

#ifndef SOFTWARE_MODEL_FLT_FFT_EXAMPLE_DUT_ADDSUBFUSEDBLOCK_TYPESFLOATIEEE_52_11_4_CORRECTROUNDING_38560000X0AO30CD06CJ6OK0DPZC_H_
#define SOFTWARE_MODEL_FLT_FFT_EXAMPLE_DUT_ADDSUBFUSEDBLOCK_TYPESFLOATIEEE_52_11_4_CORRECTROUNDING_38560000X0AO30CD06CJ6OK0DPZC_H_

#include "support/csl.h"
#ifdef WRITE_STM_FILES
#include "support/csl_io.h"
#endif

class flt_fft_example_DUT_addSubFusedBlock_typeSFloatIEEE_52_11_4_correctRounding_38560000x0ao30cd06cj6ok0dpzc
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

    // IO struct for "out_primWireAux"
    struct io_struct_out_primWireAux
    {
        double port_out_primwireaux = 0;
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
    void read(io_struct_out_primWireAux& output)
    {
        bool needs_to_execute = (m_io_cycle[2] == m_update_cycle);
        if (needs_to_execute)
        {
            m_io_cycle[2]++;
            execute();
        }

        mask_lower(output.port_out_primwireaux, m_w[PORT_OUT_PRIMWIREAUX2], 64, m_temps);
    }

    void read(io_struct_out_primWireOut& output)
    {
        bool needs_to_execute = (m_io_cycle[3] == m_update_cycle);
        if (needs_to_execute)
        {
            m_io_cycle[3]++;
            execute();
        }

        mask_lower(output.port_out_primwireout, m_w[PORT_OUT_PRIMWIREOUT3], 64, m_temps);
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
        static constexpr size_t native_reset_range_sizes[] = { 241 };
        for (size_t i = 0; i < 1; ++i)
        {
            csl::fill_n(&m_n[native_reset_range_indices[i]], native_reset_range_sizes[i], native_reset_range_values[native_reset_range_value_indices[i]]);
        }
        static constexpr uint64_t wide_reset_range_values[] = { 0 };
        static constexpr csl::mp_int_info wide_reset_range_infos[] = { { 0, 1, 0 } };
        static constexpr uint32_t wide_reset_range_value_indices[] = { 0 };
        static constexpr size_t wide_reset_range_indices[] = { 0 };
        static constexpr size_t wide_reset_range_sizes[] = { 20 };
        for (size_t i = 0; i < 1; ++i)
        {
            for (size_t j = 0; j < wide_reset_range_sizes[i]; ++j)
            {
                csl::fill_mpz_data(m_w[wide_reset_range_indices[i] + j],
                    wide_reset_range_values, wide_reset_range_infos, wide_reset_range_value_indices[i]);
            }
        }
        csl::fill_n(m_io_cycle, 4, -1);
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
            m_io_cycle[3] = m_update_cycle + 1;
            execute();
        }
    }

private:
    // Segments are chunks of execution that depend on unique sets of inputs
    // These functions are invoked automatically as inputs are provided to the model.
    void execute_segment_0_fragment_0()
    {
        csl::set(m_n[0], 0ll); // step_const
        csl::set(m_n[1], 1ll); // step_const
        csl::set(m_n[22], 0ll); // step_const
        csl::set(m_n[8], 0ll); // step_const
        csl::set(m_n[75], 57ll); // step_const
        csl::set(m_n[197], 0ll); // step_const
        csl::set(m_n[41], 0ll); // step_const
        csl::set(m_n[44], 55ll); // step_const
        csl::set(m_n[45], 54ll); // step_const
        csl::set(m_n[57], 0ll); // step_const
        csl::set(m_n[59], 0ll); // step_const
        csl::set(m_n[42], 1ll); // step_const
        csl::set(m_n[64], 0ll); // step_const
        csl::set(m_n[205], 0ll); // step_const
        csl::set(m_n[200], -1ll); // step_const
        csl::set(m_n[211], 0ll); // step_const
        csl::set(m_n[217], 0ll); // step_const
        csl::set(m_n[223], 0ll); // step_const
        csl::set(m_n[229], 0ll); // step_const
        csl::set(m_n[20], 2047ll); // step_const
        csl::set(m_n[6], 2047ll); // step_const
        csl::set(m_n[21], 0ll); // step_const
        csl::set(m_n[7], 0ll); // step_const
        csl::set(m_n[135], 2047ll); // step_const
        csl::set(m_n[136], 2047ll); // step_const
        csl::set(m_n[88], 8ll); // step_const
        csl::set(m_n[161], 0ll); // step_const
        csl::set(m_n[169], 0ll); // step_const
        csl::set(m_n[164], -1ll); // step_const
        csl::set(m_n[175], 0ll); // step_const
        csl::set(m_n[181], 0ll); // step_const
        csl::set(m_n[187], 0ll); // step_const
        csl::set(m_n[193], 0ll); // step_const
        csl::set(m_n[137], 0ll); // step_const
        csl::set(m_n[99], 2047ll); // step_const
        csl::set(m_n[131], 1ll); // step_const
        csl::set(m_n[132], 0ll); // step_const
        csl::set(m_n[133], 0ll); // step_const
        csl::set(m_n[149], 2047ll); // step_const
        csl::set(m_n[150], 2047ll); // step_const
        csl::set(m_n[151], 0ll); // step_const
        csl::set(m_n[145], 1ll); // step_const
        csl::set(m_n[146], 0ll); // step_const
        csl::set(m_n[147], 0ll); // step_const
    }

    void execute_segment_0()
    {
        execute_segment_0_fragment_0();
    }

    void execute_segment_1_fragment_0()
    {
        csl::mask_lower(m_segment_temps_w[0], m_w[PORT_IN_11], 64, m_temps);
        csl::step_bit_extract(m_n[3], m_segment_temps_w[0], 63, true, 0, m_temps);
    }

    void execute_segment_1()
    {
        execute_segment_1_fragment_0();
    }

    void execute_segment_2_fragment_0()
    {
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
        csl::step_sub(m_w[4], m_segment_temps_n[0], m_segment_temps_w[0], m_temps);
        csl::step_nsign_bit(m_n[4], m_w[4], 65);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[4], 1, m_temps);
        csl::step_reduce(m_segment_temps_w[1], m_w[PORT_IN_00], 64, m_temps);
        csl::step_reduce(m_segment_temps_w[2], m_w[PORT_IN_11], 64, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_18[2] = { 1, 2 };
        csl::set(m_w[6], m_segment_temps_w[csl::checked_array_value(mux_lookup_18, m_segment_temps_n[4])]);
        csl::step_reduce(m_segment_temps_w[3], m_w[6], 64, m_temps);
        csl::step_bit_extract(m_n[39], m_segment_temps_w[3], 1, true, 63, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[4], 1, m_temps);
        csl::step_reduce(m_segment_temps_w[1], m_w[PORT_IN_11], 64, m_temps);
        csl::step_reduce(m_segment_temps_w[2], m_w[PORT_IN_00], 64, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_26[2] = { 1, 2 };
        csl::set(m_w[5], m_segment_temps_w[csl::checked_array_value(mux_lookup_26, m_segment_temps_n[4])]);
        csl::step_reduce(m_segment_temps_w[3], m_w[5], 64, m_temps);
        csl::step_bit_extract(m_n[38], m_segment_temps_w[3], 1, true, 63, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[38], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[39], 1, m_temps);
        csl::step_xor(m_n[40], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[40], m_n[40], 1);
        csl::mask_lower(m_segment_temps_n[0], m_w[6], 63, m_temps);
        csl::step_bit_extract(m_n[23], m_segment_temps_n[0], 11, true, 52, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[23], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[22], 11, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 11, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 11);
        csl::step_reducing_and(m_n[25], m_segment_temps_n[3], 11, false, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_w[5], 63, m_temps);
        csl::step_bit_extract(m_n[9], m_segment_temps_n[0], 11, true, 52, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[9], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[8], 11, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 11, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 11);
        csl::step_reducing_and(m_n[11], m_segment_temps_n[3], 11, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[11], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[25], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[40], 1, m_temps);
        csl::step_and(m_n[139], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[139], m_n[139], 1);
        csl::step_and(m_n[139], m_n[139], m_segment_temps_n[3], 1, m_temps);
        csl::mask(m_n[139], m_n[139], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[139], 1, m_temps);
        csl::step_not_signed(m_n[140], m_segment_temps_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_w[6], 63, m_temps);
        csl::step_bit_extract(m_n[35], m_segment_temps_n[0], 11, true, 52, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_w[5], 63, m_temps);
        csl::step_bit_extract(m_n[34], m_segment_temps_n[0], 11, true, 52, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[34], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[35], 11, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_sub(m_n[43], m_segment_temps_n[0], m_segment_temps_n[1], m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[43], 6, m_temps);
        csl::step_bit_extract(m_n[52], m_segment_temps_n[0], 6, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[35], 11, m_temps);
        csl::step_reducing_or(m_n[47], m_segment_temps_n[1]);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[47], 1, m_temps);
        csl::step_not_signed(m_n[48], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[43], 12, m_temps);
        csl::step_bit_extract(m_n[46], m_segment_temps_n[0], 12, false, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[45], 6, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[46], 12, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[0], 1, m_temps);
        csl::step_add(m_segment_temps_n[4], m_segment_temps_n[1], m_segment_temps_n[3], m_temps);
        csl::step_sub(m_n[50], m_segment_temps_n[0], m_segment_temps_n[4], m_temps);
        csl::step_sign_bit(m_n[49], m_n[50], 14);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[49], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[48], 1, m_temps);
        csl::step_or(m_n[51], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[51], m_n[51], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[51], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[52], 6, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[44], 6, m_temps);
    }

    void execute_segment_3_fragment_1()
    {
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_96[2] = { 2, 3 };
        csl::set(m_n[53], m_segment_temps_n[csl::checked_array_value(mux_lookup_96, m_segment_temps_n[4])]);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[25], 1, m_temps);
        csl::step_not_signed(m_n[54], m_segment_temps_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_w[6], 52, m_temps);
        csl::step_bit_extract(m_n[37], m_segment_temps_n[0], 52, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[37], 52, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[54], 1, m_temps);
        csl::step_bit_combine(m_n[55], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 52, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[57], 55, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[55], 53, m_temps);
        csl::step_bit_combine(m_w[7], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 55, 0, m_temps);
        csl::mask_lower(m_segment_temps_w[3], m_w[7], 108, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[53], 6, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_ld_exp(m_w[8], m_segment_temps_w[3], m_segment_temps_n[1], true, m_temps);
        csl::mask_lower(m_segment_temps_w[3], m_w[8], 108, m_temps);
        csl::step_bit_extract(m_n[62], m_segment_temps_w[3], 55, true, 53, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_w[8], 53, m_temps);
        csl::step_bit_extract(m_n[58], m_segment_temps_n[0], 53, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[58], 53, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[59], 52, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 53, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 53);
        csl::step_reducing_and(m_n[60], m_segment_temps_n[3], 53, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[60], 1, m_temps);
        csl::step_not_signed(m_n[61], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[61], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[62], 55, m_temps);
        csl::step_bit_combine(m_n[63], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 1, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[63], 56, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[41], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[41], 1, m_temps);
        csl::step_bit_combine(m_n[66], m_segment_temps_n[0], m_segment_temps_n[1], 3, 1, 56, 57, m_temps);
        csl::step_bit_combine(m_n[66], m_n[66], m_segment_temps_n[2], 3, 2, 57, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_w[5], 52, m_temps);
        csl::step_bit_extract(m_n[36], m_segment_temps_n[0], 52, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[36], 52, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[42], 1, m_temps);
        csl::step_bit_combine(m_n[56], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 52, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[64], 3, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[56], 53, m_temps);
        csl::step_bit_combine(m_n[65], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 3, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[65], 56, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[66], 58, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_sub(m_n[68], m_segment_temps_n[0], m_segment_temps_n[1], m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[68], 57, m_temps);
        csl::step_bit_extract(m_n[70], m_segment_temps_n[0], 57, true, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[70], 57, m_temps);
        csl::step_bit_extract(m_n[198], m_segment_temps_n[0], 32, true, 25, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[198], 32, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[197], 32, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 32, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 32);
        csl::step_reducing_and(m_n[199], m_segment_temps_n[3], 32, false, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[70], 25, m_temps);
        csl::step_bit_extract(m_n[201], m_segment_temps_n[0], 25, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[200], 7, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[201], 25, m_temps);
        csl::step_bit_combine(m_n[202], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 7, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[70], 57, m_temps);
        csl::step_bit_extract(m_n[203], m_segment_temps_n[0], 32, true, 25, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[199], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[203], 32, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[202], 32, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_170[2] = { 2, 3 };
        csl::set(m_n[204], m_segment_temps_n[csl::checked_array_value(mux_lookup_170, m_segment_temps_n[4])]);
        csl::mask_lower(m_segment_temps_n[0], m_n[204], 32, m_temps);
        csl::step_bit_extract(m_n[206], m_segment_temps_n[0], 16, true, 16, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[206], 16, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[205], 16, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 16, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 16);
        csl::step_reducing_and(m_n[207], m_segment_temps_n[3], 16, false, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[204], 16, m_temps);
        csl::step_bit_extract(m_n[208], m_segment_temps_n[0], 16, true, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[204], 32, m_temps);
        csl::step_bit_extract(m_n[209], m_segment_temps_n[0], 16, true, 16, m_temps);
    }

    void execute_segment_3_fragment_2()
    {
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[207], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[209], 16, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[208], 16, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_188[2] = { 2, 3 };
        csl::set(m_n[210], m_segment_temps_n[csl::checked_array_value(mux_lookup_188, m_segment_temps_n[4])]);
        csl::mask_lower(m_segment_temps_n[0], m_n[210], 16, m_temps);
        csl::step_bit_extract(m_n[212], m_segment_temps_n[0], 8, true, 8, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[212], 8, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[211], 8, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 8, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 8);
        csl::step_reducing_and(m_n[213], m_segment_temps_n[3], 8, false, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[210], 8, m_temps);
        csl::step_bit_extract(m_n[214], m_segment_temps_n[0], 8, true, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[210], 16, m_temps);
        csl::step_bit_extract(m_n[215], m_segment_temps_n[0], 8, true, 8, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[213], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[215], 8, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[214], 8, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_206[2] = { 2, 3 };
        csl::set(m_n[216], m_segment_temps_n[csl::checked_array_value(mux_lookup_206, m_segment_temps_n[4])]);
        csl::mask_lower(m_segment_temps_n[0], m_n[216], 8, m_temps);
        csl::step_bit_extract(m_n[218], m_segment_temps_n[0], 4, true, 4, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[218], 4, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[217], 4, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 4, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 4);
        csl::step_reducing_and(m_n[219], m_segment_temps_n[3], 4, false, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[216], 4, m_temps);
        csl::step_bit_extract(m_n[220], m_segment_temps_n[0], 4, true, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[216], 8, m_temps);
        csl::step_bit_extract(m_n[221], m_segment_temps_n[0], 4, true, 4, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[219], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[221], 4, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[220], 4, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_224[2] = { 2, 3 };
        csl::set(m_n[222], m_segment_temps_n[csl::checked_array_value(mux_lookup_224, m_segment_temps_n[4])]);
        csl::mask_lower(m_segment_temps_n[0], m_n[222], 4, m_temps);
        csl::step_bit_extract(m_n[224], m_segment_temps_n[0], 2, true, 2, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[224], 2, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[223], 2, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 2, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 2);
        csl::step_reducing_and(m_n[225], m_segment_temps_n[3], 2, false, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[222], 2, m_temps);
        csl::step_bit_extract(m_n[226], m_segment_temps_n[0], 2, true, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[222], 4, m_temps);
        csl::step_bit_extract(m_n[227], m_segment_temps_n[0], 2, true, 2, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[225], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[227], 2, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[226], 2, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_242[2] = { 2, 3 };
        csl::set(m_n[228], m_segment_temps_n[csl::checked_array_value(mux_lookup_242, m_segment_temps_n[4])]);
        csl::mask_lower(m_segment_temps_n[0], m_n[228], 2, m_temps);
        csl::step_bit_extract(m_n[230], m_segment_temps_n[0], 1, true, 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[230], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[229], 1, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 1);
        csl::step_reducing_and(m_n[231], m_segment_temps_n[3], 1, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[231], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[225], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[219], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[3], m_n[213], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[4], m_n[207], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[5], m_n[199], 1, m_temps);
        csl::step_bit_combine(m_n[232], m_segment_temps_n[0], m_segment_temps_n[1], 6, 1, 1, 2, m_temps);
        csl::step_bit_combine(m_n[232], m_n[232], m_segment_temps_n[2], 6, 2, 2, 3, m_temps);
        csl::step_bit_combine(m_n[232], m_n[232], m_segment_temps_n[3], 6, 3, 3, 4, m_temps);
        csl::step_bit_combine(m_n[232], m_n[232], m_segment_temps_n[4], 6, 4, 4, 5, m_temps);
        csl::step_bit_combine(m_n[232], m_n[232], m_segment_temps_n[5], 6, 5, 5, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[232], 6, m_temps);
        csl::step_bit_extract(m_n[73], m_segment_temps_n[0], 6, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[73], 6, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[75], 6, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 6, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 6);
        csl::step_reducing_and(m_n[76], m_segment_temps_n[3], 6, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[23], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[20], 11, m_temps);
    }

    void execute_segment_3_fragment_3()
    {
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 11, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 11);
        csl::step_reducing_and(m_n[26], m_segment_temps_n[3], 11, false, m_temps);
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
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[9], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[6], 11, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 11, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 11);
        csl::step_reducing_and(m_n[12], m_segment_temps_n[3], 11, false, m_temps);
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
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[19], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[33], 1, m_temps);
        csl::step_and(m_n[110], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[110], m_n[110], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[110], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[76], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[40], 1, m_temps);
        csl::step_and(m_n[141], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[141], m_n[141], 1);
        csl::step_and(m_n[141], m_n[141], m_segment_temps_n[3], 1, m_temps);
        csl::mask(m_n[141], m_n[141], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[141], 1, m_temps);
        csl::step_not_signed(m_n[142], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_w[3], m_w[5], 64, m_temps);
        csl::step_bit_extract(m_n[107], m_segment_temps_w[3], 1, true, 63, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_w[6], 52, m_temps);
        csl::step_bit_extract(m_n[24], m_segment_temps_n[0], 52, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[21], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[24], 52, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 52, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 52);
        csl::step_reducing_and(m_n[27], m_segment_temps_n[3], 52, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[27], 1, m_temps);
        csl::step_not_signed(m_n[28], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[26], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[28], 1, m_temps);
        csl::step_and(m_n[30], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[30], m_n[30], 1);
        csl::mask_lower(m_segment_temps_n[0], m_w[5], 52, m_temps);
        csl::step_bit_extract(m_n[10], m_segment_temps_n[0], 52, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[7], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[10], 52, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 52, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 52);
        csl::step_reducing_and(m_n[13], m_segment_temps_n[3], 52, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[13], 1, m_temps);
        csl::step_not_signed(m_n[14], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[12], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[14], 1, m_temps);
        csl::step_and(m_n[16], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[16], m_n[16], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[16], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[30], 1, m_temps);
        csl::step_or(m_n[114], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[114], m_n[114], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[26], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[27], 1, m_temps);
        csl::step_and(m_n[29], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[29], m_n[29], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[12], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[13], 1, m_temps);
        csl::step_and(m_n[15], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[15], m_n[15], 1);
    }

    void execute_segment_3_fragment_4()
    {
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[15], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[29], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[40], 1, m_temps);
        csl::step_and(m_n[118], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[118], m_n[118], 1);
        csl::step_and(m_n[118], m_n[118], m_segment_temps_n[3], 1, m_temps);
        csl::mask(m_n[118], m_n[118], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[118], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[114], 1, m_temps);
        csl::step_or(m_n[119], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[119], m_n[119], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[119], 1, m_temps);
        csl::step_not_signed(m_n[143], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[143], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[107], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[142], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_n[140], 1, m_temps);
        csl::step_and(m_n[144], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[144], m_n[144], 1);
        csl::step_and(m_n[144], m_n[144], m_segment_temps_n[3], 1, m_temps);
        csl::mask(m_n[144], m_n[144], 1);
        csl::step_and(m_n[144], m_n[144], m_segment_temps_n[4], 1, m_temps);
        csl::mask(m_n[144], m_n[144], 1);
        csl::mask_lower(m_segment_temps_n[0], m_n[70], 57, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[73], 6, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_ld_exp(m_n[74], m_segment_temps_n[0], m_segment_temps_n[1], false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[74], 5, m_temps);
        csl::step_bit_extract(m_n[95], m_segment_temps_n[0], 1, true, 4, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[74], 4, m_temps);
        csl::step_bit_extract(m_n[94], m_segment_temps_n[0], 1, true, 3, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[74], 3, m_temps);
        csl::step_bit_extract(m_n[93], m_segment_temps_n[0], 1, true, 2, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[74], 2, m_temps);
        csl::step_bit_extract(m_n[92], m_segment_temps_n[0], 1, true, 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[74], 1, m_temps);
        csl::step_bit_extract(m_n[91], m_segment_temps_n[0], 1, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[91], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[92], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[93], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[3], m_n[94], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[4], m_n[95], 1, m_temps);
        csl::step_bit_combine(m_n[96], m_segment_temps_n[0], m_segment_temps_n[1], 5, 1, 1, 2, m_temps);
        csl::step_bit_combine(m_n[96], m_n[96], m_segment_temps_n[2], 5, 2, 2, 3, m_temps);
        csl::step_bit_combine(m_n[96], m_n[96], m_segment_temps_n[3], 5, 3, 3, 4, m_temps);
        csl::step_bit_combine(m_n[96], m_n[96], m_segment_temps_n[4], 5, 4, 4, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[96], 5, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[88], 5, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 5, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 5);
        csl::step_reducing_and(m_n[97], m_segment_temps_n[3], 5, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[97], 1, m_temps);
        csl::step_not_signed(m_n[98], m_segment_temps_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[34], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[42], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_add(m_n[77], m_segment_temps_n[0], m_segment_temps_n[1], m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[77], 12, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[73], 6, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_sub(m_n[78], m_segment_temps_n[0], m_segment_temps_n[1], m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[74], 56, m_temps);
        csl::step_bit_extract(m_n[80], m_segment_temps_n[0], 53, true, 3, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[80], 53, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[78], 13, m_temps);
        csl::step_bit_combine(m_w[9], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 53, 0, m_temps);
        csl::mask_lower(m_segment_temps_w[3], m_w[9], 66, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[98], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_add(m_w[12], m_segment_temps_w[3], m_segment_temps_n[1], m_temps);
        csl::mask_lower(m_segment_temps_w[3], m_w[12], 64, m_temps);
        csl::step_bit_extract(m_n[109], m_segment_temps_w[3], 11, true, 53, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[65], 56, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[66], 58, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_add(m_n[67], m_segment_temps_n[0], m_segment_temps_n[1], m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[67], 57, m_temps);
        csl::step_bit_extract(m_n[69], m_segment_temps_n[0], 57, true, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[69], 57, m_temps);
        csl::step_bit_extract(m_n[162], m_segment_temps_n[0], 32, true, 25, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[162], 32, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[161], 32, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 32, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 32);
        csl::step_reducing_and(m_n[163], m_segment_temps_n[3], 32, false, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[69], 25, m_temps);
        csl::step_bit_extract(m_n[165], m_segment_temps_n[0], 25, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[164], 7, m_temps);
    }

    void execute_segment_3_fragment_5()
    {
        csl::step_reduce(m_segment_temps_n[1], m_n[165], 25, m_temps);
        csl::step_bit_combine(m_n[166], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 7, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[69], 57, m_temps);
        csl::step_bit_extract(m_n[167], m_segment_temps_n[0], 32, true, 25, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[163], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[167], 32, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[166], 32, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_448[2] = { 2, 3 };
        csl::set(m_n[168], m_segment_temps_n[csl::checked_array_value(mux_lookup_448, m_segment_temps_n[4])]);
        csl::mask_lower(m_segment_temps_n[0], m_n[168], 32, m_temps);
        csl::step_bit_extract(m_n[170], m_segment_temps_n[0], 16, true, 16, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[170], 16, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[169], 16, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 16, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 16);
        csl::step_reducing_and(m_n[171], m_segment_temps_n[3], 16, false, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[168], 16, m_temps);
        csl::step_bit_extract(m_n[172], m_segment_temps_n[0], 16, true, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[168], 32, m_temps);
        csl::step_bit_extract(m_n[173], m_segment_temps_n[0], 16, true, 16, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[171], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[173], 16, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[172], 16, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_466[2] = { 2, 3 };
        csl::set(m_n[174], m_segment_temps_n[csl::checked_array_value(mux_lookup_466, m_segment_temps_n[4])]);
        csl::mask_lower(m_segment_temps_n[0], m_n[174], 16, m_temps);
        csl::step_bit_extract(m_n[176], m_segment_temps_n[0], 8, true, 8, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[176], 8, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[175], 8, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 8, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 8);
        csl::step_reducing_and(m_n[177], m_segment_temps_n[3], 8, false, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[174], 8, m_temps);
        csl::step_bit_extract(m_n[178], m_segment_temps_n[0], 8, true, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[174], 16, m_temps);
        csl::step_bit_extract(m_n[179], m_segment_temps_n[0], 8, true, 8, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[177], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[179], 8, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[178], 8, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_484[2] = { 2, 3 };
        csl::set(m_n[180], m_segment_temps_n[csl::checked_array_value(mux_lookup_484, m_segment_temps_n[4])]);
        csl::mask_lower(m_segment_temps_n[0], m_n[180], 8, m_temps);
        csl::step_bit_extract(m_n[182], m_segment_temps_n[0], 4, true, 4, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[182], 4, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[181], 4, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 4, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 4);
        csl::step_reducing_and(m_n[183], m_segment_temps_n[3], 4, false, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[180], 4, m_temps);
        csl::step_bit_extract(m_n[184], m_segment_temps_n[0], 4, true, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[180], 8, m_temps);
        csl::step_bit_extract(m_n[185], m_segment_temps_n[0], 4, true, 4, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[183], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[185], 4, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[184], 4, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_502[2] = { 2, 3 };
        csl::set(m_n[186], m_segment_temps_n[csl::checked_array_value(mux_lookup_502, m_segment_temps_n[4])]);
        csl::mask_lower(m_segment_temps_n[0], m_n[186], 4, m_temps);
        csl::step_bit_extract(m_n[188], m_segment_temps_n[0], 2, true, 2, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[188], 2, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[187], 2, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 2, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 2);
        csl::step_reducing_and(m_n[189], m_segment_temps_n[3], 2, false, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[186], 2, m_temps);
        csl::step_bit_extract(m_n[190], m_segment_temps_n[0], 2, true, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[186], 4, m_temps);
        csl::step_bit_extract(m_n[191], m_segment_temps_n[0], 2, true, 2, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[189], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[191], 2, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[190], 2, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_520[2] = { 2, 3 };
        csl::set(m_n[192], m_segment_temps_n[csl::checked_array_value(mux_lookup_520, m_segment_temps_n[4])]);
        csl::mask_lower(m_segment_temps_n[0], m_n[192], 2, m_temps);
        csl::step_bit_extract(m_n[194], m_segment_temps_n[0], 1, true, 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
    }

    void execute_segment_3_fragment_6()
    {
        csl::mask_lower(m_segment_temps_n[1], m_n[194], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[193], 1, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 1);
        csl::step_reducing_and(m_n[195], m_segment_temps_n[3], 1, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[195], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[189], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[183], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[3], m_n[177], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[4], m_n[171], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[5], m_n[163], 1, m_temps);
        csl::step_bit_combine(m_n[196], m_segment_temps_n[0], m_segment_temps_n[1], 6, 1, 1, 2, m_temps);
        csl::step_bit_combine(m_n[196], m_n[196], m_segment_temps_n[2], 6, 2, 2, 3, m_temps);
        csl::step_bit_combine(m_n[196], m_n[196], m_segment_temps_n[3], 6, 3, 3, 4, m_temps);
        csl::step_bit_combine(m_n[196], m_n[196], m_segment_temps_n[4], 6, 4, 4, 5, m_temps);
        csl::step_bit_combine(m_n[196], m_n[196], m_segment_temps_n[5], 6, 5, 5, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[196], 6, m_temps);
        csl::step_bit_extract(m_n[71], m_segment_temps_n[0], 6, true, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[69], 57, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[71], 6, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_ld_exp(m_n[72], m_segment_temps_n[0], m_segment_temps_n[1], false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[72], 5, m_temps);
        csl::step_bit_extract(m_n[86], m_segment_temps_n[0], 1, true, 4, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[72], 4, m_temps);
        csl::step_bit_extract(m_n[85], m_segment_temps_n[0], 1, true, 3, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[72], 3, m_temps);
        csl::step_bit_extract(m_n[84], m_segment_temps_n[0], 1, true, 2, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[72], 2, m_temps);
        csl::step_bit_extract(m_n[83], m_segment_temps_n[0], 1, true, 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[72], 1, m_temps);
        csl::step_bit_extract(m_n[82], m_segment_temps_n[0], 1, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[82], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[83], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[84], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[3], m_n[85], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[4], m_n[86], 1, m_temps);
        csl::step_bit_combine(m_n[87], m_segment_temps_n[0], m_segment_temps_n[1], 5, 1, 1, 2, m_temps);
        csl::step_bit_combine(m_n[87], m_n[87], m_segment_temps_n[2], 5, 2, 2, 3, m_temps);
        csl::step_bit_combine(m_n[87], m_n[87], m_segment_temps_n[3], 5, 3, 3, 4, m_temps);
        csl::step_bit_combine(m_n[87], m_n[87], m_segment_temps_n[4], 5, 4, 4, 0, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[87], 5, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[88], 5, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 5, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 5);
        csl::step_reducing_and(m_n[89], m_segment_temps_n[3], 5, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[89], 1, m_temps);
        csl::step_not_signed(m_n[90], m_segment_temps_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[77], 12, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[71], 6, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_sub(m_n[79], m_segment_temps_n[0], m_segment_temps_n[1], m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[72], 56, m_temps);
        csl::step_bit_extract(m_n[81], m_segment_temps_n[0], 53, true, 3, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[81], 53, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[79], 13, m_temps);
        csl::step_bit_combine(m_w[10], m_segment_temps_n[0], m_segment_temps_n[1], 2, 1, 53, 0, m_temps);
        csl::mask_lower(m_segment_temps_w[3], m_w[10], 66, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[90], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::step_add(m_w[11], m_segment_temps_w[3], m_segment_temps_n[1], m_temps);
        csl::mask_lower(m_segment_temps_w[3], m_w[11], 64, m_temps);
        csl::step_bit_extract(m_n[106], m_segment_temps_w[3], 11, true, 53, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[40], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[106], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[109], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_585[2] = { 2, 3 };
        csl::set(m_n[124], m_segment_temps_n[csl::checked_array_value(mux_lookup_585, m_segment_temps_n[4])]);
        csl::mask_lower(m_segment_temps_w[3], m_w[11], 66, m_temps);
        csl::step_bit_extract(m_n[100], m_segment_temps_w[3], 13, true, 53, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[100], 13, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[99], 13, m_temps);
        csl::step_nxor(m_segment_temps_n[3], m_segment_temps_n[1], m_segment_temps_n[2], 13, m_temps);
        csl::mask(m_segment_temps_n[3], m_segment_temps_n[3], 13);
        csl::step_reducing_and(m_n[101], m_segment_temps_n[3], 13, false, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[15], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[29], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[110], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[3], m_n[101], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[4], m_n[114], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[5], m_n[40], 1, m_temps);
        csl::step_bit_combine(m_n[115], m_segment_temps_n[0], m_segment_temps_n[1], 6, 1, 1, 2, m_temps);
        csl::step_bit_combine(m_n[115], m_n[115], m_segment_temps_n[2], 6, 2, 2, 3, m_temps);
        csl::step_bit_combine(m_n[115], m_n[115], m_segment_temps_n[3], 6, 3, 3, 4, m_temps);
        csl::step_bit_combine(m_n[115], m_n[115], m_segment_temps_n[4], 6, 4, 4, 5, m_temps);
        csl::step_bit_combine(m_n[115], m_n[115], m_segment_temps_n[5], 6, 5, 5, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[115], 6, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[1], 1, m_temps);
        static constexpr int64_t lut_init_data_604[] = {
            0ll, 1ll, 1ll, 1ll, 0ll, 0ll, 0ll, 0ll, 0ll, 1ll, 1ll, 1ll, 1ll, 0ll, 0ll, 0ll, 
            0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 
            0ll, 1ll, 1ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 1ll, 1ll, 0ll, 0ll, 0ll, 0ll, 0ll, 
            0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll
        };
        csl::step_lookup(lut_init_data_604, 64, 0, m_n[116], m_segment_temps_n[0], m_temps);
        csl::step_reduce(m_segment_temps_w[3], m_w[12], 66, m_temps);
        csl::step_bit_extract(m_n[102], m_segment_temps_w[3], 13, true, 53, m_temps);
    }

    void execute_segment_3_fragment_7()
    {
        csl::mask_lower(m_segment_temps_n[0], m_n[0], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[102], 13, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[0], 1, m_temps);
        csl::step_add(m_segment_temps_n[4], m_segment_temps_n[1], m_segment_temps_n[3], m_temps);
        csl::step_sub(m_n[104], m_segment_temps_n[0], m_segment_temps_n[4], m_temps);
        csl::step_nsign_bit(m_n[103], m_n[104], 15);
        csl::step_reduce(m_segment_temps_n[0], m_n[11], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[25], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[110], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[3], m_n[103], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[4], m_n[76], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[5], m_n[40], 1, m_temps);
        csl::step_bit_combine(m_n[111], m_segment_temps_n[0], m_segment_temps_n[1], 6, 1, 1, 2, m_temps);
        csl::step_bit_combine(m_n[111], m_n[111], m_segment_temps_n[2], 6, 2, 2, 3, m_temps);
        csl::step_bit_combine(m_n[111], m_n[111], m_segment_temps_n[3], 6, 3, 3, 4, m_temps);
        csl::step_bit_combine(m_n[111], m_n[111], m_segment_temps_n[4], 6, 4, 4, 5, m_temps);
        csl::step_bit_combine(m_n[111], m_n[111], m_segment_temps_n[5], 6, 5, 5, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[111], 6, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[1], 1, m_temps);
        static constexpr int64_t lut_init_data_623[] = {
            0ll, 0ll, 0ll, 1ll, 0ll, 0ll, 0ll, 1ll, 0ll, 0ll, 0ll, 1ll, 0ll, 0ll, 0ll, 1ll, 
            0ll, 0ll, 0ll, 1ll, 0ll, 0ll, 0ll, 1ll, 0ll, 0ll, 0ll, 1ll, 0ll, 0ll, 0ll, 1ll, 
            0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 1ll, 0ll, 0ll, 0ll, 
            0ll, 0ll, 0ll, 0ll, 1ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 1ll, 0ll, 0ll, 0ll
        };
        csl::step_lookup(lut_init_data_623, 64, 0, m_n[112], m_segment_temps_n[0], m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[112], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[116], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[119], 1, m_temps);
        csl::step_bit_combine(m_n[128], m_segment_temps_n[0], m_segment_temps_n[1], 3, 1, 1, 2, m_temps);
        csl::step_bit_combine(m_n[128], m_n[128], m_segment_temps_n[2], 3, 2, 2, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[128], 3, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[1], 1, m_temps);
        static constexpr int64_t lut_init_data_630[] = {
            1ll, 0ll, 2ll, 0ll, 3ll, 0ll, 0ll, 0ll
        };
        csl::step_lookup(lut_init_data_630, 8, 0, m_n[130], m_segment_temps_n[0], m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[130], 2, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[137], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[124], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_n[136], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[5], m_n[135], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[6], m_segment_temps_n[1], 2, m_temps);
        static constexpr size_t mux_lookup_638[4] = { 2, 3, 4, 5 };
        csl::set(m_n[138], m_segment_temps_n[csl::checked_array_value(mux_lookup_638, m_segment_temps_n[6])]);
        csl::mask_lower(m_segment_temps_n[0], m_w[12], 53, m_temps);
        csl::step_bit_extract(m_n[108], m_segment_temps_n[0], 52, true, 1, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_w[11], 53, m_temps);
        csl::step_bit_extract(m_n[105], m_segment_temps_n[0], 52, true, 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[40], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[105], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[108], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_650[2] = { 2, 3 };
        csl::set(m_n[123], m_segment_temps_n[csl::checked_array_value(mux_lookup_650, m_segment_temps_n[4])]);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[130], 2, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[133], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[123], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_n[132], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[5], m_n[131], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[6], m_segment_temps_n[1], 2, m_temps);
        static constexpr size_t mux_lookup_659[4] = { 2, 3, 4, 5 };
        csl::set(m_n[134], m_segment_temps_n[csl::checked_array_value(mux_lookup_659, m_segment_temps_n[6])]);
        csl::step_reduce(m_segment_temps_n[0], m_n[134], 52, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[138], 11, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[144], 1, m_temps);
        csl::step_bit_combine(m_w[13], m_segment_temps_n[0], m_segment_temps_n[1], 3, 1, 52, 63, m_temps);
        csl::step_bit_combine(m_w[13], m_w[13], m_segment_temps_n[2], 3, 2, 63, 0, m_temps);
        csl::step_reduce(m_segment_temps_w[3], m_w[13], 64, m_temps);
        csl::step_bit_extract(m_w[0], m_segment_temps_w[3], 64, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_w[3], m_w[0], 64, m_temps);
        m_w[PORT_OUT_PRIMWIREOUT3] = m_segment_temps_w[3];
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[40], 1, m_temps);
        csl::step_not_signed(m_n[153], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[11], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[25], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[153], 1, m_temps);
        csl::step_and(m_n[154], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[154], m_n[154], 1);
        csl::step_and(m_n[154], m_n[154], m_segment_temps_n[3], 1, m_temps);
        csl::mask(m_n[154], m_n[154], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[154], 1, m_temps);
        csl::step_not_signed(m_n[155], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[110], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[76], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[153], 1, m_temps);
        csl::step_and(m_n[156], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[156], m_n[156], 1);
        csl::step_and(m_n[156], m_n[156], m_segment_temps_n[3], 1, m_temps);
        csl::mask(m_n[156], m_n[156], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[156], 1, m_temps);
        csl::step_not_signed(m_n[157], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[4], 1, m_temps);
        csl::step_not_signed(m_n[5], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
    }

    void execute_segment_3_fragment_8()
    {
        csl::mask_lower(m_segment_temps_n[1], m_n[107], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[5], 1, m_temps);
        csl::step_xor(m_n[158], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[158], m_n[158], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[40], 1, m_temps);
        csl::step_not_signed(m_n[120], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[15], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[29], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[120], 1, m_temps);
        csl::step_and(m_n[121], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[121], m_n[121], 1);
        csl::step_and(m_n[121], m_n[121], m_segment_temps_n[3], 1, m_temps);
        csl::mask(m_n[121], m_n[121], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[121], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[114], 1, m_temps);
        csl::step_or(m_n[122], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[122], m_n[122], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[122], 1, m_temps);
        csl::step_not_signed(m_n[159], m_segment_temps_n[1], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[159], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[158], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[157], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_n[155], 1, m_temps);
        csl::step_and(m_n[160], m_segment_temps_n[1], m_segment_temps_n[2], 1, m_temps);
        csl::mask(m_n[160], m_n[160], 1);
        csl::step_and(m_n[160], m_n[160], m_segment_temps_n[3], 1, m_temps);
        csl::mask(m_n[160], m_n[160], 1);
        csl::step_and(m_n[160], m_n[160], m_segment_temps_n[4], 1, m_temps);
        csl::mask(m_n[160], m_n[160], 1);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[40], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[109], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[106], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_722[2] = { 2, 3 };
        csl::set(m_n[126], m_segment_temps_n[csl::checked_array_value(mux_lookup_722, m_segment_temps_n[4])]);
        csl::mask_lower(m_segment_temps_n[0], m_n[115], 6, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[1], 1, m_temps);
        static constexpr int64_t lut_init_data_726[] = {
            0ll, 1ll, 1ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 1ll, 1ll, 0ll, 0ll, 0ll, 0ll, 0ll, 
            0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 
            0ll, 1ll, 1ll, 1ll, 0ll, 0ll, 0ll, 0ll, 0ll, 1ll, 1ll, 1ll, 1ll, 0ll, 0ll, 0ll, 
            0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll
        };
        csl::step_lookup(lut_init_data_726, 64, 0, m_n[117], m_segment_temps_n[0], m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[111], 6, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[1], 1, m_temps);
        static constexpr int64_t lut_init_data_729[] = {
            0ll, 0ll, 0ll, 1ll, 0ll, 0ll, 0ll, 1ll, 0ll, 0ll, 0ll, 1ll, 1ll, 0ll, 0ll, 1ll, 
            0ll, 0ll, 0ll, 1ll, 1ll, 0ll, 0ll, 1ll, 0ll, 0ll, 0ll, 1ll, 1ll, 0ll, 0ll, 1ll, 
            0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 
            0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll, 0ll
        };
        csl::step_lookup(lut_init_data_729, 64, 0, m_n[113], m_segment_temps_n[0], m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[113], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[117], 1, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[122], 1, m_temps);
        csl::step_bit_combine(m_n[127], m_segment_temps_n[0], m_segment_temps_n[1], 3, 1, 1, 2, m_temps);
        csl::step_bit_combine(m_n[127], m_n[127], m_segment_temps_n[2], 3, 2, 2, 0, m_temps);
        csl::mask_lower(m_segment_temps_n[0], m_n[127], 3, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[1], 1, m_temps);
        static constexpr int64_t lut_init_data_736[] = {
            1ll, 0ll, 2ll, 0ll, 3ll, 0ll, 0ll, 0ll
        };
        csl::step_lookup(lut_init_data_736, 8, 0, m_n[129], m_segment_temps_n[0], m_temps);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[129], 2, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[151], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[126], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_n[150], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[5], m_n[149], 11, m_temps);
        csl::mask_lower(m_segment_temps_n[6], m_segment_temps_n[1], 2, m_temps);
        static constexpr size_t mux_lookup_744[4] = { 2, 3, 4, 5 };
        csl::set(m_n[152], m_segment_temps_n[csl::checked_array_value(mux_lookup_744, m_segment_temps_n[6])]);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[40], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[108], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[105], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_segment_temps_n[1], 1, m_temps);
        static constexpr size_t mux_lookup_752[2] = { 2, 3 };
        csl::set(m_n[125], m_segment_temps_n[csl::checked_array_value(mux_lookup_752, m_segment_temps_n[4])]);
        csl::step_reduce(m_segment_temps_n[0], m_n[1], 1, m_temps);
        csl::mask_lower(m_segment_temps_n[1], m_n[129], 2, m_temps);
        csl::mask_lower(m_segment_temps_n[2], m_n[147], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[3], m_n[125], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[4], m_n[146], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[5], m_n[145], 52, m_temps);
        csl::mask_lower(m_segment_temps_n[6], m_segment_temps_n[1], 2, m_temps);
        static constexpr size_t mux_lookup_761[4] = { 2, 3, 4, 5 };
        csl::set(m_n[148], m_segment_temps_n[csl::checked_array_value(mux_lookup_761, m_segment_temps_n[6])]);
        csl::step_reduce(m_segment_temps_n[0], m_n[148], 52, m_temps);
        csl::step_reduce(m_segment_temps_n[1], m_n[152], 11, m_temps);
        csl::step_reduce(m_segment_temps_n[2], m_n[160], 1, m_temps);
        csl::step_bit_combine(m_w[14], m_segment_temps_n[0], m_segment_temps_n[1], 3, 1, 52, 63, m_temps);
        csl::step_bit_combine(m_w[14], m_w[14], m_segment_temps_n[2], 3, 2, 63, 0, m_temps);
        csl::step_reduce(m_segment_temps_w[3], m_w[14], 64, m_temps);
        csl::step_bit_extract(m_w[1], m_segment_temps_w[3], 64, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_w[3], m_w[1], 64, m_temps);
        m_w[PORT_OUT_PRIMWIREAUX2] = m_segment_temps_w[3];
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
        execute_segment_3_fragment_7();
        execute_segment_3_fragment_8();
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
        if ((m_segment_cycle[3] == m_update_cycle) && (m_io_cycle[0] > m_update_cycle) && (m_io_cycle[1] > m_update_cycle))
        {
            execute_segment_3();
            ++m_segment_cycle[3];
        }
        const bool all_io_ready = (m_io_cycle[0] > m_update_cycle) && (m_io_cycle[1] > m_update_cycle) && (m_io_cycle[2] > m_update_cycle) && (m_io_cycle[3] > m_update_cycle);
        if (all_io_ready && (m_segment_cycle[0] > m_update_cycle) && (m_segment_cycle[1] > m_update_cycle) && (m_segment_cycle[2] > m_update_cycle) && (m_segment_cycle[3] > m_update_cycle))
        {
            execute_segment_update();
            ++m_update_cycle;
        }
    }

    static constexpr size_t PORT_IN_00 = 2;
    static constexpr size_t PORT_IN_11 = 3;
    static constexpr size_t PORT_OUT_PRIMWIREOUT3 = 18;
    static constexpr size_t PORT_OUT_PRIMWIREAUX2 = 19;

    int64_t m_io_cycle[4];
    int64_t m_segment_cycle[4];
    int64_t m_update_cycle;

    int64_t m_segment_temps_n[7] = { 0 };
    csl::mp_int m_segment_temps_w[4];
    csl::mp_int_temps m_temps;
    int64_t m_n[241];
    csl::mp_int m_w[20];
};

#endif // SOFTWARE_MODEL_FLT_FFT_EXAMPLE_DUT_ADDSUBFUSEDBLOCK_TYPESFLOATIEEE_52_11_4_CORRECTROUNDING_38560000X0AO30CD06CJ6OK0DPZC_H_