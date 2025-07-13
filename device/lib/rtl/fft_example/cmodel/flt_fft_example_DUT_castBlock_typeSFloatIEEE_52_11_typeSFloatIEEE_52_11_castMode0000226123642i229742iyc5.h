// ------------------------------------------------------------------------- 
// High Level Design Compiler for Altera(R) FPGAs Version 25.1 (Release Build #6a12354d2f)
// Software model created on 2025-07-08 08:39:19
// Generation mode: Bit Accurate
// ------------------------------------------------------------------------- 
#pragma once

#ifndef SOFTWARE_MODEL_FLT_FFT_EXAMPLE_DUT_CASTBLOCK_TYPESFLOATIEEE_52_11_TYPESFLOATIEEE_52_11_CASTMODE0000226123642I229742IYC5_H_
#define SOFTWARE_MODEL_FLT_FFT_EXAMPLE_DUT_CASTBLOCK_TYPESFLOATIEEE_52_11_TYPESFLOATIEEE_52_11_CASTMODE0000226123642I229742IYC5_H_

#include "support/csl.h"
#ifdef WRITE_STM_FILES
#include "support/csl_io.h"
#endif

class flt_fft_example_DUT_castBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_castMode0000226123642i229742iyc5
{
public:
    // IO struct for "in_0"
    struct io_struct_in_0
    {
        double port_in_0 = 0;
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
        bool needs_to_execute = (m_io_cycle[1] == m_update_cycle);
        if (needs_to_execute)
        {
            m_io_cycle[1]++;
            execute();
        }

        mask_lower(output.port_out_primwireout, m_w[PORT_OUT_PRIMWIREOUT1], 64, m_temps);
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

    // Resets internal simulation state to default values and opens stimulus files.
    void reset()
    {
        static constexpr int64_t native_type_reset_values[] = { 0, 0 };
        std::memcpy(m_n, native_type_reset_values, 2 * sizeof(int64_t));
        static constexpr uint64_t wide_type_reset_values[] = { 0, 0, 0 };
        static constexpr csl::mp_int_info wide_type_reset_infos[] = { { 0, 1, 0 }, { 1, 1, 0 }, { 2, 1, 0 } };
        for (size_t i = 0; i < 3; ++i)
        {
            csl::fill_mpz_data(m_w[i], wide_type_reset_values, wide_type_reset_infos, i);
        }
        csl::fill_n(m_io_cycle, 2, -1);
        csl::fill_n(m_segment_cycle, 2, -1);
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
            m_io_cycle[1] = m_update_cycle + 1;
            execute();
        }
    }

private:
    // Segments are chunks of execution that depend on unique sets of inputs
    // These functions are invoked automatically as inputs are provided to the model.
    void execute_segment_0()
    {
    }

    void execute_segment_1_fragment_0()
    {
        csl::step_reduce(m_segment_temps_w[0], m_w[PORT_IN_00], 64, m_temps);
        csl::step_bit_extract(m_w[0], m_segment_temps_w[0], 64, true, 0, m_temps);
        csl::step_reduce(m_segment_temps_w[0], m_w[0], 64, m_temps);
        m_w[PORT_OUT_PRIMWIREOUT1] = m_segment_temps_w[0];
    }

    void execute_segment_1()
    {
        execute_segment_1_fragment_0();
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
        const bool all_io_ready = (m_io_cycle[0] > m_update_cycle) && (m_io_cycle[1] > m_update_cycle);
        if (all_io_ready && (m_segment_cycle[0] > m_update_cycle) && (m_segment_cycle[1] > m_update_cycle))
        {
            execute_segment_update();
            ++m_update_cycle;
        }
    }

    static constexpr size_t PORT_IN_00 = 1;
    static constexpr size_t PORT_OUT_PRIMWIREOUT1 = 2;

    int64_t m_io_cycle[2];
    int64_t m_segment_cycle[2];
    int64_t m_update_cycle;

    csl::mp_int m_segment_temps_w[1];
    csl::mp_int_temps m_temps;
    int64_t m_n[2];
    csl::mp_int m_w[3];
};

#endif // SOFTWARE_MODEL_FLT_FFT_EXAMPLE_DUT_CASTBLOCK_TYPESFLOATIEEE_52_11_TYPESFLOATIEEE_52_11_CASTMODE0000226123642I229742IYC5_H_