-- ------------------------------------------------------------------------- 
-- High Level Design Compiler for Altera(R) FPGAs Version 25.1 (Release Build #6a12354d2f)
-- Quartus Prime development tool and MATLAB/Simulink Interface
-- 
-- Legal Notice: Copyright 2025 Altera Corporation.  All rights reserved.
-- Your use of Altera Corporation's  design tools,  logic functions and other
-- software and  tools, and  its AMPP partner logic functions, and any output
-- files any  of the  foregoing (including  device programming  or simulation
-- files), and  any associated  documentation  or  information  are expressly
-- subject to the terms and  conditions  of the  Altera FPGA Software License
-- Agreement, Altera MegaCore Function License Agreement, or other applicable
-- license agreement,  including,  without limitation,  that  your use is for
-- the  sole  purpose of  programming  logic devices  manufactured by  Altera
-- and  sold by Altera  or its authorized  distributors. Please refer  to the
-- applicable agreement for further details.
-- ---------------------------------------------------------------------------

-- VHDL created from fft_example_DUT
-- VHDL created on Tue Jul  8 08:39:19 2025


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.NUMERIC_STD.all;
use work.dspba_sim_library_package.all;
entity fft_example_DUT_atb is
end;

architecture normal of fft_example_DUT_atb is

component fft_example_DUT is
    port (
        v_in_s : in std_logic_vector(0 downto 0);  -- ufix1
        channel_in_s : in std_logic_vector(7 downto 0);  -- ufix8
        data_in_0re : in std_logic_vector(63 downto 0);  -- float64_m52
        data_in_0im : in std_logic_vector(63 downto 0);  -- float64_m52
        data_in_1re : in std_logic_vector(63 downto 0);  -- float64_m52
        data_in_1im : in std_logic_vector(63 downto 0);  -- float64_m52
        data_in_2re : in std_logic_vector(63 downto 0);  -- float64_m52
        data_in_2im : in std_logic_vector(63 downto 0);  -- float64_m52
        data_in_3re : in std_logic_vector(63 downto 0);  -- float64_m52
        data_in_3im : in std_logic_vector(63 downto 0);  -- float64_m52
        v_out_s : out std_logic_vector(0 downto 0);  -- ufix1
        data_out_0re : out std_logic_vector(63 downto 0);  -- float64_m52
        data_out_0im : out std_logic_vector(63 downto 0);  -- float64_m52
        data_out_1re : out std_logic_vector(63 downto 0);  -- float64_m52
        data_out_1im : out std_logic_vector(63 downto 0);  -- float64_m52
        data_out_2re : out std_logic_vector(63 downto 0);  -- float64_m52
        data_out_2im : out std_logic_vector(63 downto 0);  -- float64_m52
        data_out_3re : out std_logic_vector(63 downto 0);  -- float64_m52
        data_out_3im : out std_logic_vector(63 downto 0);  -- float64_m52
        clk : in std_logic;
        areset : in std_logic
    );
end component;

component fft_example_DUT_stm is
    port (
        v_in_s_stm : out std_logic_vector(0 downto 0);
        channel_in_s_stm : out std_logic_vector(7 downto 0);
        data_in_0re_stm : out std_logic_vector(63 downto 0);
        data_in_0im_stm : out std_logic_vector(63 downto 0);
        data_in_1re_stm : out std_logic_vector(63 downto 0);
        data_in_1im_stm : out std_logic_vector(63 downto 0);
        data_in_2re_stm : out std_logic_vector(63 downto 0);
        data_in_2im_stm : out std_logic_vector(63 downto 0);
        data_in_3re_stm : out std_logic_vector(63 downto 0);
        data_in_3im_stm : out std_logic_vector(63 downto 0);
        v_out_s_stm : out std_logic_vector(0 downto 0);
        data_out_0re_stm : out std_logic_vector(63 downto 0);
        data_out_0im_stm : out std_logic_vector(63 downto 0);
        data_out_1re_stm : out std_logic_vector(63 downto 0);
        data_out_1im_stm : out std_logic_vector(63 downto 0);
        data_out_2re_stm : out std_logic_vector(63 downto 0);
        data_out_2im_stm : out std_logic_vector(63 downto 0);
        data_out_3re_stm : out std_logic_vector(63 downto 0);
        data_out_3im_stm : out std_logic_vector(63 downto 0);
        clk : out std_logic;
        areset : out std_logic
    );
end component;

signal v_in_s_stm : STD_LOGIC_VECTOR (0 downto 0);
signal channel_in_s_stm : STD_LOGIC_VECTOR (7 downto 0);
signal data_in_0re_stm : STD_LOGIC_VECTOR (63 downto 0);
signal data_in_0im_stm : STD_LOGIC_VECTOR (63 downto 0);
signal data_in_1re_stm : STD_LOGIC_VECTOR (63 downto 0);
signal data_in_1im_stm : STD_LOGIC_VECTOR (63 downto 0);
signal data_in_2re_stm : STD_LOGIC_VECTOR (63 downto 0);
signal data_in_2im_stm : STD_LOGIC_VECTOR (63 downto 0);
signal data_in_3re_stm : STD_LOGIC_VECTOR (63 downto 0);
signal data_in_3im_stm : STD_LOGIC_VECTOR (63 downto 0);
signal v_out_s_stm : STD_LOGIC_VECTOR (0 downto 0);
signal data_out_0re_stm : STD_LOGIC_VECTOR (63 downto 0);
signal data_out_0im_stm : STD_LOGIC_VECTOR (63 downto 0);
signal data_out_1re_stm : STD_LOGIC_VECTOR (63 downto 0);
signal data_out_1im_stm : STD_LOGIC_VECTOR (63 downto 0);
signal data_out_2re_stm : STD_LOGIC_VECTOR (63 downto 0);
signal data_out_2im_stm : STD_LOGIC_VECTOR (63 downto 0);
signal data_out_3re_stm : STD_LOGIC_VECTOR (63 downto 0);
signal data_out_3im_stm : STD_LOGIC_VECTOR (63 downto 0);
signal v_in_s_dut : STD_LOGIC_VECTOR (0 downto 0);
signal channel_in_s_dut : STD_LOGIC_VECTOR (7 downto 0);
signal data_in_0re_dut : STD_LOGIC_VECTOR (63 downto 0);
signal data_in_0im_dut : STD_LOGIC_VECTOR (63 downto 0);
signal data_in_1re_dut : STD_LOGIC_VECTOR (63 downto 0);
signal data_in_1im_dut : STD_LOGIC_VECTOR (63 downto 0);
signal data_in_2re_dut : STD_LOGIC_VECTOR (63 downto 0);
signal data_in_2im_dut : STD_LOGIC_VECTOR (63 downto 0);
signal data_in_3re_dut : STD_LOGIC_VECTOR (63 downto 0);
signal data_in_3im_dut : STD_LOGIC_VECTOR (63 downto 0);
signal v_out_s_dut : STD_LOGIC_VECTOR (0 downto 0);
signal data_out_0re_dut : STD_LOGIC_VECTOR (63 downto 0);
signal data_out_0im_dut : STD_LOGIC_VECTOR (63 downto 0);
signal data_out_1re_dut : STD_LOGIC_VECTOR (63 downto 0);
signal data_out_1im_dut : STD_LOGIC_VECTOR (63 downto 0);
signal data_out_2re_dut : STD_LOGIC_VECTOR (63 downto 0);
signal data_out_2im_dut : STD_LOGIC_VECTOR (63 downto 0);
signal data_out_3re_dut : STD_LOGIC_VECTOR (63 downto 0);
signal data_out_3im_dut : STD_LOGIC_VECTOR (63 downto 0);
        signal clk : std_logic;
        signal areset : std_logic;

begin

-- Channelized data in real output
checkChannelIn_vunroll_cunroll_x : process (clk, areset, data_in_0re_dut, data_in_0re_stm, data_in_0im_dut, data_in_0im_stm, data_in_1re_dut, data_in_1re_stm, data_in_1im_dut, data_in_1im_stm, data_in_2re_dut, data_in_2re_stm, data_in_2im_dut, data_in_2im_stm, data_in_3re_dut, data_in_3re_stm, data_in_3im_dut, data_in_3im_stm)
variable data_in_0re_real : REAL := 0.0;
variable data_in_0re_stm_real : REAL := 0.0;
variable data_in_0im_real : REAL := 0.0;
variable data_in_0im_stm_real : REAL := 0.0;
variable data_in_1re_real : REAL := 0.0;
variable data_in_1re_stm_real : REAL := 0.0;
variable data_in_1im_real : REAL := 0.0;
variable data_in_1im_stm_real : REAL := 0.0;
variable data_in_2re_real : REAL := 0.0;
variable data_in_2re_stm_real : REAL := 0.0;
variable data_in_2im_real : REAL := 0.0;
variable data_in_2im_stm_real : REAL := 0.0;
variable data_in_3re_real : REAL := 0.0;
variable data_in_3re_stm_real : REAL := 0.0;
variable data_in_3im_real : REAL := 0.0;
variable data_in_3im_stm_real : REAL := 0.0;
begin
 data_in_0re_real := vIEEE_2_real(data_in_0re_dut, 11, 52, false);
 data_in_0re_stm_real := vIEEE_2_real(data_in_0re_stm, 11, 52, false);
 data_in_0im_real := vIEEE_2_real(data_in_0im_dut, 11, 52, false);
 data_in_0im_stm_real := vIEEE_2_real(data_in_0im_stm, 11, 52, false);
 data_in_1re_real := vIEEE_2_real(data_in_1re_dut, 11, 52, false);
 data_in_1re_stm_real := vIEEE_2_real(data_in_1re_stm, 11, 52, false);
 data_in_1im_real := vIEEE_2_real(data_in_1im_dut, 11, 52, false);
 data_in_1im_stm_real := vIEEE_2_real(data_in_1im_stm, 11, 52, false);
 data_in_2re_real := vIEEE_2_real(data_in_2re_dut, 11, 52, false);
 data_in_2re_stm_real := vIEEE_2_real(data_in_2re_stm, 11, 52, false);
 data_in_2im_real := vIEEE_2_real(data_in_2im_dut, 11, 52, false);
 data_in_2im_stm_real := vIEEE_2_real(data_in_2im_stm, 11, 52, false);
 data_in_3re_real := vIEEE_2_real(data_in_3re_dut, 11, 52, false);
 data_in_3re_stm_real := vIEEE_2_real(data_in_3re_stm, 11, 52, false);
 data_in_3im_real := vIEEE_2_real(data_in_3im_dut, 11, 52, false);
 data_in_3im_stm_real := vIEEE_2_real(data_in_3im_stm, 11, 52, false);
END PROCESS;


-- Channelized data out check
checkChannelOut_vunroll_cunroll_x : process (clk, areset, data_out_0re_dut, data_out_0re_stm, data_out_0im_dut, data_out_0im_stm, data_out_1re_dut, data_out_1re_stm, data_out_1im_dut, data_out_1im_stm, data_out_2re_dut, data_out_2re_stm, data_out_2im_dut, data_out_2im_stm, data_out_3re_dut, data_out_3re_stm, data_out_3im_dut, data_out_3im_stm)
variable mismatch_v_out_s : BOOLEAN := FALSE;
variable mismatch_data_out_0re : BOOLEAN := FALSE;
variable data_out_0re_real : REAL := 0.0;
variable data_out_0re_stm_real : REAL := 0.0;
variable mismatch_data_out_0im : BOOLEAN := FALSE;
variable data_out_0im_real : REAL := 0.0;
variable data_out_0im_stm_real : REAL := 0.0;
variable mismatch_data_out_1re : BOOLEAN := FALSE;
variable data_out_1re_real : REAL := 0.0;
variable data_out_1re_stm_real : REAL := 0.0;
variable mismatch_data_out_1im : BOOLEAN := FALSE;
variable data_out_1im_real : REAL := 0.0;
variable data_out_1im_stm_real : REAL := 0.0;
variable mismatch_data_out_2re : BOOLEAN := FALSE;
variable data_out_2re_real : REAL := 0.0;
variable data_out_2re_stm_real : REAL := 0.0;
variable mismatch_data_out_2im : BOOLEAN := FALSE;
variable data_out_2im_real : REAL := 0.0;
variable data_out_2im_stm_real : REAL := 0.0;
variable mismatch_data_out_3re : BOOLEAN := FALSE;
variable data_out_3re_real : REAL := 0.0;
variable data_out_3re_stm_real : REAL := 0.0;
variable mismatch_data_out_3im : BOOLEAN := FALSE;
variable data_out_3im_real : REAL := 0.0;
variable data_out_3im_stm_real : REAL := 0.0;
variable ok : BOOLEAN := TRUE;
begin
 data_out_0re_real := vIEEE_2_real(data_out_0re_dut, 11, 52, false);
 data_out_0re_stm_real := vIEEE_2_real(data_out_0re_stm, 11, 52, false);
 data_out_0im_real := vIEEE_2_real(data_out_0im_dut, 11, 52, false);
 data_out_0im_stm_real := vIEEE_2_real(data_out_0im_stm, 11, 52, false);
 data_out_1re_real := vIEEE_2_real(data_out_1re_dut, 11, 52, false);
 data_out_1re_stm_real := vIEEE_2_real(data_out_1re_stm, 11, 52, false);
 data_out_1im_real := vIEEE_2_real(data_out_1im_dut, 11, 52, false);
 data_out_1im_stm_real := vIEEE_2_real(data_out_1im_stm, 11, 52, false);
 data_out_2re_real := vIEEE_2_real(data_out_2re_dut, 11, 52, false);
 data_out_2re_stm_real := vIEEE_2_real(data_out_2re_stm, 11, 52, false);
 data_out_2im_real := vIEEE_2_real(data_out_2im_dut, 11, 52, false);
 data_out_2im_stm_real := vIEEE_2_real(data_out_2im_stm, 11, 52, false);
 data_out_3re_real := vIEEE_2_real(data_out_3re_dut, 11, 52, false);
 data_out_3re_stm_real := vIEEE_2_real(data_out_3re_stm, 11, 52, false);
 data_out_3im_real := vIEEE_2_real(data_out_3im_dut, 11, 52, false);
 data_out_3im_stm_real := vIEEE_2_real(data_out_3im_stm, 11, 52, false);
    IF ((areset = '1')) THEN
        -- do nothing during reset
    ELSIF (clk'EVENT AND clk = '0') THEN -- falling clock edge to avoid transitions
        ok := TRUE;
        mismatch_v_out_s := FALSE;
        mismatch_data_out_0re := FALSE;
        mismatch_data_out_0im := FALSE;
        mismatch_data_out_1re := FALSE;
        mismatch_data_out_1im := FALSE;
        mismatch_data_out_2re := FALSE;
        mismatch_data_out_2im := FALSE;
        mismatch_data_out_3re := FALSE;
        mismatch_data_out_3im := FALSE;
        IF ( (v_out_s_dut /= v_out_s_stm)) THEN
            mismatch_v_out_s := TRUE;
            report "mismatch in v_out_s signal" severity Failure;
        END IF;
        IF ((v_out_s_dut = "1")) THEN
            IF ( not vIEEEisEqual(data_out_0re_dut, data_out_0re_stm, 11, 52, 0.000000e+00, 0.000000e+00)) THEN
                mismatch_data_out_0re := TRUE;
                report "mismatch in data_out_0re signal" severity Warning;
            END IF;
            IF ( not vIEEEisEqual(data_out_0im_dut, data_out_0im_stm, 11, 52, 0.000000e+00, 0.000000e+00)) THEN
                mismatch_data_out_0im := TRUE;
                report "mismatch in data_out_0im signal" severity Warning;
            END IF;
            IF ( not vIEEEisEqual(data_out_1re_dut, data_out_1re_stm, 11, 52, 0.000000e+00, 0.000000e+00)) THEN
                mismatch_data_out_1re := TRUE;
                report "mismatch in data_out_1re signal" severity Warning;
            END IF;
            IF ( not vIEEEisEqual(data_out_1im_dut, data_out_1im_stm, 11, 52, 0.000000e+00, 0.000000e+00)) THEN
                mismatch_data_out_1im := TRUE;
                report "mismatch in data_out_1im signal" severity Warning;
            END IF;
            IF ( not vIEEEisEqual(data_out_2re_dut, data_out_2re_stm, 11, 52, 0.000000e+00, 0.000000e+00)) THEN
                mismatch_data_out_2re := TRUE;
                report "mismatch in data_out_2re signal" severity Warning;
            END IF;
            IF ( not vIEEEisEqual(data_out_2im_dut, data_out_2im_stm, 11, 52, 0.000000e+00, 0.000000e+00)) THEN
                mismatch_data_out_2im := TRUE;
                report "mismatch in data_out_2im signal" severity Warning;
            END IF;
            IF ( not vIEEEisEqual(data_out_3re_dut, data_out_3re_stm, 11, 52, 0.000000e+00, 0.000000e+00)) THEN
                mismatch_data_out_3re := TRUE;
                report "mismatch in data_out_3re signal" severity Warning;
            END IF;
            IF ( not vIEEEisEqual(data_out_3im_dut, data_out_3im_stm, 11, 52, 0.000000e+00, 0.000000e+00)) THEN
                mismatch_data_out_3im := TRUE;
                report "mismatch in data_out_3im signal" severity Warning;
            END IF;
        END IF;
        IF (mismatch_v_out_s = TRUE or mismatch_data_out_0re = TRUE or mismatch_data_out_0im = TRUE or mismatch_data_out_1re = TRUE or mismatch_data_out_1im = TRUE or mismatch_data_out_2re = TRUE or mismatch_data_out_2im = TRUE or mismatch_data_out_3re = TRUE or mismatch_data_out_3im = TRUE) THEN
            ok := FALSE;
        END IF;
        IF (ok = FALSE) THEN
            report "Mismatch detected" severity Failure;
        END IF;
    END IF;
END PROCESS;


dut : fft_example_DUT port map (
    v_in_s_stm,
    channel_in_s_stm,
    data_in_0re_stm,
    data_in_0im_stm,
    data_in_1re_stm,
    data_in_1im_stm,
    data_in_2re_stm,
    data_in_2im_stm,
    data_in_3re_stm,
    data_in_3im_stm,
    v_out_s_dut,
    data_out_0re_dut,
    data_out_0im_dut,
    data_out_1re_dut,
    data_out_1im_dut,
    data_out_2re_dut,
    data_out_2im_dut,
    data_out_3re_dut,
    data_out_3im_dut,
        clk,
        areset
);

sim : fft_example_DUT_stm port map (
    v_in_s_stm,
    channel_in_s_stm,
    data_in_0re_stm,
    data_in_0im_stm,
    data_in_1re_stm,
    data_in_1im_stm,
    data_in_2re_stm,
    data_in_2im_stm,
    data_in_3re_stm,
    data_in_3im_stm,
    v_out_s_stm,
    data_out_0re_stm,
    data_out_0im_stm,
    data_out_1re_stm,
    data_out_1im_stm,
    data_out_2re_stm,
    data_out_2im_stm,
    data_out_3re_stm,
    data_out_3im_stm,
        clk,
        areset
);

end normal;
