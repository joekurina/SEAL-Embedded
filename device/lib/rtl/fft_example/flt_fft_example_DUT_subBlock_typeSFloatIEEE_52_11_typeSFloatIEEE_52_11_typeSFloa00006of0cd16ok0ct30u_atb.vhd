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

-- VHDL created from flt_fft_example_DUT_subBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_addFpArchSoleDatapath_correctRounding_3m5cq5c35in4uia1u6737j47pg9pig2jg8vcexdqyfw8i063061663c61i64oc1764761di4p61v64vi1e64e61kc5363l60uq5ux0ao30cd06cj0of0cdj6of0cd16ok0ct30u
-- VHDL created on Tue Jul  8 08:39:19 2025


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.NUMERIC_STD.all;
use work.dspba_sim_library_package.all;
entity flt_fft_example_DUT_subBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa00000cdj6of0cd16ok0ct30u_atb is
end;

architecture normal of flt_fft_example_DUT_subBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa00000cdj6of0cd16ok0ct30u_atb is

component flt_fft_example_DUT_subBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa00000cdj6of0cd16ok0ct30u is
    port (
        in_0 : in std_logic_vector(63 downto 0);  -- float64_m52
        in_1 : in std_logic_vector(63 downto 0);  -- float64_m52
        out_primWireOut : out std_logic_vector(63 downto 0);  -- float64_m52
        clk : in std_logic;
        areset : in std_logic
    );
end component;

component flt_fft_example_DUT_subBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa00000cdj6of0cd16ok0ct30u_stm is
    port (
        in_0_stm : out std_logic_vector(63 downto 0);
        in_1_stm : out std_logic_vector(63 downto 0);
        out_primWireOut_stm : out std_logic_vector(63 downto 0);
        clk : out std_logic;
        areset : out std_logic
    );
end component;

signal in_0_stm : STD_LOGIC_VECTOR (63 downto 0);
signal in_1_stm : STD_LOGIC_VECTOR (63 downto 0);
signal out_primWireOut_stm : STD_LOGIC_VECTOR (63 downto 0);
signal in_0_dut : STD_LOGIC_VECTOR (63 downto 0);
signal in_1_dut : STD_LOGIC_VECTOR (63 downto 0);
signal out_primWireOut_dut : STD_LOGIC_VECTOR (63 downto 0);
        signal clk : std_logic;
        signal areset : std_logic;

begin

-- General Purpose data in real output
checkin_0 : process (clk, areset, in_0_dut, in_0_stm)
variable in_0_real : REAL := 0.0;
variable in_0_stm_real : REAL := 0.0;
begin
 in_0_real := vIEEE_2_real(in_0_dut, 11, 52, false);
 in_0_stm_real := vIEEE_2_real(in_0_stm, 11, 52, false);
END PROCESS;


-- General Purpose data in real output
checkin_1 : process (clk, areset, in_1_dut, in_1_stm)
variable in_1_real : REAL := 0.0;
variable in_1_stm_real : REAL := 0.0;
begin
 in_1_real := vIEEE_2_real(in_1_dut, 11, 52, false);
 in_1_stm_real := vIEEE_2_real(in_1_stm, 11, 52, false);
END PROCESS;


dut : flt_fft_example_DUT_subBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa00000cdj6of0cd16ok0ct30u port map (
    in_0_stm,
    in_1_stm,
    out_primWireOut_dut,
        clk,
        areset
);

sim : flt_fft_example_DUT_subBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa00000cdj6of0cd16ok0ct30u_stm port map (
    in_0_stm,
    in_1_stm,
    out_primWireOut_stm,
        clk,
        areset
);

end normal;
