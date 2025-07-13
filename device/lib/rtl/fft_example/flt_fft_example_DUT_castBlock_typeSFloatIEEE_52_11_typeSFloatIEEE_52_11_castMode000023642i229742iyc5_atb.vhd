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

-- VHDL created from flt_fft_example_DUT_castBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_castModeConvert_331cq5c30ot5uob117798ja8pmapoe8010i10710k10910y20d70v70e90y70h70z80kd054cz5i1s0i226123642i229742iyc5
-- VHDL created on Tue Jul  8 08:39:19 2025


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.NUMERIC_STD.all;
use work.dspba_sim_library_package.all;
entity flt_fft_example_DUT_castBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_castMode0000226123642i229742iyc5_atb is
end;

architecture normal of flt_fft_example_DUT_castBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_castMode0000226123642i229742iyc5_atb is

component flt_fft_example_DUT_castBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_castMode0000226123642i229742iyc5 is
    port (
        in_0 : in std_logic_vector(63 downto 0);  -- float64_m52
        out_primWireOut : out std_logic_vector(63 downto 0);  -- float64_m52
        clk : in std_logic;
        areset : in std_logic
    );
end component;

component flt_fft_example_DUT_castBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_castMode0000226123642i229742iyc5_stm is
    port (
        in_0_stm : out std_logic_vector(63 downto 0);
        out_primWireOut_stm : out std_logic_vector(63 downto 0);
        clk : out std_logic;
        areset : out std_logic
    );
end component;

signal in_0_stm : STD_LOGIC_VECTOR (63 downto 0);
signal out_primWireOut_stm : STD_LOGIC_VECTOR (63 downto 0);
signal in_0_dut : STD_LOGIC_VECTOR (63 downto 0);
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


dut : flt_fft_example_DUT_castBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_castMode0000226123642i229742iyc5 port map (
    in_0_stm,
    out_primWireOut_dut,
        clk,
        areset
);

sim : flt_fft_example_DUT_castBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_castMode0000226123642i229742iyc5_stm port map (
    in_0_stm,
    out_primWireOut_stm,
        clk,
        areset
);

end normal;
