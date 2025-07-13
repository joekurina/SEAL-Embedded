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

-- VHDL created from flt_fft_example_DUT_multBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_correctRounding_3x4cq5c30ot5uob117798ja8pmapoh2ph82desc063061663c61i65oc2765762di5p62v65vi2e65e62kc0360uq5ux0ao30cd06cj0of0cdj6of0cd16ol0qcz
-- VHDL created on Tue Jul  8 08:39:19 2025


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.NUMERIC_STD.all;
use IEEE.MATH_REAL.all;
use std.TextIO.all;
use work.dspba_library_package.all;

LIBRARY altera_mf;
USE altera_mf.altera_mf_components.all;
LIBRARY altera_lnsim;
USE altera_lnsim.altera_lnsim_components.altera_syncram;

library tennm;
use tennm.tennm_components.tennm_mac;
use tennm.tennm_components.tennm_fp_mac;

USE work.fft_example_DUT_safe_path.all;
entity flt_fft_example_DUT_multBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFlo0000of0cdj6of0cd16ol0qcz is
    port (
        in_0 : in std_logic_vector(63 downto 0);  -- float64_m52
        in_1 : in std_logic_vector(63 downto 0);  -- float64_m52
        out_primWireOut : out std_logic_vector(63 downto 0);  -- float64_m52
        clk : in std_logic;
        areset : in std_logic
    );
end flt_fft_example_DUT_multBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFlo0000of0cdj6of0cd16ol0qcz;

architecture normal of flt_fft_example_DUT_multBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFlo0000of0cdj6of0cd16ol0qcz is

    attribute altera_attribute : string;
    attribute altera_attribute of normal : architecture is "-name AUTO_SHIFT_REGISTER_RECOGNITION OFF; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 10037; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 15400; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 12020; -name MESSAGE_DISABLE 12030; -name MESSAGE_DISABLE 12010; -name MESSAGE_DISABLE 12110; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 13410; -name MESSAGE_DISABLE 113007";
    
    signal GND_q : STD_LOGIC_VECTOR (0 downto 0);
    signal VCC_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expX_uid7_block_rsrvd_fix_b : STD_LOGIC_VECTOR (10 downto 0);
    signal expY_uid8_block_rsrvd_fix_b : STD_LOGIC_VECTOR (10 downto 0);
    signal signX_uid9_block_rsrvd_fix_b : STD_LOGIC_VECTOR (0 downto 0);
    signal signY_uid10_block_rsrvd_fix_b : STD_LOGIC_VECTOR (0 downto 0);
    signal cstAllOWE_uid11_block_rsrvd_fix_q : STD_LOGIC_VECTOR (10 downto 0);
    signal cstZeroWF_uid12_block_rsrvd_fix_q : STD_LOGIC_VECTOR (51 downto 0);
    signal cstAllZWE_uid13_block_rsrvd_fix_q : STD_LOGIC_VECTOR (10 downto 0);
    signal frac_x_uid15_block_rsrvd_fix_b : STD_LOGIC_VECTOR (51 downto 0);
    signal excZ_x_uid16_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excZ_x_uid16_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expXIsMax_uid17_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal expXIsMax_uid17_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal fracXIsZero_uid18_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal fracXIsZero_uid18_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal fracXIsNotZero_uid19_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excI_x_uid20_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excN_x_uid21_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal invExpXIsMax_uid22_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal InvExpXIsZero_uid23_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excR_x_uid24_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal frac_y_uid29_block_rsrvd_fix_b : STD_LOGIC_VECTOR (51 downto 0);
    signal excZ_y_uid30_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excZ_y_uid30_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expXIsMax_uid31_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal expXIsMax_uid31_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal fracXIsZero_uid32_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal fracXIsZero_uid32_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal fracXIsNotZero_uid33_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excI_y_uid34_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excN_y_uid35_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal invExpXIsMax_uid36_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal InvExpXIsZero_uid37_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excR_y_uid38_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal ofracX_uid41_block_rsrvd_fix_q : STD_LOGIC_VECTOR (52 downto 0);
    signal ofracY_uid44_block_rsrvd_fix_q : STD_LOGIC_VECTOR (52 downto 0);
    signal expSum_uid45_block_rsrvd_fix_a : STD_LOGIC_VECTOR (11 downto 0);
    signal expSum_uid45_block_rsrvd_fix_b : STD_LOGIC_VECTOR (11 downto 0);
    signal expSum_uid45_block_rsrvd_fix_o : STD_LOGIC_VECTOR (11 downto 0);
    signal expSum_uid45_block_rsrvd_fix_q : STD_LOGIC_VECTOR (11 downto 0);
    signal biasInc_uid46_block_rsrvd_fix_q : STD_LOGIC_VECTOR (12 downto 0);
    signal expSumMBias_uid47_block_rsrvd_fix_a : STD_LOGIC_VECTOR (14 downto 0);
    signal expSumMBias_uid47_block_rsrvd_fix_b : STD_LOGIC_VECTOR (14 downto 0);
    signal expSumMBias_uid47_block_rsrvd_fix_o : STD_LOGIC_VECTOR (14 downto 0);
    signal expSumMBias_uid47_block_rsrvd_fix_q : STD_LOGIC_VECTOR (13 downto 0);
    signal signR_uid49_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal signR_uid49_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal normalizeBit_uid50_block_rsrvd_fix_b : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNormHigh_uid52_block_rsrvd_fix_in : STD_LOGIC_VECTOR (104 downto 0);
    signal fracRPostNormHigh_uid52_block_rsrvd_fix_b : STD_LOGIC_VECTOR (52 downto 0);
    signal fracRPostNormLow_uid53_block_rsrvd_fix_in : STD_LOGIC_VECTOR (103 downto 0);
    signal fracRPostNormLow_uid53_block_rsrvd_fix_b : STD_LOGIC_VECTOR (52 downto 0);
    signal fracRPostNorm_uid54_block_rsrvd_fix_s : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm_uid54_block_rsrvd_fix_q : STD_LOGIC_VECTOR (52 downto 0);
    signal stickyRange_uid55_block_rsrvd_fix_in : STD_LOGIC_VECTOR (50 downto 0);
    signal stickyRange_uid55_block_rsrvd_fix_b : STD_LOGIC_VECTOR (50 downto 0);
    signal extraStickyBitOfProd_uid56_block_rsrvd_fix_in : STD_LOGIC_VECTOR (51 downto 0);
    signal extraStickyBitOfProd_uid56_block_rsrvd_fix_b : STD_LOGIC_VECTOR (0 downto 0);
    signal extraStickyBit_uid57_block_rsrvd_fix_s : STD_LOGIC_VECTOR (0 downto 0);
    signal extraStickyBit_uid57_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal stickyExtendedRange_uid58_block_rsrvd_fix_q : STD_LOGIC_VECTOR (51 downto 0);
    signal stickyRangeComparator_uid60_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal sticky_uid61_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPostNorm1dto0_uid62_block_rsrvd_fix_in : STD_LOGIC_VECTOR (1 downto 0);
    signal fracRPostNorm1dto0_uid62_block_rsrvd_fix_b : STD_LOGIC_VECTOR (1 downto 0);
    signal lrs_uid63_block_rsrvd_fix_q : STD_LOGIC_VECTOR (2 downto 0);
    signal roundBitDetectionConstant_uid64_block_rsrvd_fix_q : STD_LOGIC_VECTOR (2 downto 0);
    signal roundBitDetectionPattern_uid65_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal roundBitDetectionPattern_uid65_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal roundBit_uid66_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expFracPreRound_uid67_block_rsrvd_fix_q : STD_LOGIC_VECTOR (66 downto 0);
    signal roundBitAndNormalizationOp_uid69_block_rsrvd_fix_q : STD_LOGIC_VECTOR (54 downto 0);
    signal expFracRPostRounding_uid70_block_rsrvd_fix_a : STD_LOGIC_VECTOR (68 downto 0);
    signal expFracRPostRounding_uid70_block_rsrvd_fix_b : STD_LOGIC_VECTOR (68 downto 0);
    signal expFracRPostRounding_uid70_block_rsrvd_fix_o : STD_LOGIC_VECTOR (68 downto 0);
    signal expFracRPostRounding_uid70_block_rsrvd_fix_q : STD_LOGIC_VECTOR (67 downto 0);
    signal fracRPreExc_uid71_block_rsrvd_fix_in : STD_LOGIC_VECTOR (52 downto 0);
    signal fracRPreExc_uid71_block_rsrvd_fix_b : STD_LOGIC_VECTOR (51 downto 0);
    signal expRPreExcExt_uid72_block_rsrvd_fix_b : STD_LOGIC_VECTOR (14 downto 0);
    signal expRPreExc_uid73_block_rsrvd_fix_in : STD_LOGIC_VECTOR (10 downto 0);
    signal expRPreExc_uid73_block_rsrvd_fix_b : STD_LOGIC_VECTOR (10 downto 0);
    signal expUdf_uid74_block_rsrvd_fix_a : STD_LOGIC_VECTOR (16 downto 0);
    signal expUdf_uid74_block_rsrvd_fix_b : STD_LOGIC_VECTOR (16 downto 0);
    signal expUdf_uid74_block_rsrvd_fix_o : STD_LOGIC_VECTOR (16 downto 0);
    signal expUdf_uid74_block_rsrvd_fix_n : STD_LOGIC_VECTOR (0 downto 0);
    signal expOvf_uid76_block_rsrvd_fix_a : STD_LOGIC_VECTOR (16 downto 0);
    signal expOvf_uid76_block_rsrvd_fix_b : STD_LOGIC_VECTOR (16 downto 0);
    signal expOvf_uid76_block_rsrvd_fix_o : STD_LOGIC_VECTOR (16 downto 0);
    signal expOvf_uid76_block_rsrvd_fix_n : STD_LOGIC_VECTOR (0 downto 0);
    signal excXZAndExcYZ_uid77_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excXZAndExcYR_uid78_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excYZAndExcXR_uid79_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excZC3_uid80_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excRZero_uid81_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excXIAndExcYI_uid82_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excXRAndExcYI_uid83_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excYRAndExcXI_uid84_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal ExcROvfAndInReg_uid85_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excRInf_uid86_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excYZAndExcXI_uid87_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excXZAndExcYI_uid88_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal ZeroTimesInf_uid89_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excRNaN_uid90_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal concExc_uid91_block_rsrvd_fix_q : STD_LOGIC_VECTOR (2 downto 0);
    signal excREnc_uid92_block_rsrvd_fix_q : STD_LOGIC_VECTOR (1 downto 0);
    signal oneFracRPostExc2_uid93_block_rsrvd_fix_q : STD_LOGIC_VECTOR (51 downto 0);
    signal fracRPostExc_uid96_block_rsrvd_fix_s : STD_LOGIC_VECTOR (1 downto 0);
    signal fracRPostExc_uid96_block_rsrvd_fix_q : STD_LOGIC_VECTOR (51 downto 0);
    signal expRPostExc_uid101_block_rsrvd_fix_s : STD_LOGIC_VECTOR (1 downto 0);
    signal expRPostExc_uid101_block_rsrvd_fix_q : STD_LOGIC_VECTOR (10 downto 0);
    signal invExcRNaN_uid102_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal signRPostExc_uid103_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal signRPostExc_uid103_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal R_uid104_block_rsrvd_fix_q : STD_LOGIC_VECTOR (63 downto 0);
    signal aboveLeftY_mergedSignalTM_uid112_prod_uid48_block_rsrvd_fix_q : STD_LOGIC_VECTOR (26 downto 0);
    signal rightBottomX_mergedSignalTM_uid116_prod_uid48_block_rsrvd_fix_q : STD_LOGIC_VECTOR (26 downto 0);
    signal multSumOfTwoTS_uid120_prod_uid48_block_rsrvd_fix_in : STD_LOGIC_VECTOR (54 downto 0);
    signal multSumOfTwoTS_uid120_prod_uid48_block_rsrvd_fix_b : STD_LOGIC_VECTOR (54 downto 0);
    signal add0_uid130_prod_uid48_block_rsrvd_fix_q : STD_LOGIC_VECTOR (107 downto 0);
    signal add1sumAHighB_uid133_prod_uid48_block_rsrvd_fix_a : STD_LOGIC_VECTOR (81 downto 0);
    signal add1sumAHighB_uid133_prod_uid48_block_rsrvd_fix_b : STD_LOGIC_VECTOR (81 downto 0);
    signal add1sumAHighB_uid133_prod_uid48_block_rsrvd_fix_o : STD_LOGIC_VECTOR (81 downto 0);
    signal add1sumAHighB_uid133_prod_uid48_block_rsrvd_fix_q : STD_LOGIC_VECTOR (81 downto 0);
    signal add1_uid134_prod_uid48_block_rsrvd_fix_q : STD_LOGIC_VECTOR (108 downto 0);
    signal osig_uid135_prod_uid48_block_rsrvd_fix_in : STD_LOGIC_VECTOR (107 downto 0);
    signal osig_uid135_prod_uid48_block_rsrvd_fix_b : STD_LOGIC_VECTOR (105 downto 0);
    signal topProd_uid108_prod_uid48_block_rsrvd_fix_cma_reset : std_logic;
    signal topProd_uid108_prod_uid48_block_rsrvd_fix_cma_a0 : STD_LOGIC_VECTOR (26 downto 0);
    signal topProd_uid108_prod_uid48_block_rsrvd_fix_cma_c0 : STD_LOGIC_VECTOR (26 downto 0);
    signal topProd_uid108_prod_uid48_block_rsrvd_fix_cma_s0 : STD_LOGIC_VECTOR (53 downto 0);
    signal topProd_uid108_prod_uid48_block_rsrvd_fix_cma_qq0 : STD_LOGIC_VECTOR (53 downto 0);
    signal topProd_uid108_prod_uid48_block_rsrvd_fix_cma_q : STD_LOGIC_VECTOR (53 downto 0);
    signal topProd_uid108_prod_uid48_block_rsrvd_fix_cma_ena0 : std_logic;
    signal topProd_uid108_prod_uid48_block_rsrvd_fix_cma_ena1 : std_logic;
    signal topProd_uid108_prod_uid48_block_rsrvd_fix_cma_ena2 : std_logic;
    signal sm0_uid129_prod_uid48_block_rsrvd_fix_cma_reset : std_logic;
    signal sm0_uid129_prod_uid48_block_rsrvd_fix_cma_a0 : STD_LOGIC_VECTOR (26 downto 0);
    signal sm0_uid129_prod_uid48_block_rsrvd_fix_cma_c0 : STD_LOGIC_VECTOR (26 downto 0);
    signal sm0_uid129_prod_uid48_block_rsrvd_fix_cma_s0 : STD_LOGIC_VECTOR (53 downto 0);
    signal sm0_uid129_prod_uid48_block_rsrvd_fix_cma_qq0 : STD_LOGIC_VECTOR (53 downto 0);
    signal sm0_uid129_prod_uid48_block_rsrvd_fix_cma_q : STD_LOGIC_VECTOR (53 downto 0);
    signal sm0_uid129_prod_uid48_block_rsrvd_fix_cma_ena0 : std_logic;
    signal sm0_uid129_prod_uid48_block_rsrvd_fix_cma_ena1 : std_logic;
    signal sm0_uid129_prod_uid48_block_rsrvd_fix_cma_ena2 : std_logic;
    signal multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_reset : std_logic;
    signal multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_a0 : STD_LOGIC_VECTOR (26 downto 0);
    signal multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_c0 : STD_LOGIC_VECTOR (26 downto 0);
    signal multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_a1 : STD_LOGIC_VECTOR (26 downto 0);
    signal multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_c1 : STD_LOGIC_VECTOR (26 downto 0);
    signal multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_s0 : STD_LOGIC_VECTOR (54 downto 0);
    signal multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_s1 : STD_LOGIC_VECTOR (63 downto 0);
    signal multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_qq0 : STD_LOGIC_VECTOR (54 downto 0);
    signal multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_q : STD_LOGIC_VECTOR (55 downto 0);
    signal multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_ena0 : std_logic;
    signal multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_ena1 : std_logic;
    signal multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_ena2 : std_logic;
    signal topRangeX_uid106_prod_uid48_block_rsrvd_fix_bit_select_merged_b : STD_LOGIC_VECTOR (26 downto 0);
    signal topRangeX_uid106_prod_uid48_block_rsrvd_fix_bit_select_merged_c : STD_LOGIC_VECTOR (25 downto 0);
    signal topRangeY_uid107_prod_uid48_block_rsrvd_fix_bit_select_merged_b : STD_LOGIC_VECTOR (26 downto 0);
    signal topRangeY_uid107_prod_uid48_block_rsrvd_fix_bit_select_merged_c : STD_LOGIC_VECTOR (25 downto 0);
    signal lowRangeB_uid131_prod_uid48_block_rsrvd_fix_bit_select_merged_b : STD_LOGIC_VECTOR (26 downto 0);
    signal lowRangeB_uid131_prod_uid48_block_rsrvd_fix_bit_select_merged_c : STD_LOGIC_VECTOR (80 downto 0);
    signal redist0_topRangeY_uid107_prod_uid48_block_rsrvd_fix_bit_select_merged_b_1_q : STD_LOGIC_VECTOR (26 downto 0);
    signal redist1_topRangeX_uid106_prod_uid48_block_rsrvd_fix_bit_select_merged_b_1_q : STD_LOGIC_VECTOR (26 downto 0);
    signal redist2_topRangeX_uid106_prod_uid48_block_rsrvd_fix_bit_select_merged_c_1_q : STD_LOGIC_VECTOR (25 downto 0);
    signal redist3_osig_uid135_prod_uid48_block_rsrvd_fix_b_1_q : STD_LOGIC_VECTOR (105 downto 0);
    signal redist4_aboveLeftY_mergedSignalTM_uid112_prod_uid48_block_rsrvd_fix_q_1_q : STD_LOGIC_VECTOR (26 downto 0);
    signal redist5_expRPreExc_uid73_block_rsrvd_fix_b_1_q : STD_LOGIC_VECTOR (10 downto 0);
    signal redist6_expRPreExcExt_uid72_block_rsrvd_fix_b_1_q : STD_LOGIC_VECTOR (14 downto 0);
    signal redist7_fracRPreExc_uid71_block_rsrvd_fix_b_2_q : STD_LOGIC_VECTOR (51 downto 0);
    signal redist7_fracRPreExc_uid71_block_rsrvd_fix_b_2_delay_0 : STD_LOGIC_VECTOR (51 downto 0);
    signal redist8_fracRPostNorm_uid54_block_rsrvd_fix_q_1_q : STD_LOGIC_VECTOR (52 downto 0);
    signal redist9_normalizeBit_uid50_block_rsrvd_fix_b_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist10_signR_uid49_block_rsrvd_fix_q_9_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist12_fracXIsZero_uid32_block_rsrvd_fix_q_9_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist13_expXIsMax_uid31_block_rsrvd_fix_q_9_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist14_excZ_y_uid30_block_rsrvd_fix_q_9_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist15_fracXIsZero_uid18_block_rsrvd_fix_q_9_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist16_expXIsMax_uid17_block_rsrvd_fix_q_9_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist17_excZ_x_uid16_block_rsrvd_fix_q_9_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist11_expSum_uid45_block_rsrvd_fix_q_7_mem_reset0 : std_logic;
    signal redist11_expSum_uid45_block_rsrvd_fix_q_7_mem_ia : STD_LOGIC_VECTOR (11 downto 0);
    signal redist11_expSum_uid45_block_rsrvd_fix_q_7_mem_aa : STD_LOGIC_VECTOR (2 downto 0);
    signal redist11_expSum_uid45_block_rsrvd_fix_q_7_mem_ab : STD_LOGIC_VECTOR (2 downto 0);
    signal redist11_expSum_uid45_block_rsrvd_fix_q_7_mem_iq : STD_LOGIC_VECTOR (11 downto 0);
    signal redist11_expSum_uid45_block_rsrvd_fix_q_7_mem_q : STD_LOGIC_VECTOR (11 downto 0);
    signal redist11_expSum_uid45_block_rsrvd_fix_q_7_wraddr_q : STD_LOGIC_VECTOR (2 downto 0);
    -- Initial-value here is arbitrary, but a resolved value is necessary for simulation.
    signal redist11_expSum_uid45_block_rsrvd_fix_q_7_wraddr_i : UNSIGNED (2 downto 0) := "111";
    attribute preserve_syn_only : boolean;
    attribute preserve_syn_only of redist11_expSum_uid45_block_rsrvd_fix_q_7_wraddr_i : signal is true;
    signal redist11_expSum_uid45_block_rsrvd_fix_q_7_offset_q : STD_LOGIC_VECTOR (2 downto 0);
    signal redist11_expSum_uid45_block_rsrvd_fix_q_7_rdcnt_a : STD_LOGIC_VECTOR (3 downto 0);
    signal redist11_expSum_uid45_block_rsrvd_fix_q_7_rdcnt_b : STD_LOGIC_VECTOR (3 downto 0);
    signal redist11_expSum_uid45_block_rsrvd_fix_q_7_rdcnt_o : STD_LOGIC_VECTOR (3 downto 0);
    signal redist11_expSum_uid45_block_rsrvd_fix_q_7_rdcnt_q : STD_LOGIC_VECTOR (3 downto 0);

begin


    -- frac_x_uid15_block_rsrvd_fix(BITSELECT,14)@0
    frac_x_uid15_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(in_0(51 downto 0));

    -- cstZeroWF_uid12_block_rsrvd_fix(CONSTANT,11)
    cstZeroWF_uid12_block_rsrvd_fix_q <= "0000000000000000000000000000000000000000000000000000";

    -- fracXIsZero_uid18_block_rsrvd_fix(LOGICAL,17)@0 + 1
    fracXIsZero_uid18_block_rsrvd_fix_qi <= "1" WHEN cstZeroWF_uid12_block_rsrvd_fix_q = frac_x_uid15_block_rsrvd_fix_b ELSE "0";
    fracXIsZero_uid18_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => fracXIsZero_uid18_block_rsrvd_fix_qi, xout => fracXIsZero_uid18_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- redist15_fracXIsZero_uid18_block_rsrvd_fix_q_9(DELAY,157)
    redist15_fracXIsZero_uid18_block_rsrvd_fix_q_9 : dspba_delay
    GENERIC MAP ( width => 1, depth => 8, reset_kind => "NONE", phase => 0, modulus => 1024 )
    PORT MAP ( xin => fracXIsZero_uid18_block_rsrvd_fix_q, xout => redist15_fracXIsZero_uid18_block_rsrvd_fix_q_9_q, clk => clk, aclr => areset, ena => '1' );

    -- cstAllOWE_uid11_block_rsrvd_fix(CONSTANT,10)
    cstAllOWE_uid11_block_rsrvd_fix_q <= "11111111111";

    -- expX_uid7_block_rsrvd_fix(BITSELECT,6)@0
    expX_uid7_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(in_0(62 downto 52));

    -- expXIsMax_uid17_block_rsrvd_fix(LOGICAL,16)@0 + 1
    expXIsMax_uid17_block_rsrvd_fix_qi <= "1" WHEN expX_uid7_block_rsrvd_fix_b = cstAllOWE_uid11_block_rsrvd_fix_q ELSE "0";
    expXIsMax_uid17_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => expXIsMax_uid17_block_rsrvd_fix_qi, xout => expXIsMax_uid17_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- redist16_expXIsMax_uid17_block_rsrvd_fix_q_9(DELAY,158)
    redist16_expXIsMax_uid17_block_rsrvd_fix_q_9 : dspba_delay
    GENERIC MAP ( width => 1, depth => 8, reset_kind => "NONE", phase => 0, modulus => 1024 )
    PORT MAP ( xin => expXIsMax_uid17_block_rsrvd_fix_q, xout => redist16_expXIsMax_uid17_block_rsrvd_fix_q_9_q, clk => clk, aclr => areset, ena => '1' );

    -- excI_x_uid20_block_rsrvd_fix(LOGICAL,19)@9
    excI_x_uid20_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(redist16_expXIsMax_uid17_block_rsrvd_fix_q_9_q and redist15_fracXIsZero_uid18_block_rsrvd_fix_q_9_q);

    -- cstAllZWE_uid13_block_rsrvd_fix(CONSTANT,12)
    cstAllZWE_uid13_block_rsrvd_fix_q <= "00000000000";

    -- expY_uid8_block_rsrvd_fix(BITSELECT,7)@0
    expY_uid8_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(in_1(62 downto 52));

    -- excZ_y_uid30_block_rsrvd_fix(LOGICAL,29)@0 + 1
    excZ_y_uid30_block_rsrvd_fix_qi <= "1" WHEN expY_uid8_block_rsrvd_fix_b = cstAllZWE_uid13_block_rsrvd_fix_q ELSE "0";
    excZ_y_uid30_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => excZ_y_uid30_block_rsrvd_fix_qi, xout => excZ_y_uid30_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- redist14_excZ_y_uid30_block_rsrvd_fix_q_9(DELAY,156)
    redist14_excZ_y_uid30_block_rsrvd_fix_q_9 : dspba_delay
    GENERIC MAP ( width => 1, depth => 8, reset_kind => "NONE", phase => 0, modulus => 1024 )
    PORT MAP ( xin => excZ_y_uid30_block_rsrvd_fix_q, xout => redist14_excZ_y_uid30_block_rsrvd_fix_q_9_q, clk => clk, aclr => areset, ena => '1' );

    -- excYZAndExcXI_uid87_block_rsrvd_fix(LOGICAL,86)@9
    excYZAndExcXI_uid87_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(redist14_excZ_y_uid30_block_rsrvd_fix_q_9_q and excI_x_uid20_block_rsrvd_fix_q);

    -- frac_y_uid29_block_rsrvd_fix(BITSELECT,28)@0
    frac_y_uid29_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(in_1(51 downto 0));

    -- fracXIsZero_uid32_block_rsrvd_fix(LOGICAL,31)@0 + 1
    fracXIsZero_uid32_block_rsrvd_fix_qi <= "1" WHEN cstZeroWF_uid12_block_rsrvd_fix_q = frac_y_uid29_block_rsrvd_fix_b ELSE "0";
    fracXIsZero_uid32_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => fracXIsZero_uid32_block_rsrvd_fix_qi, xout => fracXIsZero_uid32_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- redist12_fracXIsZero_uid32_block_rsrvd_fix_q_9(DELAY,154)
    redist12_fracXIsZero_uid32_block_rsrvd_fix_q_9 : dspba_delay
    GENERIC MAP ( width => 1, depth => 8, reset_kind => "NONE", phase => 0, modulus => 1024 )
    PORT MAP ( xin => fracXIsZero_uid32_block_rsrvd_fix_q, xout => redist12_fracXIsZero_uid32_block_rsrvd_fix_q_9_q, clk => clk, aclr => areset, ena => '1' );

    -- expXIsMax_uid31_block_rsrvd_fix(LOGICAL,30)@0 + 1
    expXIsMax_uid31_block_rsrvd_fix_qi <= "1" WHEN expY_uid8_block_rsrvd_fix_b = cstAllOWE_uid11_block_rsrvd_fix_q ELSE "0";
    expXIsMax_uid31_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => expXIsMax_uid31_block_rsrvd_fix_qi, xout => expXIsMax_uid31_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- redist13_expXIsMax_uid31_block_rsrvd_fix_q_9(DELAY,155)
    redist13_expXIsMax_uid31_block_rsrvd_fix_q_9 : dspba_delay
    GENERIC MAP ( width => 1, depth => 8, reset_kind => "NONE", phase => 0, modulus => 1024 )
    PORT MAP ( xin => expXIsMax_uid31_block_rsrvd_fix_q, xout => redist13_expXIsMax_uid31_block_rsrvd_fix_q_9_q, clk => clk, aclr => areset, ena => '1' );

    -- excI_y_uid34_block_rsrvd_fix(LOGICAL,33)@9
    excI_y_uid34_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(redist13_expXIsMax_uid31_block_rsrvd_fix_q_9_q and redist12_fracXIsZero_uid32_block_rsrvd_fix_q_9_q);

    -- excZ_x_uid16_block_rsrvd_fix(LOGICAL,15)@0 + 1
    excZ_x_uid16_block_rsrvd_fix_qi <= "1" WHEN expX_uid7_block_rsrvd_fix_b = cstAllZWE_uid13_block_rsrvd_fix_q ELSE "0";
    excZ_x_uid16_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => excZ_x_uid16_block_rsrvd_fix_qi, xout => excZ_x_uid16_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- redist17_excZ_x_uid16_block_rsrvd_fix_q_9(DELAY,159)
    redist17_excZ_x_uid16_block_rsrvd_fix_q_9 : dspba_delay
    GENERIC MAP ( width => 1, depth => 8, reset_kind => "NONE", phase => 0, modulus => 1024 )
    PORT MAP ( xin => excZ_x_uid16_block_rsrvd_fix_q, xout => redist17_excZ_x_uid16_block_rsrvd_fix_q_9_q, clk => clk, aclr => areset, ena => '1' );

    -- excXZAndExcYI_uid88_block_rsrvd_fix(LOGICAL,87)@9
    excXZAndExcYI_uid88_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(redist17_excZ_x_uid16_block_rsrvd_fix_q_9_q and excI_y_uid34_block_rsrvd_fix_q);

    -- ZeroTimesInf_uid89_block_rsrvd_fix(LOGICAL,88)@9
    ZeroTimesInf_uid89_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(excXZAndExcYI_uid88_block_rsrvd_fix_q or excYZAndExcXI_uid87_block_rsrvd_fix_q);

    -- fracXIsNotZero_uid33_block_rsrvd_fix(LOGICAL,32)@9
    fracXIsNotZero_uid33_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (redist12_fracXIsZero_uid32_block_rsrvd_fix_q_9_q));

    -- excN_y_uid35_block_rsrvd_fix(LOGICAL,34)@9
    excN_y_uid35_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(redist13_expXIsMax_uid31_block_rsrvd_fix_q_9_q and fracXIsNotZero_uid33_block_rsrvd_fix_q);

    -- fracXIsNotZero_uid19_block_rsrvd_fix(LOGICAL,18)@9
    fracXIsNotZero_uid19_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (redist15_fracXIsZero_uid18_block_rsrvd_fix_q_9_q));

    -- excN_x_uid21_block_rsrvd_fix(LOGICAL,20)@9
    excN_x_uid21_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(redist16_expXIsMax_uid17_block_rsrvd_fix_q_9_q and fracXIsNotZero_uid19_block_rsrvd_fix_q);

    -- excRNaN_uid90_block_rsrvd_fix(LOGICAL,89)@9
    excRNaN_uid90_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(excN_x_uid21_block_rsrvd_fix_q or excN_y_uid35_block_rsrvd_fix_q or ZeroTimesInf_uid89_block_rsrvd_fix_q);

    -- invExcRNaN_uid102_block_rsrvd_fix(LOGICAL,101)@9
    invExcRNaN_uid102_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (excRNaN_uid90_block_rsrvd_fix_q));

    -- signY_uid10_block_rsrvd_fix(BITSELECT,9)@0
    signY_uid10_block_rsrvd_fix_b <= in_1(63 downto 63);

    -- signX_uid9_block_rsrvd_fix(BITSELECT,8)@0
    signX_uid9_block_rsrvd_fix_b <= in_0(63 downto 63);

    -- signR_uid49_block_rsrvd_fix(LOGICAL,48)@0 + 1
    signR_uid49_block_rsrvd_fix_qi <= signX_uid9_block_rsrvd_fix_b xor signY_uid10_block_rsrvd_fix_b;
    signR_uid49_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => signR_uid49_block_rsrvd_fix_qi, xout => signR_uid49_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- redist10_signR_uid49_block_rsrvd_fix_q_9(DELAY,152)
    redist10_signR_uid49_block_rsrvd_fix_q_9 : dspba_delay
    GENERIC MAP ( width => 1, depth => 8, reset_kind => "NONE", phase => 0, modulus => 1024 )
    PORT MAP ( xin => signR_uid49_block_rsrvd_fix_q, xout => redist10_signR_uid49_block_rsrvd_fix_q_9_q, clk => clk, aclr => areset, ena => '1' );

    -- VCC(CONSTANT,1)
    VCC_q <= "1";

    -- signRPostExc_uid103_block_rsrvd_fix(LOGICAL,102)@9 + 1
    signRPostExc_uid103_block_rsrvd_fix_qi <= redist10_signR_uid49_block_rsrvd_fix_q_9_q and invExcRNaN_uid102_block_rsrvd_fix_q;
    signRPostExc_uid103_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => signRPostExc_uid103_block_rsrvd_fix_qi, xout => signRPostExc_uid103_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- GND(CONSTANT,0)
    GND_q <= "0";

    -- ofracX_uid41_block_rsrvd_fix(BITJOIN,40)@0
    ofracX_uid41_block_rsrvd_fix_q <= VCC_q & frac_x_uid15_block_rsrvd_fix_b;

    -- topRangeX_uid106_prod_uid48_block_rsrvd_fix_bit_select_merged(BITSELECT,139)@0
    topRangeX_uid106_prod_uid48_block_rsrvd_fix_bit_select_merged_b <= STD_LOGIC_VECTOR(ofracX_uid41_block_rsrvd_fix_q(52 downto 26));
    topRangeX_uid106_prod_uid48_block_rsrvd_fix_bit_select_merged_c <= STD_LOGIC_VECTOR(ofracX_uid41_block_rsrvd_fix_q(25 downto 0));

    -- ofracY_uid44_block_rsrvd_fix(BITJOIN,43)@0
    ofracY_uid44_block_rsrvd_fix_q <= VCC_q & frac_y_uid29_block_rsrvd_fix_b;

    -- topRangeY_uid107_prod_uid48_block_rsrvd_fix_bit_select_merged(BITSELECT,140)@0
    topRangeY_uid107_prod_uid48_block_rsrvd_fix_bit_select_merged_b <= STD_LOGIC_VECTOR(ofracY_uid44_block_rsrvd_fix_q(52 downto 26));
    topRangeY_uid107_prod_uid48_block_rsrvd_fix_bit_select_merged_c <= STD_LOGIC_VECTOR(ofracY_uid44_block_rsrvd_fix_q(25 downto 0));

    -- aboveLeftY_mergedSignalTM_uid112_prod_uid48_block_rsrvd_fix(BITJOIN,111)@0
    aboveLeftY_mergedSignalTM_uid112_prod_uid48_block_rsrvd_fix_q <= topRangeY_uid107_prod_uid48_block_rsrvd_fix_bit_select_merged_c & GND_q;

    -- redist2_topRangeX_uid106_prod_uid48_block_rsrvd_fix_bit_select_merged_c_1(DELAY,144)
    redist2_topRangeX_uid106_prod_uid48_block_rsrvd_fix_bit_select_merged_c_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist2_topRangeX_uid106_prod_uid48_block_rsrvd_fix_bit_select_merged_c_1_q <= topRangeX_uid106_prod_uid48_block_rsrvd_fix_bit_select_merged_c;
            END IF;
        END IF;
    END PROCESS;

    -- rightBottomX_mergedSignalTM_uid116_prod_uid48_block_rsrvd_fix(BITJOIN,115)@1
    rightBottomX_mergedSignalTM_uid116_prod_uid48_block_rsrvd_fix_q <= redist2_topRangeX_uid106_prod_uid48_block_rsrvd_fix_bit_select_merged_c_1_q & GND_q;

    -- redist0_topRangeY_uid107_prod_uid48_block_rsrvd_fix_bit_select_merged_b_1(DELAY,142)
    redist0_topRangeY_uid107_prod_uid48_block_rsrvd_fix_bit_select_merged_b_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist0_topRangeY_uid107_prod_uid48_block_rsrvd_fix_bit_select_merged_b_1_q <= topRangeY_uid107_prod_uid48_block_rsrvd_fix_bit_select_merged_b;
            END IF;
        END IF;
    END PROCESS;

    -- multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma(CHAINMULTADD,138)@0 + 5
    -- in a@1
    -- in b@4
    -- in d@1
    -- in h@1
    -- in j@1
    -- out q@6
    multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_reset <= areset;
    multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_ena0 <= '1';
    multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_ena1 <= multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_ena0;
    multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_ena2 <= multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_ena0;

    multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_a0 <= STD_LOGIC_VECTOR(RESIZE(UNSIGNED(redist0_topRangeY_uid107_prod_uid48_block_rsrvd_fix_bit_select_merged_b_1_q),27));
    multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_c0 <= STD_LOGIC_VECTOR(RESIZE(UNSIGNED(rightBottomX_mergedSignalTM_uid116_prod_uid48_block_rsrvd_fix_q),27));
    multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_a1 <= STD_LOGIC_VECTOR(RESIZE(UNSIGNED(aboveLeftY_mergedSignalTM_uid112_prod_uid48_block_rsrvd_fix_q),27));
    multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_c1 <= STD_LOGIC_VECTOR(RESIZE(UNSIGNED(topRangeX_uid106_prod_uid48_block_rsrvd_fix_bit_select_merged_b),27));
    multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_DSP1 : tennm_mac
    GENERIC MAP (
        operation_mode => "m27x27",
        chain_inout_width => 64,
        clear_type => "none",
        use_chainadder => "false",
        ay_scan_in_clken => "0",
        ay_scan_in_width => 27,
        ax_clken => "0",
        ax_width => 27,
        signed_may => "false",
        signed_max => "false",
        input_pipeline_clken => "2",
        second_pipeline_clken => "2",
        output_clken => "1"
    )
    PORT MAP (
        clk => clk,
        ena(0) => multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_ena0,
        ena(1) => multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_ena1,
        ena(2) => multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_ena2,
        clr(0) => '0',
        clr(1) => '0',
        ay => multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_a1,
        ax => multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_c1,
        chainout => multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_s1
    );
    multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_DSP0 : tennm_mac
    GENERIC MAP (
        operation_mode => "m27x27",
        chain_inout_width => 64,
        clear_type => "none",
        use_chainadder => "true",
        ay_scan_in_clken => "0",
        ay_scan_in_width => 27,
        ax_clken => "0",
        ax_width => 27,
        signed_may => "false",
        signed_max => "false",
        input_pipeline_clken => "2",
        second_pipeline_clken => "2",
        output_clken => "1",
        result_a_width => 55
    )
    PORT MAP (
        clk => clk,
        ena(0) => multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_ena0,
        ena(1) => multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_ena1,
        ena(2) => multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_ena2,
        clr(0) => '0',
        clr(1) => '0',
        ay => multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_a0,
        ax => multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_c0,
        chainin => multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_s1,
        resulta => multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_s0
    );
    multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_delay0 : dspba_delay
    GENERIC MAP ( width => 55, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_s0, xout => multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_qq0, clk => clk, aclr => areset, ena => '1' );
    multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_q <= STD_LOGIC_VECTOR(std_logic_vector(resize(signed(multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_qq0(54 downto 0)), 56)));

    -- multSumOfTwoTS_uid120_prod_uid48_block_rsrvd_fix(BITSELECT,119)@6
    multSumOfTwoTS_uid120_prod_uid48_block_rsrvd_fix_in <= multSumOfTwoTS_uid119_prod_uid48_block_rsrvd_fix_cma_q(54 downto 0);
    multSumOfTwoTS_uid120_prod_uid48_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(multSumOfTwoTS_uid120_prod_uid48_block_rsrvd_fix_in(54 downto 0));

    -- add1sumAHighB_uid133_prod_uid48_block_rsrvd_fix(ADD,132)@6
    add1sumAHighB_uid133_prod_uid48_block_rsrvd_fix_a <= STD_LOGIC_VECTOR("000000000000000000000000000" & multSumOfTwoTS_uid120_prod_uid48_block_rsrvd_fix_b);
    add1sumAHighB_uid133_prod_uid48_block_rsrvd_fix_b <= STD_LOGIC_VECTOR("0" & lowRangeB_uid131_prod_uid48_block_rsrvd_fix_bit_select_merged_c);
    add1sumAHighB_uid133_prod_uid48_block_rsrvd_fix_o <= STD_LOGIC_VECTOR(UNSIGNED(add1sumAHighB_uid133_prod_uid48_block_rsrvd_fix_a) + UNSIGNED(add1sumAHighB_uid133_prod_uid48_block_rsrvd_fix_b));
    add1sumAHighB_uid133_prod_uid48_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(add1sumAHighB_uid133_prod_uid48_block_rsrvd_fix_o(81 downto 0));

    -- redist1_topRangeX_uid106_prod_uid48_block_rsrvd_fix_bit_select_merged_b_1(DELAY,143)
    redist1_topRangeX_uid106_prod_uid48_block_rsrvd_fix_bit_select_merged_b_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist1_topRangeX_uid106_prod_uid48_block_rsrvd_fix_bit_select_merged_b_1_q <= topRangeX_uid106_prod_uid48_block_rsrvd_fix_bit_select_merged_b;
            END IF;
        END IF;
    END PROCESS;

    -- topProd_uid108_prod_uid48_block_rsrvd_fix_cma(CHAINMULTADD,136)@1 + 5
    -- in b@4
    topProd_uid108_prod_uid48_block_rsrvd_fix_cma_reset <= areset;
    topProd_uid108_prod_uid48_block_rsrvd_fix_cma_ena0 <= '1';
    topProd_uid108_prod_uid48_block_rsrvd_fix_cma_ena1 <= topProd_uid108_prod_uid48_block_rsrvd_fix_cma_ena0;
    topProd_uid108_prod_uid48_block_rsrvd_fix_cma_ena2 <= topProd_uid108_prod_uid48_block_rsrvd_fix_cma_ena0;

    topProd_uid108_prod_uid48_block_rsrvd_fix_cma_a0 <= STD_LOGIC_VECTOR(RESIZE(UNSIGNED(redist1_topRangeX_uid106_prod_uid48_block_rsrvd_fix_bit_select_merged_b_1_q),27));
    topProd_uid108_prod_uid48_block_rsrvd_fix_cma_c0 <= STD_LOGIC_VECTOR(RESIZE(UNSIGNED(redist0_topRangeY_uid107_prod_uid48_block_rsrvd_fix_bit_select_merged_b_1_q),27));
    topProd_uid108_prod_uid48_block_rsrvd_fix_cma_DSP0 : tennm_mac
    GENERIC MAP (
        operation_mode => "m27x27",
        clear_type => "none",
        use_chainadder => "false",
        ay_scan_in_clken => "0",
        ay_scan_in_width => 27,
        ax_clken => "0",
        ax_width => 27,
        signed_may => "false",
        signed_max => "false",
        input_pipeline_clken => "2",
        second_pipeline_clken => "2",
        output_clken => "1",
        result_a_width => 54
    )
    PORT MAP (
        clk => clk,
        ena(0) => topProd_uid108_prod_uid48_block_rsrvd_fix_cma_ena0,
        ena(1) => topProd_uid108_prod_uid48_block_rsrvd_fix_cma_ena1,
        ena(2) => topProd_uid108_prod_uid48_block_rsrvd_fix_cma_ena2,
        clr(0) => '0',
        clr(1) => '0',
        ay => topProd_uid108_prod_uid48_block_rsrvd_fix_cma_a0,
        ax => topProd_uid108_prod_uid48_block_rsrvd_fix_cma_c0,
        resulta => topProd_uid108_prod_uid48_block_rsrvd_fix_cma_s0
    );
    topProd_uid108_prod_uid48_block_rsrvd_fix_cma_delay0 : dspba_delay
    GENERIC MAP ( width => 54, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => topProd_uid108_prod_uid48_block_rsrvd_fix_cma_s0, xout => topProd_uid108_prod_uid48_block_rsrvd_fix_cma_qq0, clk => clk, aclr => areset, ena => '1' );
    topProd_uid108_prod_uid48_block_rsrvd_fix_cma_q <= STD_LOGIC_VECTOR(topProd_uid108_prod_uid48_block_rsrvd_fix_cma_qq0(53 downto 0));

    -- redist4_aboveLeftY_mergedSignalTM_uid112_prod_uid48_block_rsrvd_fix_q_1(DELAY,146)
    redist4_aboveLeftY_mergedSignalTM_uid112_prod_uid48_block_rsrvd_fix_q_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist4_aboveLeftY_mergedSignalTM_uid112_prod_uid48_block_rsrvd_fix_q_1_q <= aboveLeftY_mergedSignalTM_uid112_prod_uid48_block_rsrvd_fix_q;
            END IF;
        END IF;
    END PROCESS;

    -- sm0_uid129_prod_uid48_block_rsrvd_fix_cma(CHAINMULTADD,137)@1 + 5
    -- in b@4
    sm0_uid129_prod_uid48_block_rsrvd_fix_cma_reset <= areset;
    sm0_uid129_prod_uid48_block_rsrvd_fix_cma_ena0 <= '1';
    sm0_uid129_prod_uid48_block_rsrvd_fix_cma_ena1 <= sm0_uid129_prod_uid48_block_rsrvd_fix_cma_ena0;
    sm0_uid129_prod_uid48_block_rsrvd_fix_cma_ena2 <= sm0_uid129_prod_uid48_block_rsrvd_fix_cma_ena0;

    sm0_uid129_prod_uid48_block_rsrvd_fix_cma_a0 <= STD_LOGIC_VECTOR(RESIZE(UNSIGNED(rightBottomX_mergedSignalTM_uid116_prod_uid48_block_rsrvd_fix_q),27));
    sm0_uid129_prod_uid48_block_rsrvd_fix_cma_c0 <= STD_LOGIC_VECTOR(RESIZE(UNSIGNED(redist4_aboveLeftY_mergedSignalTM_uid112_prod_uid48_block_rsrvd_fix_q_1_q),27));
    sm0_uid129_prod_uid48_block_rsrvd_fix_cma_DSP0 : tennm_mac
    GENERIC MAP (
        operation_mode => "m27x27",
        clear_type => "none",
        use_chainadder => "false",
        ay_scan_in_clken => "0",
        ay_scan_in_width => 27,
        ax_clken => "0",
        ax_width => 27,
        signed_may => "false",
        signed_max => "false",
        input_pipeline_clken => "2",
        second_pipeline_clken => "2",
        output_clken => "1",
        result_a_width => 54
    )
    PORT MAP (
        clk => clk,
        ena(0) => sm0_uid129_prod_uid48_block_rsrvd_fix_cma_ena0,
        ena(1) => sm0_uid129_prod_uid48_block_rsrvd_fix_cma_ena1,
        ena(2) => sm0_uid129_prod_uid48_block_rsrvd_fix_cma_ena2,
        clr(0) => '0',
        clr(1) => '0',
        ay => sm0_uid129_prod_uid48_block_rsrvd_fix_cma_a0,
        ax => sm0_uid129_prod_uid48_block_rsrvd_fix_cma_c0,
        resulta => sm0_uid129_prod_uid48_block_rsrvd_fix_cma_s0
    );
    sm0_uid129_prod_uid48_block_rsrvd_fix_cma_delay0 : dspba_delay
    GENERIC MAP ( width => 54, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => sm0_uid129_prod_uid48_block_rsrvd_fix_cma_s0, xout => sm0_uid129_prod_uid48_block_rsrvd_fix_cma_qq0, clk => clk, aclr => areset, ena => '1' );
    sm0_uid129_prod_uid48_block_rsrvd_fix_cma_q <= STD_LOGIC_VECTOR(sm0_uid129_prod_uid48_block_rsrvd_fix_cma_qq0(53 downto 0));

    -- add0_uid130_prod_uid48_block_rsrvd_fix(BITJOIN,129)@6
    add0_uid130_prod_uid48_block_rsrvd_fix_q <= topProd_uid108_prod_uid48_block_rsrvd_fix_cma_q & sm0_uid129_prod_uid48_block_rsrvd_fix_cma_q;

    -- lowRangeB_uid131_prod_uid48_block_rsrvd_fix_bit_select_merged(BITSELECT,141)@6
    lowRangeB_uid131_prod_uid48_block_rsrvd_fix_bit_select_merged_b <= STD_LOGIC_VECTOR(add0_uid130_prod_uid48_block_rsrvd_fix_q(26 downto 0));
    lowRangeB_uid131_prod_uid48_block_rsrvd_fix_bit_select_merged_c <= STD_LOGIC_VECTOR(add0_uid130_prod_uid48_block_rsrvd_fix_q(107 downto 27));

    -- add1_uid134_prod_uid48_block_rsrvd_fix(BITJOIN,133)@6
    add1_uid134_prod_uid48_block_rsrvd_fix_q <= add1sumAHighB_uid133_prod_uid48_block_rsrvd_fix_q & lowRangeB_uid131_prod_uid48_block_rsrvd_fix_bit_select_merged_b;

    -- osig_uid135_prod_uid48_block_rsrvd_fix(BITSELECT,134)@6
    osig_uid135_prod_uid48_block_rsrvd_fix_in <= add1_uid134_prod_uid48_block_rsrvd_fix_q(107 downto 0);
    osig_uid135_prod_uid48_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(osig_uid135_prod_uid48_block_rsrvd_fix_in(107 downto 2));

    -- redist3_osig_uid135_prod_uid48_block_rsrvd_fix_b_1(DELAY,145)
    redist3_osig_uid135_prod_uid48_block_rsrvd_fix_b_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist3_osig_uid135_prod_uid48_block_rsrvd_fix_b_1_q <= osig_uid135_prod_uid48_block_rsrvd_fix_b;
            END IF;
        END IF;
    END PROCESS;

    -- normalizeBit_uid50_block_rsrvd_fix(BITSELECT,49)@7
    normalizeBit_uid50_block_rsrvd_fix_b <= redist3_osig_uid135_prod_uid48_block_rsrvd_fix_b_1_q(105 downto 105);

    -- redist9_normalizeBit_uid50_block_rsrvd_fix_b_1(DELAY,151)
    redist9_normalizeBit_uid50_block_rsrvd_fix_b_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist9_normalizeBit_uid50_block_rsrvd_fix_b_1_q <= normalizeBit_uid50_block_rsrvd_fix_b;
            END IF;
        END IF;
    END PROCESS;

    -- roundBitDetectionConstant_uid64_block_rsrvd_fix(CONSTANT,63)
    roundBitDetectionConstant_uid64_block_rsrvd_fix_q <= "010";

    -- fracRPostNormHigh_uid52_block_rsrvd_fix(BITSELECT,51)@7
    fracRPostNormHigh_uid52_block_rsrvd_fix_in <= redist3_osig_uid135_prod_uid48_block_rsrvd_fix_b_1_q(104 downto 0);
    fracRPostNormHigh_uid52_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(fracRPostNormHigh_uid52_block_rsrvd_fix_in(104 downto 52));

    -- fracRPostNormLow_uid53_block_rsrvd_fix(BITSELECT,52)@7
    fracRPostNormLow_uid53_block_rsrvd_fix_in <= redist3_osig_uid135_prod_uid48_block_rsrvd_fix_b_1_q(103 downto 0);
    fracRPostNormLow_uid53_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(fracRPostNormLow_uid53_block_rsrvd_fix_in(103 downto 51));

    -- fracRPostNorm_uid54_block_rsrvd_fix(MUX,53)@7
    fracRPostNorm_uid54_block_rsrvd_fix_s <= normalizeBit_uid50_block_rsrvd_fix_b;
    fracRPostNorm_uid54_block_rsrvd_fix_combproc: PROCESS (fracRPostNorm_uid54_block_rsrvd_fix_s, fracRPostNormLow_uid53_block_rsrvd_fix_b, fracRPostNormHigh_uid52_block_rsrvd_fix_b)
    BEGIN
        CASE (fracRPostNorm_uid54_block_rsrvd_fix_s) IS
            WHEN "0" => fracRPostNorm_uid54_block_rsrvd_fix_q <= fracRPostNormLow_uid53_block_rsrvd_fix_b;
            WHEN "1" => fracRPostNorm_uid54_block_rsrvd_fix_q <= fracRPostNormHigh_uid52_block_rsrvd_fix_b;
            WHEN OTHERS => fracRPostNorm_uid54_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- fracRPostNorm1dto0_uid62_block_rsrvd_fix(BITSELECT,61)@7
    fracRPostNorm1dto0_uid62_block_rsrvd_fix_in <= fracRPostNorm_uid54_block_rsrvd_fix_q(1 downto 0);
    fracRPostNorm1dto0_uid62_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(fracRPostNorm1dto0_uid62_block_rsrvd_fix_in(1 downto 0));

    -- extraStickyBitOfProd_uid56_block_rsrvd_fix(BITSELECT,55)@7
    extraStickyBitOfProd_uid56_block_rsrvd_fix_in <= STD_LOGIC_VECTOR(redist3_osig_uid135_prod_uid48_block_rsrvd_fix_b_1_q(51 downto 0));
    extraStickyBitOfProd_uid56_block_rsrvd_fix_b <= extraStickyBitOfProd_uid56_block_rsrvd_fix_in(51 downto 51);

    -- extraStickyBit_uid57_block_rsrvd_fix(MUX,56)@7
    extraStickyBit_uid57_block_rsrvd_fix_s <= normalizeBit_uid50_block_rsrvd_fix_b;
    extraStickyBit_uid57_block_rsrvd_fix_combproc: PROCESS (extraStickyBit_uid57_block_rsrvd_fix_s, GND_q, extraStickyBitOfProd_uid56_block_rsrvd_fix_b)
    BEGIN
        CASE (extraStickyBit_uid57_block_rsrvd_fix_s) IS
            WHEN "0" => extraStickyBit_uid57_block_rsrvd_fix_q <= GND_q;
            WHEN "1" => extraStickyBit_uid57_block_rsrvd_fix_q <= extraStickyBitOfProd_uid56_block_rsrvd_fix_b;
            WHEN OTHERS => extraStickyBit_uid57_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- stickyRange_uid55_block_rsrvd_fix(BITSELECT,54)@7
    stickyRange_uid55_block_rsrvd_fix_in <= redist3_osig_uid135_prod_uid48_block_rsrvd_fix_b_1_q(50 downto 0);
    stickyRange_uid55_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(stickyRange_uid55_block_rsrvd_fix_in(50 downto 0));

    -- stickyExtendedRange_uid58_block_rsrvd_fix(BITJOIN,57)@7
    stickyExtendedRange_uid58_block_rsrvd_fix_q <= extraStickyBit_uid57_block_rsrvd_fix_q & stickyRange_uid55_block_rsrvd_fix_b;

    -- stickyRangeComparator_uid60_block_rsrvd_fix(LOGICAL,59)@7
    stickyRangeComparator_uid60_block_rsrvd_fix_q <= "1" WHEN stickyExtendedRange_uid58_block_rsrvd_fix_q = cstZeroWF_uid12_block_rsrvd_fix_q ELSE "0";

    -- sticky_uid61_block_rsrvd_fix(LOGICAL,60)@7
    sticky_uid61_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (stickyRangeComparator_uid60_block_rsrvd_fix_q));

    -- lrs_uid63_block_rsrvd_fix(BITJOIN,62)@7
    lrs_uid63_block_rsrvd_fix_q <= fracRPostNorm1dto0_uid62_block_rsrvd_fix_b & sticky_uid61_block_rsrvd_fix_q;

    -- roundBitDetectionPattern_uid65_block_rsrvd_fix(LOGICAL,64)@7 + 1
    roundBitDetectionPattern_uid65_block_rsrvd_fix_qi <= "1" WHEN lrs_uid63_block_rsrvd_fix_q = roundBitDetectionConstant_uid64_block_rsrvd_fix_q ELSE "0";
    roundBitDetectionPattern_uid65_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => roundBitDetectionPattern_uid65_block_rsrvd_fix_qi, xout => roundBitDetectionPattern_uid65_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- roundBit_uid66_block_rsrvd_fix(LOGICAL,65)@8
    roundBit_uid66_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (roundBitDetectionPattern_uid65_block_rsrvd_fix_q));

    -- roundBitAndNormalizationOp_uid69_block_rsrvd_fix(BITJOIN,68)@8
    roundBitAndNormalizationOp_uid69_block_rsrvd_fix_q <= GND_q & redist9_normalizeBit_uid50_block_rsrvd_fix_b_1_q & cstZeroWF_uid12_block_rsrvd_fix_q & roundBit_uid66_block_rsrvd_fix_q;

    -- biasInc_uid46_block_rsrvd_fix(CONSTANT,45)
    biasInc_uid46_block_rsrvd_fix_q <= "0001111111111";

    -- redist11_expSum_uid45_block_rsrvd_fix_q_7_offset(CONSTANT,162)
    redist11_expSum_uid45_block_rsrvd_fix_q_7_offset_q <= "101";

    -- redist11_expSum_uid45_block_rsrvd_fix_q_7_rdcnt(ADD,163)
    redist11_expSum_uid45_block_rsrvd_fix_q_7_rdcnt_a <= STD_LOGIC_VECTOR("0" & redist11_expSum_uid45_block_rsrvd_fix_q_7_wraddr_q);
    redist11_expSum_uid45_block_rsrvd_fix_q_7_rdcnt_b <= STD_LOGIC_VECTOR("0" & redist11_expSum_uid45_block_rsrvd_fix_q_7_offset_q);
    redist11_expSum_uid45_block_rsrvd_fix_q_7_rdcnt_clkproc: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist11_expSum_uid45_block_rsrvd_fix_q_7_rdcnt_o <= STD_LOGIC_VECTOR(UNSIGNED(redist11_expSum_uid45_block_rsrvd_fix_q_7_rdcnt_a) + UNSIGNED(redist11_expSum_uid45_block_rsrvd_fix_q_7_rdcnt_b));
            END IF;
        END IF;
    END PROCESS;
    redist11_expSum_uid45_block_rsrvd_fix_q_7_rdcnt_q <= redist11_expSum_uid45_block_rsrvd_fix_q_7_rdcnt_o(3 downto 0);

    -- expSum_uid45_block_rsrvd_fix(ADD,44)@0 + 1
    expSum_uid45_block_rsrvd_fix_a <= STD_LOGIC_VECTOR("0" & expX_uid7_block_rsrvd_fix_b);
    expSum_uid45_block_rsrvd_fix_b <= STD_LOGIC_VECTOR("0" & expY_uid8_block_rsrvd_fix_b);
    expSum_uid45_block_rsrvd_fix_clkproc: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                expSum_uid45_block_rsrvd_fix_o <= STD_LOGIC_VECTOR(UNSIGNED(expSum_uid45_block_rsrvd_fix_a) + UNSIGNED(expSum_uid45_block_rsrvd_fix_b));
            END IF;
        END IF;
    END PROCESS;
    expSum_uid45_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(expSum_uid45_block_rsrvd_fix_o(11 downto 0));

    -- redist11_expSum_uid45_block_rsrvd_fix_q_7_wraddr(COUNTER,161)
    -- low=0, high=7, step=1, init=0
    redist11_expSum_uid45_block_rsrvd_fix_q_7_wraddr_clkproc: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist11_expSum_uid45_block_rsrvd_fix_q_7_wraddr_i <= redist11_expSum_uid45_block_rsrvd_fix_q_7_wraddr_i + 1;
            END IF;
        END IF;
    END PROCESS;
    redist11_expSum_uid45_block_rsrvd_fix_q_7_wraddr_q <= STD_LOGIC_VECTOR(RESIZE(redist11_expSum_uid45_block_rsrvd_fix_q_7_wraddr_i, 3));

    -- redist11_expSum_uid45_block_rsrvd_fix_q_7_mem(DUALMEM,160)
    redist11_expSum_uid45_block_rsrvd_fix_q_7_mem_ia <= STD_LOGIC_VECTOR(expSum_uid45_block_rsrvd_fix_q);
    redist11_expSum_uid45_block_rsrvd_fix_q_7_mem_aa <= redist11_expSum_uid45_block_rsrvd_fix_q_7_wraddr_q;
    redist11_expSum_uid45_block_rsrvd_fix_q_7_mem_ab <= redist11_expSum_uid45_block_rsrvd_fix_q_7_rdcnt_q(2 downto 0);
    redist11_expSum_uid45_block_rsrvd_fix_q_7_mem_dmem : altera_syncram
    GENERIC MAP (
        ram_block_type => "MLAB",
        operation_mode => "DUAL_PORT",
        width_a => 12,
        widthad_a => 3,
        numwords_a => 8,
        width_b => 12,
        widthad_b => 3,
        numwords_b => 8,
        lpm_type => "altera_syncram",
        width_byteena_a => 1,
        address_reg_b => "CLOCK0",
        indata_reg_b => "CLOCK0",
        rdcontrol_reg_b => "CLOCK0",
        byteena_reg_b => "CLOCK0",
        outdata_reg_b => "CLOCK0",
        outdata_sclr_b => "NONE",
        clock_enable_input_a => "NORMAL",
        clock_enable_input_b => "NORMAL",
        clock_enable_output_b => "NORMAL",
        read_during_write_mode_mixed_ports => "DONT_CARE",
        power_up_uninitialized => "TRUE",
        intended_device_family => "Agilex 7"
    )
    PORT MAP (
        clocken0 => '1',
        clock0 => clk,
        address_a => redist11_expSum_uid45_block_rsrvd_fix_q_7_mem_aa,
        data_a => redist11_expSum_uid45_block_rsrvd_fix_q_7_mem_ia,
        wren_a => VCC_q(0),
        address_b => redist11_expSum_uid45_block_rsrvd_fix_q_7_mem_ab,
        q_b => redist11_expSum_uid45_block_rsrvd_fix_q_7_mem_iq
    );
    redist11_expSum_uid45_block_rsrvd_fix_q_7_mem_q <= STD_LOGIC_VECTOR(redist11_expSum_uid45_block_rsrvd_fix_q_7_mem_iq(11 downto 0));

    -- expSumMBias_uid47_block_rsrvd_fix(SUB,46)@7 + 1
    expSumMBias_uid47_block_rsrvd_fix_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("000" & redist11_expSum_uid45_block_rsrvd_fix_q_7_mem_q));
    expSumMBias_uid47_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((14 downto 13 => biasInc_uid46_block_rsrvd_fix_q(12)) & biasInc_uid46_block_rsrvd_fix_q));
    expSumMBias_uid47_block_rsrvd_fix_clkproc: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                expSumMBias_uid47_block_rsrvd_fix_o <= STD_LOGIC_VECTOR(SIGNED(expSumMBias_uid47_block_rsrvd_fix_a) - SIGNED(expSumMBias_uid47_block_rsrvd_fix_b));
            END IF;
        END IF;
    END PROCESS;
    expSumMBias_uid47_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(expSumMBias_uid47_block_rsrvd_fix_o(13 downto 0));

    -- redist8_fracRPostNorm_uid54_block_rsrvd_fix_q_1(DELAY,150)
    redist8_fracRPostNorm_uid54_block_rsrvd_fix_q_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist8_fracRPostNorm_uid54_block_rsrvd_fix_q_1_q <= fracRPostNorm_uid54_block_rsrvd_fix_q;
            END IF;
        END IF;
    END PROCESS;

    -- expFracPreRound_uid67_block_rsrvd_fix(BITJOIN,66)@8
    expFracPreRound_uid67_block_rsrvd_fix_q <= expSumMBias_uid47_block_rsrvd_fix_q & redist8_fracRPostNorm_uid54_block_rsrvd_fix_q_1_q;

    -- expFracRPostRounding_uid70_block_rsrvd_fix(ADD,69)@8
    expFracRPostRounding_uid70_block_rsrvd_fix_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((68 downto 67 => expFracPreRound_uid67_block_rsrvd_fix_q(66)) & expFracPreRound_uid67_block_rsrvd_fix_q));
    expFracRPostRounding_uid70_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("00000000000000" & roundBitAndNormalizationOp_uid69_block_rsrvd_fix_q));
    expFracRPostRounding_uid70_block_rsrvd_fix_o <= STD_LOGIC_VECTOR(SIGNED(expFracRPostRounding_uid70_block_rsrvd_fix_a) + SIGNED(expFracRPostRounding_uid70_block_rsrvd_fix_b));
    expFracRPostRounding_uid70_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(expFracRPostRounding_uid70_block_rsrvd_fix_o(67 downto 0));

    -- expRPreExcExt_uid72_block_rsrvd_fix(BITSELECT,71)@8
    expRPreExcExt_uid72_block_rsrvd_fix_b <= expFracRPostRounding_uid70_block_rsrvd_fix_q(67 downto 53);

    -- redist6_expRPreExcExt_uid72_block_rsrvd_fix_b_1(DELAY,148)
    redist6_expRPreExcExt_uid72_block_rsrvd_fix_b_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist6_expRPreExcExt_uid72_block_rsrvd_fix_b_1_q <= expRPreExcExt_uid72_block_rsrvd_fix_b;
            END IF;
        END IF;
    END PROCESS;

    -- expRPreExc_uid73_block_rsrvd_fix(BITSELECT,72)@9
    expRPreExc_uid73_block_rsrvd_fix_in <= redist6_expRPreExcExt_uid72_block_rsrvd_fix_b_1_q(10 downto 0);
    expRPreExc_uid73_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(expRPreExc_uid73_block_rsrvd_fix_in(10 downto 0));

    -- redist5_expRPreExc_uid73_block_rsrvd_fix_b_1(DELAY,147)
    redist5_expRPreExc_uid73_block_rsrvd_fix_b_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist5_expRPreExc_uid73_block_rsrvd_fix_b_1_q <= expRPreExc_uid73_block_rsrvd_fix_b;
            END IF;
        END IF;
    END PROCESS;

    -- expOvf_uid76_block_rsrvd_fix(COMPARE,75)@9
    expOvf_uid76_block_rsrvd_fix_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((16 downto 15 => redist6_expRPreExcExt_uid72_block_rsrvd_fix_b_1_q(14)) & redist6_expRPreExcExt_uid72_block_rsrvd_fix_b_1_q));
    expOvf_uid76_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("000000" & cstAllOWE_uid11_block_rsrvd_fix_q));
    expOvf_uid76_block_rsrvd_fix_o <= STD_LOGIC_VECTOR(SIGNED(expOvf_uid76_block_rsrvd_fix_a) - SIGNED(expOvf_uid76_block_rsrvd_fix_b));
    expOvf_uid76_block_rsrvd_fix_n(0) <= not (expOvf_uid76_block_rsrvd_fix_o(16));

    -- invExpXIsMax_uid36_block_rsrvd_fix(LOGICAL,35)@9
    invExpXIsMax_uid36_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (redist13_expXIsMax_uid31_block_rsrvd_fix_q_9_q));

    -- InvExpXIsZero_uid37_block_rsrvd_fix(LOGICAL,36)@9
    InvExpXIsZero_uid37_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (redist14_excZ_y_uid30_block_rsrvd_fix_q_9_q));

    -- excR_y_uid38_block_rsrvd_fix(LOGICAL,37)@9
    excR_y_uid38_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(InvExpXIsZero_uid37_block_rsrvd_fix_q and invExpXIsMax_uid36_block_rsrvd_fix_q);

    -- invExpXIsMax_uid22_block_rsrvd_fix(LOGICAL,21)@9
    invExpXIsMax_uid22_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (redist16_expXIsMax_uid17_block_rsrvd_fix_q_9_q));

    -- InvExpXIsZero_uid23_block_rsrvd_fix(LOGICAL,22)@9
    InvExpXIsZero_uid23_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (redist17_excZ_x_uid16_block_rsrvd_fix_q_9_q));

    -- excR_x_uid24_block_rsrvd_fix(LOGICAL,23)@9
    excR_x_uid24_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(InvExpXIsZero_uid23_block_rsrvd_fix_q and invExpXIsMax_uid22_block_rsrvd_fix_q);

    -- ExcROvfAndInReg_uid85_block_rsrvd_fix(LOGICAL,84)@9
    ExcROvfAndInReg_uid85_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(excR_x_uid24_block_rsrvd_fix_q and excR_y_uid38_block_rsrvd_fix_q and expOvf_uid76_block_rsrvd_fix_n);

    -- excYRAndExcXI_uid84_block_rsrvd_fix(LOGICAL,83)@9
    excYRAndExcXI_uid84_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(excR_y_uid38_block_rsrvd_fix_q and excI_x_uid20_block_rsrvd_fix_q);

    -- excXRAndExcYI_uid83_block_rsrvd_fix(LOGICAL,82)@9
    excXRAndExcYI_uid83_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(excR_x_uid24_block_rsrvd_fix_q and excI_y_uid34_block_rsrvd_fix_q);

    -- excXIAndExcYI_uid82_block_rsrvd_fix(LOGICAL,81)@9
    excXIAndExcYI_uid82_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(excI_x_uid20_block_rsrvd_fix_q and excI_y_uid34_block_rsrvd_fix_q);

    -- excRInf_uid86_block_rsrvd_fix(LOGICAL,85)@9
    excRInf_uid86_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(excXIAndExcYI_uid82_block_rsrvd_fix_q or excXRAndExcYI_uid83_block_rsrvd_fix_q or excYRAndExcXI_uid84_block_rsrvd_fix_q or ExcROvfAndInReg_uid85_block_rsrvd_fix_q);

    -- expUdf_uid74_block_rsrvd_fix(COMPARE,73)@9
    expUdf_uid74_block_rsrvd_fix_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("0000000000000000" & GND_q));
    expUdf_uid74_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((16 downto 15 => redist6_expRPreExcExt_uid72_block_rsrvd_fix_b_1_q(14)) & redist6_expRPreExcExt_uid72_block_rsrvd_fix_b_1_q));
    expUdf_uid74_block_rsrvd_fix_o <= STD_LOGIC_VECTOR(SIGNED(expUdf_uid74_block_rsrvd_fix_a) - SIGNED(expUdf_uid74_block_rsrvd_fix_b));
    expUdf_uid74_block_rsrvd_fix_n(0) <= not (expUdf_uid74_block_rsrvd_fix_o(16));

    -- excZC3_uid80_block_rsrvd_fix(LOGICAL,79)@9
    excZC3_uid80_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(excR_x_uid24_block_rsrvd_fix_q and excR_y_uid38_block_rsrvd_fix_q and expUdf_uid74_block_rsrvd_fix_n);

    -- excYZAndExcXR_uid79_block_rsrvd_fix(LOGICAL,78)@9
    excYZAndExcXR_uid79_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(redist14_excZ_y_uid30_block_rsrvd_fix_q_9_q and excR_x_uid24_block_rsrvd_fix_q);

    -- excXZAndExcYR_uid78_block_rsrvd_fix(LOGICAL,77)@9
    excXZAndExcYR_uid78_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(redist17_excZ_x_uid16_block_rsrvd_fix_q_9_q and excR_y_uid38_block_rsrvd_fix_q);

    -- excXZAndExcYZ_uid77_block_rsrvd_fix(LOGICAL,76)@9
    excXZAndExcYZ_uid77_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(redist17_excZ_x_uid16_block_rsrvd_fix_q_9_q and redist14_excZ_y_uid30_block_rsrvd_fix_q_9_q);

    -- excRZero_uid81_block_rsrvd_fix(LOGICAL,80)@9
    excRZero_uid81_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(excXZAndExcYZ_uid77_block_rsrvd_fix_q or excXZAndExcYR_uid78_block_rsrvd_fix_q or excYZAndExcXR_uid79_block_rsrvd_fix_q or excZC3_uid80_block_rsrvd_fix_q);

    -- concExc_uid91_block_rsrvd_fix(BITJOIN,90)@9
    concExc_uid91_block_rsrvd_fix_q <= excRNaN_uid90_block_rsrvd_fix_q & excRInf_uid86_block_rsrvd_fix_q & excRZero_uid81_block_rsrvd_fix_q;

    -- excREnc_uid92_block_rsrvd_fix(LOOKUP,91)@9 + 1
    excREnc_uid92_block_rsrvd_fix_clkproc: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                CASE (concExc_uid91_block_rsrvd_fix_q) IS
                    WHEN "000" => excREnc_uid92_block_rsrvd_fix_q <= "01";
                    WHEN "001" => excREnc_uid92_block_rsrvd_fix_q <= "00";
                    WHEN "010" => excREnc_uid92_block_rsrvd_fix_q <= "10";
                    WHEN "011" => excREnc_uid92_block_rsrvd_fix_q <= "00";
                    WHEN "100" => excREnc_uid92_block_rsrvd_fix_q <= "11";
                    WHEN "101" => excREnc_uid92_block_rsrvd_fix_q <= "00";
                    WHEN "110" => excREnc_uid92_block_rsrvd_fix_q <= "00";
                    WHEN "111" => excREnc_uid92_block_rsrvd_fix_q <= "00";
                    WHEN OTHERS => -- unreachable
                                   excREnc_uid92_block_rsrvd_fix_q <= (others => '-');
                END CASE;
            END IF;
        END IF;
    END PROCESS;

    -- expRPostExc_uid101_block_rsrvd_fix(MUX,100)@10
    expRPostExc_uid101_block_rsrvd_fix_s <= excREnc_uid92_block_rsrvd_fix_q;
    expRPostExc_uid101_block_rsrvd_fix_combproc: PROCESS (expRPostExc_uid101_block_rsrvd_fix_s, cstAllZWE_uid13_block_rsrvd_fix_q, redist5_expRPreExc_uid73_block_rsrvd_fix_b_1_q, cstAllOWE_uid11_block_rsrvd_fix_q)
    BEGIN
        CASE (expRPostExc_uid101_block_rsrvd_fix_s) IS
            WHEN "00" => expRPostExc_uid101_block_rsrvd_fix_q <= cstAllZWE_uid13_block_rsrvd_fix_q;
            WHEN "01" => expRPostExc_uid101_block_rsrvd_fix_q <= redist5_expRPreExc_uid73_block_rsrvd_fix_b_1_q;
            WHEN "10" => expRPostExc_uid101_block_rsrvd_fix_q <= cstAllOWE_uid11_block_rsrvd_fix_q;
            WHEN "11" => expRPostExc_uid101_block_rsrvd_fix_q <= cstAllOWE_uid11_block_rsrvd_fix_q;
            WHEN OTHERS => expRPostExc_uid101_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- oneFracRPostExc2_uid93_block_rsrvd_fix(CONSTANT,92)
    oneFracRPostExc2_uid93_block_rsrvd_fix_q <= "0000000000000000000000000000000000000000000000000001";

    -- fracRPreExc_uid71_block_rsrvd_fix(BITSELECT,70)@8
    fracRPreExc_uid71_block_rsrvd_fix_in <= expFracRPostRounding_uid70_block_rsrvd_fix_q(52 downto 0);
    fracRPreExc_uid71_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(fracRPreExc_uid71_block_rsrvd_fix_in(52 downto 1));

    -- redist7_fracRPreExc_uid71_block_rsrvd_fix_b_2(DELAY,149)
    redist7_fracRPreExc_uid71_block_rsrvd_fix_b_2_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist7_fracRPreExc_uid71_block_rsrvd_fix_b_2_delay_0 <= STD_LOGIC_VECTOR(fracRPreExc_uid71_block_rsrvd_fix_b);
                redist7_fracRPreExc_uid71_block_rsrvd_fix_b_2_q <= STD_LOGIC_VECTOR(redist7_fracRPreExc_uid71_block_rsrvd_fix_b_2_delay_0);
            END IF;
        END IF;
    END PROCESS;

    -- fracRPostExc_uid96_block_rsrvd_fix(MUX,95)@10
    fracRPostExc_uid96_block_rsrvd_fix_s <= excREnc_uid92_block_rsrvd_fix_q;
    fracRPostExc_uid96_block_rsrvd_fix_combproc: PROCESS (fracRPostExc_uid96_block_rsrvd_fix_s, cstZeroWF_uid12_block_rsrvd_fix_q, redist7_fracRPreExc_uid71_block_rsrvd_fix_b_2_q, oneFracRPostExc2_uid93_block_rsrvd_fix_q)
    BEGIN
        CASE (fracRPostExc_uid96_block_rsrvd_fix_s) IS
            WHEN "00" => fracRPostExc_uid96_block_rsrvd_fix_q <= cstZeroWF_uid12_block_rsrvd_fix_q;
            WHEN "01" => fracRPostExc_uid96_block_rsrvd_fix_q <= redist7_fracRPreExc_uid71_block_rsrvd_fix_b_2_q;
            WHEN "10" => fracRPostExc_uid96_block_rsrvd_fix_q <= cstZeroWF_uid12_block_rsrvd_fix_q;
            WHEN "11" => fracRPostExc_uid96_block_rsrvd_fix_q <= oneFracRPostExc2_uid93_block_rsrvd_fix_q;
            WHEN OTHERS => fracRPostExc_uid96_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- R_uid104_block_rsrvd_fix(BITJOIN,103)@10
    R_uid104_block_rsrvd_fix_q <= signRPostExc_uid103_block_rsrvd_fix_q & expRPostExc_uid101_block_rsrvd_fix_q & fracRPostExc_uid96_block_rsrvd_fix_q;

    -- out_primWireOut(GPOUT,5)@10
    out_primWireOut <= R_uid104_block_rsrvd_fix_q;

END normal;
