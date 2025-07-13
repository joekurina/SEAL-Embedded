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

-- VHDL created from flt_fft_example_DUT_addSubFusedBlock_typeSFloatIEEE_52_11_4_correctRounding_3856q5c35ig1uu67v6d88db6063061663c61i601c3d60j63ji5j63260uq5ux0ao30cd06cj6ok0dpzc
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
entity flt_fft_example_DUT_addSubFusedBlock_typeSFloatIEEE_52_11_4_correctRounding_38560000x0ao30cd06cj6ok0dpzc is
    port (
        in_0 : in std_logic_vector(63 downto 0);  -- float64_m52
        in_1 : in std_logic_vector(63 downto 0);  -- float64_m52
        out_primWireAux : out std_logic_vector(63 downto 0);  -- float64_m52
        out_primWireOut : out std_logic_vector(63 downto 0);  -- float64_m52
        clk : in std_logic;
        areset : in std_logic
    );
end flt_fft_example_DUT_addSubFusedBlock_typeSFloatIEEE_52_11_4_correctRounding_38560000x0ao30cd06cj6ok0dpzc;

architecture normal of flt_fft_example_DUT_addSubFusedBlock_typeSFloatIEEE_52_11_4_correctRounding_38560000x0ao30cd06cj6ok0dpzc is

    attribute altera_attribute : string;
    attribute altera_attribute of normal : architecture is "-name AUTO_SHIFT_REGISTER_RECOGNITION OFF; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 10037; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 15400; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 12020; -name MESSAGE_DISABLE 12030; -name MESSAGE_DISABLE 12010; -name MESSAGE_DISABLE 12110; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 13410; -name MESSAGE_DISABLE 113007";
    
    signal GND_q : STD_LOGIC_VECTOR (0 downto 0);
    signal VCC_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expFracX_uid8_block_rsrvd_fix_b : STD_LOGIC_VECTOR (62 downto 0);
    signal expFracY_uid9_block_rsrvd_fix_b : STD_LOGIC_VECTOR (62 downto 0);
    signal xGTEy_uid10_block_rsrvd_fix_a : STD_LOGIC_VECTOR (64 downto 0);
    signal xGTEy_uid10_block_rsrvd_fix_b : STD_LOGIC_VECTOR (64 downto 0);
    signal xGTEy_uid10_block_rsrvd_fix_o : STD_LOGIC_VECTOR (64 downto 0);
    signal xGTEy_uid10_block_rsrvd_fix_n : STD_LOGIC_VECTOR (0 downto 0);
    signal swap_uid11_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal swap_uid11_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal siga_uid12_block_rsrvd_fix_s : STD_LOGIC_VECTOR (0 downto 0);
    signal siga_uid12_block_rsrvd_fix_q : STD_LOGIC_VECTOR (63 downto 0);
    signal sigb_uid13_block_rsrvd_fix_s : STD_LOGIC_VECTOR (0 downto 0);
    signal sigb_uid13_block_rsrvd_fix_q : STD_LOGIC_VECTOR (63 downto 0);
    signal cstAllOWE_uid14_block_rsrvd_fix_q : STD_LOGIC_VECTOR (10 downto 0);
    signal cstZeroWF_uid15_block_rsrvd_fix_q : STD_LOGIC_VECTOR (51 downto 0);
    signal cstAllZWE_uid16_block_rsrvd_fix_q : STD_LOGIC_VECTOR (10 downto 0);
    signal exp_siga_uid17_block_rsrvd_fix_in : STD_LOGIC_VECTOR (62 downto 0);
    signal exp_siga_uid17_block_rsrvd_fix_b : STD_LOGIC_VECTOR (10 downto 0);
    signal frac_siga_uid18_block_rsrvd_fix_in : STD_LOGIC_VECTOR (51 downto 0);
    signal frac_siga_uid18_block_rsrvd_fix_b : STD_LOGIC_VECTOR (51 downto 0);
    signal excZ_siga_uid12_uid19_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expXIsMax_uid20_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal fracXIsZero_uid21_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal fracXIsZero_uid21_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal fracXIsNotZero_uid22_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excI_siga_uid23_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excI_siga_uid23_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excN_siga_uid24_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal invExpXIsMax_uid25_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal InvExpXIsZero_uid26_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excR_siga_uid27_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal exp_sigb_uid31_block_rsrvd_fix_in : STD_LOGIC_VECTOR (62 downto 0);
    signal exp_sigb_uid31_block_rsrvd_fix_b : STD_LOGIC_VECTOR (10 downto 0);
    signal frac_sigb_uid32_block_rsrvd_fix_in : STD_LOGIC_VECTOR (51 downto 0);
    signal frac_sigb_uid32_block_rsrvd_fix_b : STD_LOGIC_VECTOR (51 downto 0);
    signal excZ_sigb_uid13_uid33_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excZ_sigb_uid13_uid33_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expXIsMax_uid34_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal expXIsMax_uid34_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal fracXIsZero_uid35_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal fracXIsZero_uid35_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal fracXIsNotZero_uid36_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excI_sigb_uid37_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal excI_sigb_uid37_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excN_sigb_uid38_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal invExpXIsMax_uid39_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal InvExpXIsZero_uid40_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excR_sigb_uid41_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal sigA_uid46_block_rsrvd_fix_b : STD_LOGIC_VECTOR (0 downto 0);
    signal sigB_uid47_block_rsrvd_fix_b : STD_LOGIC_VECTOR (0 downto 0);
    signal effSub_uid48_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expAmExpB_uid51_block_rsrvd_fix_a : STD_LOGIC_VECTOR (11 downto 0);
    signal expAmExpB_uid51_block_rsrvd_fix_b : STD_LOGIC_VECTOR (11 downto 0);
    signal expAmExpB_uid51_block_rsrvd_fix_o : STD_LOGIC_VECTOR (11 downto 0);
    signal expAmExpB_uid51_block_rsrvd_fix_q : STD_LOGIC_VECTOR (11 downto 0);
    signal shiftOutConst_uid52_block_rsrvd_fix_q : STD_LOGIC_VECTOR (5 downto 0);
    signal alignShiftMaxM1_uid53_block_rsrvd_fix_q : STD_LOGIC_VECTOR (5 downto 0);
    signal expBIsZero_uid55_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal expBIsZero_uid55_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expBIsZero_uid56_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal shiftedOut1_uid57_block_rsrvd_fix_a : STD_LOGIC_VECTOR (13 downto 0);
    signal shiftedOut1_uid57_block_rsrvd_fix_b : STD_LOGIC_VECTOR (13 downto 0);
    signal shiftedOut1_uid57_block_rsrvd_fix_o : STD_LOGIC_VECTOR (13 downto 0);
    signal shiftedOut1_uid57_block_rsrvd_fix_c : STD_LOGIC_VECTOR (0 downto 0);
    signal shiftedOut_uid58_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expAmExpBShiftRange_uid59_block_rsrvd_fix_in : STD_LOGIC_VECTOR (5 downto 0);
    signal expAmExpBShiftRange_uid59_block_rsrvd_fix_b : STD_LOGIC_VECTOR (5 downto 0);
    signal shiftValue_uid60_block_rsrvd_fix_s : STD_LOGIC_VECTOR (0 downto 0);
    signal shiftValue_uid60_block_rsrvd_fix_q : STD_LOGIC_VECTOR (5 downto 0);
    signal oFracB_uid62_block_rsrvd_fix_q : STD_LOGIC_VECTOR (52 downto 0);
    signal oFracA_uid63_block_rsrvd_fix_q : STD_LOGIC_VECTOR (52 downto 0);
    signal padConst_uid65_block_rsrvd_fix_q : STD_LOGIC_VECTOR (54 downto 0);
    signal rightPaddedIn_uid66_block_rsrvd_fix_q : STD_LOGIC_VECTOR (107 downto 0);
    signal cmpStickyWZero_uid70_block_rsrvd_fix_b : STD_LOGIC_VECTOR (52 downto 0);
    signal cmpStickyWZero_uid70_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal cmpStickyWZero_uid70_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal sticky_uid71_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal alignFracB_uid73_block_rsrvd_fix_q : STD_LOGIC_VECTOR (55 downto 0);
    signal zv_uid74_block_rsrvd_fix_q : STD_LOGIC_VECTOR (2 downto 0);
    signal fracAOp_uid75_block_rsrvd_fix_q : STD_LOGIC_VECTOR (55 downto 0);
    signal fracBOp_uid76_block_rsrvd_fix_q : STD_LOGIC_VECTOR (57 downto 0);
    signal fracResSub_uid78_block_rsrvd_fix_a : STD_LOGIC_VECTOR (58 downto 0);
    signal fracResSub_uid78_block_rsrvd_fix_b : STD_LOGIC_VECTOR (58 downto 0);
    signal fracResSub_uid78_block_rsrvd_fix_o : STD_LOGIC_VECTOR (58 downto 0);
    signal fracResSub_uid78_block_rsrvd_fix_q : STD_LOGIC_VECTOR (58 downto 0);
    signal fracResAddNoSignExt_uid79_block_rsrvd_fix_in : STD_LOGIC_VECTOR (56 downto 0);
    signal fracResAddNoSignExt_uid79_block_rsrvd_fix_b : STD_LOGIC_VECTOR (56 downto 0);
    signal fracResSubNoSignExt_uid80_block_rsrvd_fix_in : STD_LOGIC_VECTOR (56 downto 0);
    signal fracResSubNoSignExt_uid80_block_rsrvd_fix_b : STD_LOGIC_VECTOR (56 downto 0);
    signal cAmA_uid85_block_rsrvd_fix_q : STD_LOGIC_VECTOR (5 downto 0);
    signal aMinusA_uid86_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal aMinusA_uid86_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expInc_uid87_block_rsrvd_fix_a : STD_LOGIC_VECTOR (11 downto 0);
    signal expInc_uid87_block_rsrvd_fix_b : STD_LOGIC_VECTOR (11 downto 0);
    signal expInc_uid87_block_rsrvd_fix_o : STD_LOGIC_VECTOR (11 downto 0);
    signal expInc_uid87_block_rsrvd_fix_q : STD_LOGIC_VECTOR (11 downto 0);
    signal expPostNormSub_uid88_block_rsrvd_fix_a : STD_LOGIC_VECTOR (12 downto 0);
    signal expPostNormSub_uid88_block_rsrvd_fix_b : STD_LOGIC_VECTOR (12 downto 0);
    signal expPostNormSub_uid88_block_rsrvd_fix_o : STD_LOGIC_VECTOR (12 downto 0);
    signal expPostNormSub_uid88_block_rsrvd_fix_q : STD_LOGIC_VECTOR (12 downto 0);
    signal expPostNormAdd_uid89_block_rsrvd_fix_a : STD_LOGIC_VECTOR (12 downto 0);
    signal expPostNormAdd_uid89_block_rsrvd_fix_b : STD_LOGIC_VECTOR (12 downto 0);
    signal expPostNormAdd_uid89_block_rsrvd_fix_o : STD_LOGIC_VECTOR (12 downto 0);
    signal expPostNormAdd_uid89_block_rsrvd_fix_q : STD_LOGIC_VECTOR (12 downto 0);
    signal fracPostNormSubRndRange_uid90_block_rsrvd_fix_in : STD_LOGIC_VECTOR (55 downto 0);
    signal fracPostNormSubRndRange_uid90_block_rsrvd_fix_b : STD_LOGIC_VECTOR (52 downto 0);
    signal expFracRSub_uid91_block_rsrvd_fix_q : STD_LOGIC_VECTOR (65 downto 0);
    signal fracPostNormAddRndRange_uid92_block_rsrvd_fix_in : STD_LOGIC_VECTOR (55 downto 0);
    signal fracPostNormAddRndRange_uid92_block_rsrvd_fix_b : STD_LOGIC_VECTOR (52 downto 0);
    signal expFracRAdd_uid93_block_rsrvd_fix_q : STD_LOGIC_VECTOR (65 downto 0);
    signal sticky0_add_uid94_block_rsrvd_fix_in : STD_LOGIC_VECTOR (0 downto 0);
    signal sticky0_add_uid94_block_rsrvd_fix_b : STD_LOGIC_VECTOR (0 downto 0);
    signal sticky1_add_uid95_block_rsrvd_fix_in : STD_LOGIC_VECTOR (1 downto 0);
    signal sticky1_add_uid95_block_rsrvd_fix_b : STD_LOGIC_VECTOR (0 downto 0);
    signal Round_add_uid96_block_rsrvd_fix_in : STD_LOGIC_VECTOR (2 downto 0);
    signal Round_add_uid96_block_rsrvd_fix_b : STD_LOGIC_VECTOR (0 downto 0);
    signal Guard_add_uid97_block_rsrvd_fix_in : STD_LOGIC_VECTOR (3 downto 0);
    signal Guard_add_uid97_block_rsrvd_fix_b : STD_LOGIC_VECTOR (0 downto 0);
    signal LSB_add_uid98_block_rsrvd_fix_in : STD_LOGIC_VECTOR (4 downto 0);
    signal LSB_add_uid98_block_rsrvd_fix_b : STD_LOGIC_VECTOR (0 downto 0);
    signal rndBitCond_add_uid99_block_rsrvd_fix_q : STD_LOGIC_VECTOR (4 downto 0);
    signal cRBit_uid100_block_rsrvd_fix_q : STD_LOGIC_VECTOR (4 downto 0);
    signal rBi_add_uid101_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal rBi_add_uid101_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal roundBit_add_uid102_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expFracRAddPostRound_uid103_block_rsrvd_fix_a : STD_LOGIC_VECTOR (66 downto 0);
    signal expFracRAddPostRound_uid103_block_rsrvd_fix_b : STD_LOGIC_VECTOR (66 downto 0);
    signal expFracRAddPostRound_uid103_block_rsrvd_fix_o : STD_LOGIC_VECTOR (66 downto 0);
    signal expFracRAddPostRound_uid103_block_rsrvd_fix_q : STD_LOGIC_VECTOR (66 downto 0);
    signal Sticky0_sub_uid104_block_rsrvd_fix_in : STD_LOGIC_VECTOR (0 downto 0);
    signal Sticky0_sub_uid104_block_rsrvd_fix_b : STD_LOGIC_VECTOR (0 downto 0);
    signal Sticky1_sub_uid105_block_rsrvd_fix_in : STD_LOGIC_VECTOR (1 downto 0);
    signal Sticky1_sub_uid105_block_rsrvd_fix_b : STD_LOGIC_VECTOR (0 downto 0);
    signal Round_sub_uid106_block_rsrvd_fix_in : STD_LOGIC_VECTOR (2 downto 0);
    signal Round_sub_uid106_block_rsrvd_fix_b : STD_LOGIC_VECTOR (0 downto 0);
    signal Guard_sub_uid107_block_rsrvd_fix_in : STD_LOGIC_VECTOR (3 downto 0);
    signal Guard_sub_uid107_block_rsrvd_fix_b : STD_LOGIC_VECTOR (0 downto 0);
    signal LSB_sub_uid108_block_rsrvd_fix_in : STD_LOGIC_VECTOR (4 downto 0);
    signal LSB_sub_uid108_block_rsrvd_fix_b : STD_LOGIC_VECTOR (0 downto 0);
    signal rndBitCond_sub_uid109_block_rsrvd_fix_q : STD_LOGIC_VECTOR (4 downto 0);
    signal rBi_sub_uid110_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal rBi_sub_uid110_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal roundBit_sub_uid111_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal expFracRSubPostRound_uid112_block_rsrvd_fix_a : STD_LOGIC_VECTOR (66 downto 0);
    signal expFracRSubPostRound_uid112_block_rsrvd_fix_b : STD_LOGIC_VECTOR (66 downto 0);
    signal expFracRSubPostRound_uid112_block_rsrvd_fix_o : STD_LOGIC_VECTOR (66 downto 0);
    signal expFracRSubPostRound_uid112_block_rsrvd_fix_q : STD_LOGIC_VECTOR (66 downto 0);
    signal wEP2AllOwE_uid113_block_rsrvd_fix_q : STD_LOGIC_VECTOR (12 downto 0);
    signal rndExp_uid114_block_rsrvd_fix_in : STD_LOGIC_VECTOR (65 downto 0);
    signal rndExp_uid114_block_rsrvd_fix_b : STD_LOGIC_VECTOR (12 downto 0);
    signal rOvf_uid115_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal rOvf_uid115_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal signedExp_uid116_block_rsrvd_fix_in : STD_LOGIC_VECTOR (65 downto 0);
    signal signedExp_uid116_block_rsrvd_fix_b : STD_LOGIC_VECTOR (12 downto 0);
    signal rUdf_uid117_block_rsrvd_fix_a : STD_LOGIC_VECTOR (14 downto 0);
    signal rUdf_uid117_block_rsrvd_fix_b : STD_LOGIC_VECTOR (14 downto 0);
    signal rUdf_uid117_block_rsrvd_fix_o : STD_LOGIC_VECTOR (14 downto 0);
    signal rUdf_uid117_block_rsrvd_fix_n : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPreExcAdd_uid118_block_rsrvd_fix_in : STD_LOGIC_VECTOR (52 downto 0);
    signal fracRPreExcAdd_uid118_block_rsrvd_fix_b : STD_LOGIC_VECTOR (51 downto 0);
    signal expRPreExcAdd_uid119_block_rsrvd_fix_in : STD_LOGIC_VECTOR (63 downto 0);
    signal expRPreExcAdd_uid119_block_rsrvd_fix_b : STD_LOGIC_VECTOR (10 downto 0);
    signal fracRPreExcSub_uid121_block_rsrvd_fix_in : STD_LOGIC_VECTOR (52 downto 0);
    signal fracRPreExcSub_uid121_block_rsrvd_fix_b : STD_LOGIC_VECTOR (51 downto 0);
    signal expRPreExcSub_uid122_block_rsrvd_fix_in : STD_LOGIC_VECTOR (63 downto 0);
    signal expRPreExcSub_uid122_block_rsrvd_fix_b : STD_LOGIC_VECTOR (10 downto 0);
    signal regInputs_uid124_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal regInputs_uid124_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excRZeroVInC_uid125_block_rsrvd_fix_q : STD_LOGIC_VECTOR (5 downto 0);
    signal excRZeroAdd_uid126_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excRZeroSub_uid127_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal oneIsNaN_uid128_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal oneIsNaN_uid128_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excRInfVInC_uid129_block_rsrvd_fix_q : STD_LOGIC_VECTOR (5 downto 0);
    signal excRInfAdd_uid130_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excRInfSub_uid131_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal infMinf_uid132_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excRNaNA_uid133_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal invEffSub_uid134_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal infPinfForSub_uid135_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal excRNaNS_uid136_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPreExcAddition_uid137_block_rsrvd_fix_s : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPreExcAddition_uid137_block_rsrvd_fix_q : STD_LOGIC_VECTOR (51 downto 0);
    signal expRPreExcAddition_uid138_block_rsrvd_fix_s : STD_LOGIC_VECTOR (0 downto 0);
    signal expRPreExcAddition_uid138_block_rsrvd_fix_q : STD_LOGIC_VECTOR (10 downto 0);
    signal fracRPreExcSubtraction_uid139_block_rsrvd_fix_s : STD_LOGIC_VECTOR (0 downto 0);
    signal fracRPreExcSubtraction_uid139_block_rsrvd_fix_q : STD_LOGIC_VECTOR (51 downto 0);
    signal expRPreExcSubtraction_uid140_block_rsrvd_fix_s : STD_LOGIC_VECTOR (0 downto 0);
    signal expRPreExcSubtraction_uid140_block_rsrvd_fix_q : STD_LOGIC_VECTOR (10 downto 0);
    signal concExcSub_uid141_block_rsrvd_fix_q : STD_LOGIC_VECTOR (2 downto 0);
    signal concExcAdd_uid142_block_rsrvd_fix_q : STD_LOGIC_VECTOR (2 downto 0);
    signal excREncSub_uid143_block_rsrvd_fix_q : STD_LOGIC_VECTOR (1 downto 0);
    signal excREncAdd_uid144_block_rsrvd_fix_q : STD_LOGIC_VECTOR (1 downto 0);
    signal oneFracRPostExc2_uid145_block_rsrvd_fix_q : STD_LOGIC_VECTOR (51 downto 0);
    signal fracRPostExcAdd_uid148_block_rsrvd_fix_s : STD_LOGIC_VECTOR (1 downto 0);
    signal fracRPostExcAdd_uid148_block_rsrvd_fix_q : STD_LOGIC_VECTOR (51 downto 0);
    signal expRPostExcAdd_uid152_block_rsrvd_fix_s : STD_LOGIC_VECTOR (1 downto 0);
    signal expRPostExcAdd_uid152_block_rsrvd_fix_q : STD_LOGIC_VECTOR (10 downto 0);
    signal zMz_uid153_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal invZMZ_uid154_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal aMa_uid155_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal invAMA_uid156_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal invExcRNaNA_uid157_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal signRPostExcAdd_uid158_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal signRPostExcAdd_uid158_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal RSum_uid159_block_rsrvd_fix_q : STD_LOGIC_VECTOR (63 downto 0);
    signal fracRPostExcSub_uid163_block_rsrvd_fix_s : STD_LOGIC_VECTOR (1 downto 0);
    signal fracRPostExcSub_uid163_block_rsrvd_fix_q : STD_LOGIC_VECTOR (51 downto 0);
    signal expRPostExcSub_uid167_block_rsrvd_fix_s : STD_LOGIC_VECTOR (1 downto 0);
    signal expRPostExcSub_uid167_block_rsrvd_fix_q : STD_LOGIC_VECTOR (10 downto 0);
    signal zMz_uid169_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal invZMZSub_uid170_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal aMa_uid171_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal invAMASub_uid172_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal signASwap_uid173_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal invExcRNaNS_uid174_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal signRPostExcSub_uid175_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal signRPostExcSub_uid175_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal RDiff_uid176_block_rsrvd_fix_q : STD_LOGIC_VECTOR (63 downto 0);
    signal zs_uid179_lzCountValAdd_uid81_block_rsrvd_fix_q : STD_LOGIC_VECTOR (31 downto 0);
    signal rVStage_uid180_lzCountValAdd_uid81_block_rsrvd_fix_b : STD_LOGIC_VECTOR (31 downto 0);
    signal vCount_uid181_lzCountValAdd_uid81_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal mO_uid182_lzCountValAdd_uid81_block_rsrvd_fix_q : STD_LOGIC_VECTOR (6 downto 0);
    signal vStage_uid183_lzCountValAdd_uid81_block_rsrvd_fix_in : STD_LOGIC_VECTOR (24 downto 0);
    signal vStage_uid183_lzCountValAdd_uid81_block_rsrvd_fix_b : STD_LOGIC_VECTOR (24 downto 0);
    signal cStage_uid184_lzCountValAdd_uid81_block_rsrvd_fix_q : STD_LOGIC_VECTOR (31 downto 0);
    signal vStagei_uid186_lzCountValAdd_uid81_block_rsrvd_fix_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid186_lzCountValAdd_uid81_block_rsrvd_fix_q : STD_LOGIC_VECTOR (31 downto 0);
    signal zs_uid187_lzCountValAdd_uid81_block_rsrvd_fix_q : STD_LOGIC_VECTOR (15 downto 0);
    signal vCount_uid189_lzCountValAdd_uid81_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid192_lzCountValAdd_uid81_block_rsrvd_fix_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid192_lzCountValAdd_uid81_block_rsrvd_fix_q : STD_LOGIC_VECTOR (15 downto 0);
    signal zs_uid193_lzCountValAdd_uid81_block_rsrvd_fix_q : STD_LOGIC_VECTOR (7 downto 0);
    signal vCount_uid195_lzCountValAdd_uid81_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid198_lzCountValAdd_uid81_block_rsrvd_fix_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid198_lzCountValAdd_uid81_block_rsrvd_fix_q : STD_LOGIC_VECTOR (7 downto 0);
    signal zs_uid199_lzCountValAdd_uid81_block_rsrvd_fix_q : STD_LOGIC_VECTOR (3 downto 0);
    signal vCount_uid201_lzCountValAdd_uid81_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid204_lzCountValAdd_uid81_block_rsrvd_fix_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid204_lzCountValAdd_uid81_block_rsrvd_fix_q : STD_LOGIC_VECTOR (3 downto 0);
    signal zs_uid205_lzCountValAdd_uid81_block_rsrvd_fix_q : STD_LOGIC_VECTOR (1 downto 0);
    signal vCount_uid207_lzCountValAdd_uid81_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal vCount_uid207_lzCountValAdd_uid81_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid210_lzCountValAdd_uid81_block_rsrvd_fix_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid210_lzCountValAdd_uid81_block_rsrvd_fix_q : STD_LOGIC_VECTOR (1 downto 0);
    signal rVStage_uid212_lzCountValAdd_uid81_block_rsrvd_fix_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vCount_uid213_lzCountValAdd_uid81_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid214_lzCountValAdd_uid81_block_rsrvd_fix_q : STD_LOGIC_VECTOR (5 downto 0);
    signal rVStage_uid217_lzCountValSub_uid83_block_rsrvd_fix_b : STD_LOGIC_VECTOR (31 downto 0);
    signal vCount_uid218_lzCountValSub_uid83_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStage_uid220_lzCountValSub_uid83_block_rsrvd_fix_in : STD_LOGIC_VECTOR (24 downto 0);
    signal vStage_uid220_lzCountValSub_uid83_block_rsrvd_fix_b : STD_LOGIC_VECTOR (24 downto 0);
    signal cStage_uid221_lzCountValSub_uid83_block_rsrvd_fix_q : STD_LOGIC_VECTOR (31 downto 0);
    signal vStagei_uid223_lzCountValSub_uid83_block_rsrvd_fix_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid223_lzCountValSub_uid83_block_rsrvd_fix_q : STD_LOGIC_VECTOR (31 downto 0);
    signal vCount_uid226_lzCountValSub_uid83_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid229_lzCountValSub_uid83_block_rsrvd_fix_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid229_lzCountValSub_uid83_block_rsrvd_fix_q : STD_LOGIC_VECTOR (15 downto 0);
    signal vCount_uid232_lzCountValSub_uid83_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid235_lzCountValSub_uid83_block_rsrvd_fix_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid235_lzCountValSub_uid83_block_rsrvd_fix_q : STD_LOGIC_VECTOR (7 downto 0);
    signal vCount_uid238_lzCountValSub_uid83_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid241_lzCountValSub_uid83_block_rsrvd_fix_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid241_lzCountValSub_uid83_block_rsrvd_fix_q : STD_LOGIC_VECTOR (3 downto 0);
    signal vCount_uid244_lzCountValSub_uid83_block_rsrvd_fix_qi : STD_LOGIC_VECTOR (0 downto 0);
    signal vCount_uid244_lzCountValSub_uid83_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid247_lzCountValSub_uid83_block_rsrvd_fix_s : STD_LOGIC_VECTOR (0 downto 0);
    signal vStagei_uid247_lzCountValSub_uid83_block_rsrvd_fix_q : STD_LOGIC_VECTOR (1 downto 0);
    signal rVStage_uid249_lzCountValSub_uid83_block_rsrvd_fix_b : STD_LOGIC_VECTOR (0 downto 0);
    signal vCount_uid250_lzCountValSub_uid83_block_rsrvd_fix_q : STD_LOGIC_VECTOR (0 downto 0);
    signal r_uid251_lzCountValSub_uid83_block_rsrvd_fix_q : STD_LOGIC_VECTOR (5 downto 0);
    signal rightShiftStage0Idx1Rng1_uid255_alignmentShifter_uid65_block_rsrvd_fix_b : STD_LOGIC_VECTOR (106 downto 0);
    signal rightShiftStage0Idx1_uid257_alignmentShifter_uid65_block_rsrvd_fix_q : STD_LOGIC_VECTOR (107 downto 0);
    signal rightShiftStage0Idx2Rng2_uid258_alignmentShifter_uid65_block_rsrvd_fix_b : STD_LOGIC_VECTOR (105 downto 0);
    signal rightShiftStage0Idx2_uid260_alignmentShifter_uid65_block_rsrvd_fix_q : STD_LOGIC_VECTOR (107 downto 0);
    signal rightShiftStage0Idx3Rng3_uid261_alignmentShifter_uid65_block_rsrvd_fix_b : STD_LOGIC_VECTOR (104 downto 0);
    signal rightShiftStage0Idx3_uid263_alignmentShifter_uid65_block_rsrvd_fix_q : STD_LOGIC_VECTOR (107 downto 0);
    signal rightShiftStage0_uid265_alignmentShifter_uid65_block_rsrvd_fix_s : STD_LOGIC_VECTOR (1 downto 0);
    signal rightShiftStage0_uid265_alignmentShifter_uid65_block_rsrvd_fix_q : STD_LOGIC_VECTOR (107 downto 0);
    signal rightShiftStage1Idx1Rng4_uid266_alignmentShifter_uid65_block_rsrvd_fix_b : STD_LOGIC_VECTOR (103 downto 0);
    signal rightShiftStage1Idx1_uid268_alignmentShifter_uid65_block_rsrvd_fix_q : STD_LOGIC_VECTOR (107 downto 0);
    signal rightShiftStage1Idx2Rng8_uid269_alignmentShifter_uid65_block_rsrvd_fix_b : STD_LOGIC_VECTOR (99 downto 0);
    signal rightShiftStage1Idx2_uid271_alignmentShifter_uid65_block_rsrvd_fix_q : STD_LOGIC_VECTOR (107 downto 0);
    signal rightShiftStage1Idx3Rng12_uid272_alignmentShifter_uid65_block_rsrvd_fix_b : STD_LOGIC_VECTOR (95 downto 0);
    signal rightShiftStage1Idx3Pad12_uid273_alignmentShifter_uid65_block_rsrvd_fix_q : STD_LOGIC_VECTOR (11 downto 0);
    signal rightShiftStage1Idx3_uid274_alignmentShifter_uid65_block_rsrvd_fix_q : STD_LOGIC_VECTOR (107 downto 0);
    signal rightShiftStage1_uid276_alignmentShifter_uid65_block_rsrvd_fix_s : STD_LOGIC_VECTOR (1 downto 0);
    signal rightShiftStage1_uid276_alignmentShifter_uid65_block_rsrvd_fix_q : STD_LOGIC_VECTOR (107 downto 0);
    signal rightShiftStage2Idx1Rng16_uid277_alignmentShifter_uid65_block_rsrvd_fix_b : STD_LOGIC_VECTOR (91 downto 0);
    signal rightShiftStage2Idx1_uid279_alignmentShifter_uid65_block_rsrvd_fix_q : STD_LOGIC_VECTOR (107 downto 0);
    signal rightShiftStage2Idx2Rng32_uid280_alignmentShifter_uid65_block_rsrvd_fix_b : STD_LOGIC_VECTOR (75 downto 0);
    signal rightShiftStage2Idx2_uid282_alignmentShifter_uid65_block_rsrvd_fix_q : STD_LOGIC_VECTOR (107 downto 0);
    signal rightShiftStage2Idx3Rng48_uid283_alignmentShifter_uid65_block_rsrvd_fix_b : STD_LOGIC_VECTOR (59 downto 0);
    signal rightShiftStage2Idx3Pad48_uid284_alignmentShifter_uid65_block_rsrvd_fix_q : STD_LOGIC_VECTOR (47 downto 0);
    signal rightShiftStage2Idx3_uid285_alignmentShifter_uid65_block_rsrvd_fix_q : STD_LOGIC_VECTOR (107 downto 0);
    signal rightShiftStage2_uid287_alignmentShifter_uid65_block_rsrvd_fix_s : STD_LOGIC_VECTOR (1 downto 0);
    signal rightShiftStage2_uid287_alignmentShifter_uid65_block_rsrvd_fix_q : STD_LOGIC_VECTOR (107 downto 0);
    signal fracResAdd_uid77_block_rsrvd_fix_lhsMSBs_select_b : STD_LOGIC_VECTOR (52 downto 0);
    signal fracResAdd_uid77_block_rsrvd_fix_MSBs_sums_a : STD_LOGIC_VECTOR (55 downto 0);
    signal fracResAdd_uid77_block_rsrvd_fix_MSBs_sums_b : STD_LOGIC_VECTOR (55 downto 0);
    signal fracResAdd_uid77_block_rsrvd_fix_MSBs_sums_o : STD_LOGIC_VECTOR (55 downto 0);
    signal fracResAdd_uid77_block_rsrvd_fix_MSBs_sums_q : STD_LOGIC_VECTOR (55 downto 0);
    signal fracResAdd_uid77_block_rsrvd_fix_split_join_q : STD_LOGIC_VECTOR (58 downto 0);
    signal leftShiftStage0Idx1Rng16_uid297_fracPostNormAdd_uid82_block_rsrvd_fix_in : STD_LOGIC_VECTOR (40 downto 0);
    signal leftShiftStage0Idx1Rng16_uid297_fracPostNormAdd_uid82_block_rsrvd_fix_b : STD_LOGIC_VECTOR (40 downto 0);
    signal leftShiftStage0Idx1_uid298_fracPostNormAdd_uid82_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal leftShiftStage0Idx2_uid301_fracPostNormAdd_uid82_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal leftShiftStage0Idx3Rng48_uid303_fracPostNormAdd_uid82_block_rsrvd_fix_in : STD_LOGIC_VECTOR (8 downto 0);
    signal leftShiftStage0Idx3Rng48_uid303_fracPostNormAdd_uid82_block_rsrvd_fix_b : STD_LOGIC_VECTOR (8 downto 0);
    signal leftShiftStage0Idx3_uid304_fracPostNormAdd_uid82_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal leftShiftStage0_uid306_fracPostNormAdd_uid82_block_rsrvd_fix_s : STD_LOGIC_VECTOR (1 downto 0);
    signal leftShiftStage0_uid306_fracPostNormAdd_uid82_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal leftShiftStage1Idx1Rng4_uid308_fracPostNormAdd_uid82_block_rsrvd_fix_in : STD_LOGIC_VECTOR (52 downto 0);
    signal leftShiftStage1Idx1Rng4_uid308_fracPostNormAdd_uid82_block_rsrvd_fix_b : STD_LOGIC_VECTOR (52 downto 0);
    signal leftShiftStage1Idx1_uid309_fracPostNormAdd_uid82_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal leftShiftStage1Idx2Rng8_uid311_fracPostNormAdd_uid82_block_rsrvd_fix_in : STD_LOGIC_VECTOR (48 downto 0);
    signal leftShiftStage1Idx2Rng8_uid311_fracPostNormAdd_uid82_block_rsrvd_fix_b : STD_LOGIC_VECTOR (48 downto 0);
    signal leftShiftStage1Idx2_uid312_fracPostNormAdd_uid82_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal leftShiftStage1Idx3Rng12_uid314_fracPostNormAdd_uid82_block_rsrvd_fix_in : STD_LOGIC_VECTOR (44 downto 0);
    signal leftShiftStage1Idx3Rng12_uid314_fracPostNormAdd_uid82_block_rsrvd_fix_b : STD_LOGIC_VECTOR (44 downto 0);
    signal leftShiftStage1Idx3_uid315_fracPostNormAdd_uid82_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal leftShiftStage1_uid317_fracPostNormAdd_uid82_block_rsrvd_fix_s : STD_LOGIC_VECTOR (1 downto 0);
    signal leftShiftStage1_uid317_fracPostNormAdd_uid82_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal leftShiftStage2Idx1Rng1_uid319_fracPostNormAdd_uid82_block_rsrvd_fix_in : STD_LOGIC_VECTOR (55 downto 0);
    signal leftShiftStage2Idx1Rng1_uid319_fracPostNormAdd_uid82_block_rsrvd_fix_b : STD_LOGIC_VECTOR (55 downto 0);
    signal leftShiftStage2Idx1_uid320_fracPostNormAdd_uid82_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal leftShiftStage2Idx2Rng2_uid322_fracPostNormAdd_uid82_block_rsrvd_fix_in : STD_LOGIC_VECTOR (54 downto 0);
    signal leftShiftStage2Idx2Rng2_uid322_fracPostNormAdd_uid82_block_rsrvd_fix_b : STD_LOGIC_VECTOR (54 downto 0);
    signal leftShiftStage2Idx2_uid323_fracPostNormAdd_uid82_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal leftShiftStage2Idx3Rng3_uid325_fracPostNormAdd_uid82_block_rsrvd_fix_in : STD_LOGIC_VECTOR (53 downto 0);
    signal leftShiftStage2Idx3Rng3_uid325_fracPostNormAdd_uid82_block_rsrvd_fix_b : STD_LOGIC_VECTOR (53 downto 0);
    signal leftShiftStage2Idx3_uid326_fracPostNormAdd_uid82_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal leftShiftStage2_uid328_fracPostNormAdd_uid82_block_rsrvd_fix_s : STD_LOGIC_VECTOR (1 downto 0);
    signal leftShiftStage2_uid328_fracPostNormAdd_uid82_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal leftShiftStage0Idx1Rng16_uid333_fracPostNormSub_uid84_block_rsrvd_fix_in : STD_LOGIC_VECTOR (40 downto 0);
    signal leftShiftStage0Idx1Rng16_uid333_fracPostNormSub_uid84_block_rsrvd_fix_b : STD_LOGIC_VECTOR (40 downto 0);
    signal leftShiftStage0Idx1_uid334_fracPostNormSub_uid84_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal leftShiftStage0Idx2_uid337_fracPostNormSub_uid84_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal leftShiftStage0Idx3Rng48_uid339_fracPostNormSub_uid84_block_rsrvd_fix_in : STD_LOGIC_VECTOR (8 downto 0);
    signal leftShiftStage0Idx3Rng48_uid339_fracPostNormSub_uid84_block_rsrvd_fix_b : STD_LOGIC_VECTOR (8 downto 0);
    signal leftShiftStage0Idx3_uid340_fracPostNormSub_uid84_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal leftShiftStage0_uid342_fracPostNormSub_uid84_block_rsrvd_fix_s : STD_LOGIC_VECTOR (1 downto 0);
    signal leftShiftStage0_uid342_fracPostNormSub_uid84_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal leftShiftStage1Idx1Rng4_uid344_fracPostNormSub_uid84_block_rsrvd_fix_in : STD_LOGIC_VECTOR (52 downto 0);
    signal leftShiftStage1Idx1Rng4_uid344_fracPostNormSub_uid84_block_rsrvd_fix_b : STD_LOGIC_VECTOR (52 downto 0);
    signal leftShiftStage1Idx1_uid345_fracPostNormSub_uid84_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal leftShiftStage1Idx2Rng8_uid347_fracPostNormSub_uid84_block_rsrvd_fix_in : STD_LOGIC_VECTOR (48 downto 0);
    signal leftShiftStage1Idx2Rng8_uid347_fracPostNormSub_uid84_block_rsrvd_fix_b : STD_LOGIC_VECTOR (48 downto 0);
    signal leftShiftStage1Idx2_uid348_fracPostNormSub_uid84_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal leftShiftStage1Idx3Rng12_uid350_fracPostNormSub_uid84_block_rsrvd_fix_in : STD_LOGIC_VECTOR (44 downto 0);
    signal leftShiftStage1Idx3Rng12_uid350_fracPostNormSub_uid84_block_rsrvd_fix_b : STD_LOGIC_VECTOR (44 downto 0);
    signal leftShiftStage1Idx3_uid351_fracPostNormSub_uid84_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal leftShiftStage1_uid353_fracPostNormSub_uid84_block_rsrvd_fix_s : STD_LOGIC_VECTOR (1 downto 0);
    signal leftShiftStage1_uid353_fracPostNormSub_uid84_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal leftShiftStage2Idx1Rng1_uid355_fracPostNormSub_uid84_block_rsrvd_fix_in : STD_LOGIC_VECTOR (55 downto 0);
    signal leftShiftStage2Idx1Rng1_uid355_fracPostNormSub_uid84_block_rsrvd_fix_b : STD_LOGIC_VECTOR (55 downto 0);
    signal leftShiftStage2Idx1_uid356_fracPostNormSub_uid84_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal leftShiftStage2Idx2Rng2_uid358_fracPostNormSub_uid84_block_rsrvd_fix_in : STD_LOGIC_VECTOR (54 downto 0);
    signal leftShiftStage2Idx2Rng2_uid358_fracPostNormSub_uid84_block_rsrvd_fix_b : STD_LOGIC_VECTOR (54 downto 0);
    signal leftShiftStage2Idx2_uid359_fracPostNormSub_uid84_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal leftShiftStage2Idx3Rng3_uid361_fracPostNormSub_uid84_block_rsrvd_fix_in : STD_LOGIC_VECTOR (53 downto 0);
    signal leftShiftStage2Idx3Rng3_uid361_fracPostNormSub_uid84_block_rsrvd_fix_b : STD_LOGIC_VECTOR (53 downto 0);
    signal leftShiftStage2Idx3_uid362_fracPostNormSub_uid84_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal leftShiftStage2_uid364_fracPostNormSub_uid84_block_rsrvd_fix_s : STD_LOGIC_VECTOR (1 downto 0);
    signal leftShiftStage2_uid364_fracPostNormSub_uid84_block_rsrvd_fix_q : STD_LOGIC_VECTOR (56 downto 0);
    signal rightShiftStageSel0Dto0_uid264_alignmentShifter_uid65_block_rsrvd_fix_bit_select_merged_b : STD_LOGIC_VECTOR (1 downto 0);
    signal rightShiftStageSel0Dto0_uid264_alignmentShifter_uid65_block_rsrvd_fix_bit_select_merged_c : STD_LOGIC_VECTOR (1 downto 0);
    signal rightShiftStageSel0Dto0_uid264_alignmentShifter_uid65_block_rsrvd_fix_bit_select_merged_d : STD_LOGIC_VECTOR (1 downto 0);
    signal fracResAdd_uid77_block_rsrvd_fix_rhsMSBs_select_bit_select_merged_b : STD_LOGIC_VECTOR (54 downto 0);
    signal fracResAdd_uid77_block_rsrvd_fix_rhsMSBs_select_bit_select_merged_c : STD_LOGIC_VECTOR (2 downto 0);
    signal rVStage_uid188_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b : STD_LOGIC_VECTOR (15 downto 0);
    signal rVStage_uid188_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_c : STD_LOGIC_VECTOR (15 downto 0);
    signal rVStage_uid194_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b : STD_LOGIC_VECTOR (7 downto 0);
    signal rVStage_uid194_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_c : STD_LOGIC_VECTOR (7 downto 0);
    signal rVStage_uid200_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b : STD_LOGIC_VECTOR (3 downto 0);
    signal rVStage_uid200_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_c : STD_LOGIC_VECTOR (3 downto 0);
    signal rVStage_uid206_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b : STD_LOGIC_VECTOR (1 downto 0);
    signal rVStage_uid206_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_c : STD_LOGIC_VECTOR (1 downto 0);
    signal leftShiftStageSel0Dto4_uid305_fracPostNormAdd_uid82_block_rsrvd_fix_bit_select_merged_b : STD_LOGIC_VECTOR (1 downto 0);
    signal leftShiftStageSel0Dto4_uid305_fracPostNormAdd_uid82_block_rsrvd_fix_bit_select_merged_c : STD_LOGIC_VECTOR (1 downto 0);
    signal leftShiftStageSel0Dto4_uid305_fracPostNormAdd_uid82_block_rsrvd_fix_bit_select_merged_d : STD_LOGIC_VECTOR (1 downto 0);
    signal rVStage_uid225_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b : STD_LOGIC_VECTOR (15 downto 0);
    signal rVStage_uid225_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_c : STD_LOGIC_VECTOR (15 downto 0);
    signal rVStage_uid231_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b : STD_LOGIC_VECTOR (7 downto 0);
    signal rVStage_uid231_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_c : STD_LOGIC_VECTOR (7 downto 0);
    signal rVStage_uid237_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b : STD_LOGIC_VECTOR (3 downto 0);
    signal rVStage_uid237_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_c : STD_LOGIC_VECTOR (3 downto 0);
    signal rVStage_uid243_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b : STD_LOGIC_VECTOR (1 downto 0);
    signal rVStage_uid243_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_c : STD_LOGIC_VECTOR (1 downto 0);
    signal leftShiftStageSel0Dto4_uid341_fracPostNormSub_uid84_block_rsrvd_fix_bit_select_merged_b : STD_LOGIC_VECTOR (1 downto 0);
    signal leftShiftStageSel0Dto4_uid341_fracPostNormSub_uid84_block_rsrvd_fix_bit_select_merged_c : STD_LOGIC_VECTOR (1 downto 0);
    signal leftShiftStageSel0Dto4_uid341_fracPostNormSub_uid84_block_rsrvd_fix_bit_select_merged_d : STD_LOGIC_VECTOR (1 downto 0);
    signal stickyBits_uid68_block_rsrvd_fix_bit_select_merged_b : STD_LOGIC_VECTOR (52 downto 0);
    signal stickyBits_uid68_block_rsrvd_fix_bit_select_merged_c : STD_LOGIC_VECTOR (54 downto 0);
    signal redist0_stickyBits_uid68_block_rsrvd_fix_bit_select_merged_c_1_q : STD_LOGIC_VECTOR (54 downto 0);
    signal redist1_rVStage_uid243_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b_1_q : STD_LOGIC_VECTOR (1 downto 0);
    signal redist2_rVStage_uid243_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_c_1_q : STD_LOGIC_VECTOR (1 downto 0);
    signal redist3_rVStage_uid206_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b_1_q : STD_LOGIC_VECTOR (1 downto 0);
    signal redist4_rVStage_uid206_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_c_1_q : STD_LOGIC_VECTOR (1 downto 0);
    signal redist5_vCount_uid238_lzCountValSub_uid83_block_rsrvd_fix_q_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist6_vCount_uid232_lzCountValSub_uid83_block_rsrvd_fix_q_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist7_vCount_uid226_lzCountValSub_uid83_block_rsrvd_fix_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist7_vCount_uid226_lzCountValSub_uid83_block_rsrvd_fix_q_2_delay_0 : STD_LOGIC_VECTOR (0 downto 0);
    signal redist8_vStage_uid220_lzCountValSub_uid83_block_rsrvd_fix_b_2_q : STD_LOGIC_VECTOR (24 downto 0);
    signal redist8_vStage_uid220_lzCountValSub_uid83_block_rsrvd_fix_b_2_delay_0 : STD_LOGIC_VECTOR (24 downto 0);
    signal redist9_vCount_uid218_lzCountValSub_uid83_block_rsrvd_fix_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist9_vCount_uid218_lzCountValSub_uid83_block_rsrvd_fix_q_2_delay_0 : STD_LOGIC_VECTOR (0 downto 0);
    signal redist10_vCount_uid201_lzCountValAdd_uid81_block_rsrvd_fix_q_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist11_vCount_uid195_lzCountValAdd_uid81_block_rsrvd_fix_q_1_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist12_vCount_uid189_lzCountValAdd_uid81_block_rsrvd_fix_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist12_vCount_uid189_lzCountValAdd_uid81_block_rsrvd_fix_q_2_delay_0 : STD_LOGIC_VECTOR (0 downto 0);
    signal redist13_vStage_uid183_lzCountValAdd_uid81_block_rsrvd_fix_b_2_q : STD_LOGIC_VECTOR (24 downto 0);
    signal redist13_vStage_uid183_lzCountValAdd_uid81_block_rsrvd_fix_b_2_delay_0 : STD_LOGIC_VECTOR (24 downto 0);
    signal redist14_vCount_uid181_lzCountValAdd_uid81_block_rsrvd_fix_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist14_vCount_uid181_lzCountValAdd_uid81_block_rsrvd_fix_q_2_delay_0 : STD_LOGIC_VECTOR (0 downto 0);
    signal redist15_oneIsNaN_uid128_block_rsrvd_fix_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist15_oneIsNaN_uid128_block_rsrvd_fix_q_3_delay_0 : STD_LOGIC_VECTOR (0 downto 0);
    signal redist16_regInputs_uid124_block_rsrvd_fix_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist16_regInputs_uid124_block_rsrvd_fix_q_3_delay_0 : STD_LOGIC_VECTOR (0 downto 0);
    signal redist17_expRPreExcAdd_uid119_block_rsrvd_fix_b_1_q : STD_LOGIC_VECTOR (10 downto 0);
    signal redist18_fracRPreExcAdd_uid118_block_rsrvd_fix_b_1_q : STD_LOGIC_VECTOR (51 downto 0);
    signal redist19_fracPostNormAddRndRange_uid92_block_rsrvd_fix_b_1_q : STD_LOGIC_VECTOR (52 downto 0);
    signal redist20_fracPostNormSubRndRange_uid90_block_rsrvd_fix_b_1_q : STD_LOGIC_VECTOR (52 downto 0);
    signal redist21_aMinusA_uid86_block_rsrvd_fix_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist22_fracResSubNoSignExt_uid80_block_rsrvd_fix_b_1_q : STD_LOGIC_VECTOR (56 downto 0);
    signal redist23_fracResSubNoSignExt_uid80_block_rsrvd_fix_b_3_q : STD_LOGIC_VECTOR (56 downto 0);
    signal redist23_fracResSubNoSignExt_uid80_block_rsrvd_fix_b_3_delay_0 : STD_LOGIC_VECTOR (56 downto 0);
    signal redist24_fracResAddNoSignExt_uid79_block_rsrvd_fix_b_1_q : STD_LOGIC_VECTOR (56 downto 0);
    signal redist25_fracResAddNoSignExt_uid79_block_rsrvd_fix_b_3_q : STD_LOGIC_VECTOR (56 downto 0);
    signal redist25_fracResAddNoSignExt_uid79_block_rsrvd_fix_b_3_delay_0 : STD_LOGIC_VECTOR (56 downto 0);
    signal redist26_sigB_uid47_block_rsrvd_fix_b_10_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist27_sigA_uid46_block_rsrvd_fix_b_10_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist28_InvExpXIsZero_uid40_block_rsrvd_fix_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist28_InvExpXIsZero_uid40_block_rsrvd_fix_q_4_delay_0 : STD_LOGIC_VECTOR (0 downto 0);
    signal redist28_InvExpXIsZero_uid40_block_rsrvd_fix_q_4_delay_1 : STD_LOGIC_VECTOR (0 downto 0);
    signal redist28_InvExpXIsZero_uid40_block_rsrvd_fix_q_4_delay_2 : STD_LOGIC_VECTOR (0 downto 0);
    signal redist29_excI_sigb_uid37_block_rsrvd_fix_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist29_excI_sigb_uid37_block_rsrvd_fix_q_3_delay_0 : STD_LOGIC_VECTOR (0 downto 0);
    signal redist30_fracXIsZero_uid35_block_rsrvd_fix_q_4_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist30_fracXIsZero_uid35_block_rsrvd_fix_q_4_delay_0 : STD_LOGIC_VECTOR (0 downto 0);
    signal redist30_fracXIsZero_uid35_block_rsrvd_fix_q_4_delay_1 : STD_LOGIC_VECTOR (0 downto 0);
    signal redist31_expXIsMax_uid34_block_rsrvd_fix_q_6_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist31_expXIsMax_uid34_block_rsrvd_fix_q_6_delay_0 : STD_LOGIC_VECTOR (0 downto 0);
    signal redist31_expXIsMax_uid34_block_rsrvd_fix_q_6_delay_1 : STD_LOGIC_VECTOR (0 downto 0);
    signal redist31_expXIsMax_uid34_block_rsrvd_fix_q_6_delay_2 : STD_LOGIC_VECTOR (0 downto 0);
    signal redist31_expXIsMax_uid34_block_rsrvd_fix_q_6_delay_3 : STD_LOGIC_VECTOR (0 downto 0);
    signal redist32_excZ_sigb_uid13_uid33_block_rsrvd_fix_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist33_excZ_sigb_uid13_uid33_block_rsrvd_fix_q_9_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist34_frac_sigb_uid32_block_rsrvd_fix_b_3_q : STD_LOGIC_VECTOR (51 downto 0);
    signal redist34_frac_sigb_uid32_block_rsrvd_fix_b_3_delay_0 : STD_LOGIC_VECTOR (51 downto 0);
    signal redist34_frac_sigb_uid32_block_rsrvd_fix_b_3_delay_1 : STD_LOGIC_VECTOR (51 downto 0);
    signal redist35_exp_sigb_uid31_block_rsrvd_fix_b_1_q : STD_LOGIC_VECTOR (10 downto 0);
    signal redist36_excI_siga_uid23_block_rsrvd_fix_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist36_excI_siga_uid23_block_rsrvd_fix_q_3_delay_0 : STD_LOGIC_VECTOR (0 downto 0);
    signal redist37_fracXIsZero_uid21_block_rsrvd_fix_q_2_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist38_excZ_siga_uid12_uid19_block_rsrvd_fix_q_3_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist38_excZ_siga_uid12_uid19_block_rsrvd_fix_q_3_delay_0 : STD_LOGIC_VECTOR (0 downto 0);
    signal redist38_excZ_siga_uid12_uid19_block_rsrvd_fix_q_3_delay_1 : STD_LOGIC_VECTOR (0 downto 0);
    signal redist39_frac_siga_uid18_block_rsrvd_fix_b_5_q : STD_LOGIC_VECTOR (51 downto 0);
    signal redist39_frac_siga_uid18_block_rsrvd_fix_b_5_delay_0 : STD_LOGIC_VECTOR (51 downto 0);
    signal redist39_frac_siga_uid18_block_rsrvd_fix_b_5_delay_1 : STD_LOGIC_VECTOR (51 downto 0);
    signal redist40_exp_siga_uid17_block_rsrvd_fix_b_1_q : STD_LOGIC_VECTOR (10 downto 0);
    signal redist42_swap_uid11_block_rsrvd_fix_q_10_q : STD_LOGIC_VECTOR (0 downto 0);
    signal redist39_frac_siga_uid18_block_rsrvd_fix_b_5_inputreg0_q : STD_LOGIC_VECTOR (51 downto 0);
    signal redist39_frac_siga_uid18_block_rsrvd_fix_b_5_outputreg0_q : STD_LOGIC_VECTOR (51 downto 0);
    signal redist41_exp_siga_uid17_block_rsrvd_fix_b_7_mem_reset0 : std_logic;
    signal redist41_exp_siga_uid17_block_rsrvd_fix_b_7_mem_ia : STD_LOGIC_VECTOR (10 downto 0);
    signal redist41_exp_siga_uid17_block_rsrvd_fix_b_7_mem_aa : STD_LOGIC_VECTOR (2 downto 0);
    signal redist41_exp_siga_uid17_block_rsrvd_fix_b_7_mem_ab : STD_LOGIC_VECTOR (2 downto 0);
    signal redist41_exp_siga_uid17_block_rsrvd_fix_b_7_mem_iq : STD_LOGIC_VECTOR (10 downto 0);
    signal redist41_exp_siga_uid17_block_rsrvd_fix_b_7_mem_q : STD_LOGIC_VECTOR (10 downto 0);
    signal redist41_exp_siga_uid17_block_rsrvd_fix_b_7_wraddr_q : STD_LOGIC_VECTOR (2 downto 0);
    -- Initial-value here is arbitrary, but a resolved value is necessary for simulation.
    signal redist41_exp_siga_uid17_block_rsrvd_fix_b_7_wraddr_i : UNSIGNED (2 downto 0) := "111";
    attribute preserve_syn_only : boolean;
    attribute preserve_syn_only of redist41_exp_siga_uid17_block_rsrvd_fix_b_7_wraddr_i : signal is true;
    signal redist41_exp_siga_uid17_block_rsrvd_fix_b_7_offset_q : STD_LOGIC_VECTOR (2 downto 0);
    signal redist41_exp_siga_uid17_block_rsrvd_fix_b_7_rdcnt_a : STD_LOGIC_VECTOR (3 downto 0);
    signal redist41_exp_siga_uid17_block_rsrvd_fix_b_7_rdcnt_b : STD_LOGIC_VECTOR (3 downto 0);
    signal redist41_exp_siga_uid17_block_rsrvd_fix_b_7_rdcnt_o : STD_LOGIC_VECTOR (3 downto 0);
    signal redist41_exp_siga_uid17_block_rsrvd_fix_b_7_rdcnt_q : STD_LOGIC_VECTOR (3 downto 0);

begin


    -- GND(CONSTANT,0)
    GND_q <= "0";

    -- expFracY_uid9_block_rsrvd_fix(BITSELECT,8)@0
    expFracY_uid9_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(in_1(62 downto 0));

    -- expFracX_uid8_block_rsrvd_fix(BITSELECT,7)@0
    expFracX_uid8_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(in_0(62 downto 0));

    -- xGTEy_uid10_block_rsrvd_fix(COMPARE,9)@0
    xGTEy_uid10_block_rsrvd_fix_a <= STD_LOGIC_VECTOR("00" & expFracX_uid8_block_rsrvd_fix_b);
    xGTEy_uid10_block_rsrvd_fix_b <= STD_LOGIC_VECTOR("00" & expFracY_uid9_block_rsrvd_fix_b);
    xGTEy_uid10_block_rsrvd_fix_o <= STD_LOGIC_VECTOR(UNSIGNED(xGTEy_uid10_block_rsrvd_fix_a) - UNSIGNED(xGTEy_uid10_block_rsrvd_fix_b));
    xGTEy_uid10_block_rsrvd_fix_n(0) <= not (xGTEy_uid10_block_rsrvd_fix_o(64));

    -- sigb_uid13_block_rsrvd_fix(MUX,12)@0
    sigb_uid13_block_rsrvd_fix_s <= xGTEy_uid10_block_rsrvd_fix_n;
    sigb_uid13_block_rsrvd_fix_combproc: PROCESS (sigb_uid13_block_rsrvd_fix_s, in_0, in_1)
    BEGIN
        CASE (sigb_uid13_block_rsrvd_fix_s) IS
            WHEN "0" => sigb_uid13_block_rsrvd_fix_q <= in_0;
            WHEN "1" => sigb_uid13_block_rsrvd_fix_q <= in_1;
            WHEN OTHERS => sigb_uid13_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- sigB_uid47_block_rsrvd_fix(BITSELECT,46)@0
    sigB_uid47_block_rsrvd_fix_b <= sigb_uid13_block_rsrvd_fix_q(63 downto 63);

    -- redist26_sigB_uid47_block_rsrvd_fix_b_10(DELAY,404)
    redist26_sigB_uid47_block_rsrvd_fix_b_10 : dspba_delay
    GENERIC MAP ( width => 1, depth => 10, reset_kind => "NONE", phase => 0, modulus => 1024 )
    PORT MAP ( xin => sigB_uid47_block_rsrvd_fix_b, xout => redist26_sigB_uid47_block_rsrvd_fix_b_10_q, clk => clk, aclr => areset, ena => '1' );

    -- siga_uid12_block_rsrvd_fix(MUX,11)@0
    siga_uid12_block_rsrvd_fix_s <= xGTEy_uid10_block_rsrvd_fix_n;
    siga_uid12_block_rsrvd_fix_combproc: PROCESS (siga_uid12_block_rsrvd_fix_s, in_1, in_0)
    BEGIN
        CASE (siga_uid12_block_rsrvd_fix_s) IS
            WHEN "0" => siga_uid12_block_rsrvd_fix_q <= in_1;
            WHEN "1" => siga_uid12_block_rsrvd_fix_q <= in_0;
            WHEN OTHERS => siga_uid12_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- sigA_uid46_block_rsrvd_fix(BITSELECT,45)@0
    sigA_uid46_block_rsrvd_fix_b <= siga_uid12_block_rsrvd_fix_q(63 downto 63);

    -- redist27_sigA_uid46_block_rsrvd_fix_b_10(DELAY,405)
    redist27_sigA_uid46_block_rsrvd_fix_b_10 : dspba_delay
    GENERIC MAP ( width => 1, depth => 10, reset_kind => "NONE", phase => 0, modulus => 1024 )
    PORT MAP ( xin => sigA_uid46_block_rsrvd_fix_b, xout => redist27_sigA_uid46_block_rsrvd_fix_b_10_q, clk => clk, aclr => areset, ena => '1' );

    -- effSub_uid48_block_rsrvd_fix(LOGICAL,47)@10
    effSub_uid48_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(redist27_sigA_uid46_block_rsrvd_fix_b_10_q xor redist26_sigB_uid47_block_rsrvd_fix_b_10_q);

    -- invEffSub_uid134_block_rsrvd_fix(LOGICAL,133)@10
    invEffSub_uid134_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (effSub_uid48_block_rsrvd_fix_q));

    -- cstAllZWE_uid16_block_rsrvd_fix(CONSTANT,15)
    cstAllZWE_uid16_block_rsrvd_fix_q <= "00000000000";

    -- exp_sigb_uid31_block_rsrvd_fix(BITSELECT,30)@0
    exp_sigb_uid31_block_rsrvd_fix_in <= sigb_uid13_block_rsrvd_fix_q(62 downto 0);
    exp_sigb_uid31_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(exp_sigb_uid31_block_rsrvd_fix_in(62 downto 52));

    -- redist35_exp_sigb_uid31_block_rsrvd_fix_b_1(DELAY,413)
    redist35_exp_sigb_uid31_block_rsrvd_fix_b_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist35_exp_sigb_uid31_block_rsrvd_fix_b_1_q <= exp_sigb_uid31_block_rsrvd_fix_b;
            END IF;
        END IF;
    END PROCESS;

    -- excZ_sigb_uid13_uid33_block_rsrvd_fix(LOGICAL,32)@1 + 1
    excZ_sigb_uid13_uid33_block_rsrvd_fix_qi <= "1" WHEN redist35_exp_sigb_uid31_block_rsrvd_fix_b_1_q = cstAllZWE_uid16_block_rsrvd_fix_q ELSE "0";
    excZ_sigb_uid13_uid33_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => excZ_sigb_uid13_uid33_block_rsrvd_fix_qi, xout => excZ_sigb_uid13_uid33_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- redist32_excZ_sigb_uid13_uid33_block_rsrvd_fix_q_2(DELAY,410)
    redist32_excZ_sigb_uid13_uid33_block_rsrvd_fix_q_2_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist32_excZ_sigb_uid13_uid33_block_rsrvd_fix_q_2_q <= excZ_sigb_uid13_uid33_block_rsrvd_fix_q;
            END IF;
        END IF;
    END PROCESS;

    -- redist33_excZ_sigb_uid13_uid33_block_rsrvd_fix_q_9(DELAY,411)
    redist33_excZ_sigb_uid13_uid33_block_rsrvd_fix_q_9 : dspba_delay
    GENERIC MAP ( width => 1, depth => 7, reset_kind => "NONE", phase => 0, modulus => 1024 )
    PORT MAP ( xin => redist32_excZ_sigb_uid13_uid33_block_rsrvd_fix_q_2_q, xout => redist33_excZ_sigb_uid13_uid33_block_rsrvd_fix_q_9_q, clk => clk, aclr => areset, ena => '1' );

    -- redist41_exp_siga_uid17_block_rsrvd_fix_b_7_offset(CONSTANT,425)
    redist41_exp_siga_uid17_block_rsrvd_fix_b_7_offset_q <= "101";

    -- redist41_exp_siga_uid17_block_rsrvd_fix_b_7_rdcnt(ADD,426)
    redist41_exp_siga_uid17_block_rsrvd_fix_b_7_rdcnt_a <= STD_LOGIC_VECTOR("0" & redist41_exp_siga_uid17_block_rsrvd_fix_b_7_wraddr_q);
    redist41_exp_siga_uid17_block_rsrvd_fix_b_7_rdcnt_b <= STD_LOGIC_VECTOR("0" & redist41_exp_siga_uid17_block_rsrvd_fix_b_7_offset_q);
    redist41_exp_siga_uid17_block_rsrvd_fix_b_7_rdcnt_clkproc: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist41_exp_siga_uid17_block_rsrvd_fix_b_7_rdcnt_o <= STD_LOGIC_VECTOR(UNSIGNED(redist41_exp_siga_uid17_block_rsrvd_fix_b_7_rdcnt_a) + UNSIGNED(redist41_exp_siga_uid17_block_rsrvd_fix_b_7_rdcnt_b));
            END IF;
        END IF;
    END PROCESS;
    redist41_exp_siga_uid17_block_rsrvd_fix_b_7_rdcnt_q <= redist41_exp_siga_uid17_block_rsrvd_fix_b_7_rdcnt_o(3 downto 0);

    -- exp_siga_uid17_block_rsrvd_fix(BITSELECT,16)@0
    exp_siga_uid17_block_rsrvd_fix_in <= siga_uid12_block_rsrvd_fix_q(62 downto 0);
    exp_siga_uid17_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(exp_siga_uid17_block_rsrvd_fix_in(62 downto 52));

    -- redist40_exp_siga_uid17_block_rsrvd_fix_b_1(DELAY,418)
    redist40_exp_siga_uid17_block_rsrvd_fix_b_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist40_exp_siga_uid17_block_rsrvd_fix_b_1_q <= exp_siga_uid17_block_rsrvd_fix_b;
            END IF;
        END IF;
    END PROCESS;

    -- redist41_exp_siga_uid17_block_rsrvd_fix_b_7_wraddr(COUNTER,424)
    -- low=0, high=7, step=1, init=0
    redist41_exp_siga_uid17_block_rsrvd_fix_b_7_wraddr_clkproc: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist41_exp_siga_uid17_block_rsrvd_fix_b_7_wraddr_i <= redist41_exp_siga_uid17_block_rsrvd_fix_b_7_wraddr_i + 1;
            END IF;
        END IF;
    END PROCESS;
    redist41_exp_siga_uid17_block_rsrvd_fix_b_7_wraddr_q <= STD_LOGIC_VECTOR(RESIZE(redist41_exp_siga_uid17_block_rsrvd_fix_b_7_wraddr_i, 3));

    -- redist41_exp_siga_uid17_block_rsrvd_fix_b_7_mem(DUALMEM,423)
    redist41_exp_siga_uid17_block_rsrvd_fix_b_7_mem_ia <= STD_LOGIC_VECTOR(redist40_exp_siga_uid17_block_rsrvd_fix_b_1_q);
    redist41_exp_siga_uid17_block_rsrvd_fix_b_7_mem_aa <= redist41_exp_siga_uid17_block_rsrvd_fix_b_7_wraddr_q;
    redist41_exp_siga_uid17_block_rsrvd_fix_b_7_mem_ab <= redist41_exp_siga_uid17_block_rsrvd_fix_b_7_rdcnt_q(2 downto 0);
    redist41_exp_siga_uid17_block_rsrvd_fix_b_7_mem_dmem : altera_syncram
    GENERIC MAP (
        ram_block_type => "MLAB",
        operation_mode => "DUAL_PORT",
        width_a => 11,
        widthad_a => 3,
        numwords_a => 8,
        width_b => 11,
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
        address_a => redist41_exp_siga_uid17_block_rsrvd_fix_b_7_mem_aa,
        data_a => redist41_exp_siga_uid17_block_rsrvd_fix_b_7_mem_ia,
        wren_a => VCC_q(0),
        address_b => redist41_exp_siga_uid17_block_rsrvd_fix_b_7_mem_ab,
        q_b => redist41_exp_siga_uid17_block_rsrvd_fix_b_7_mem_iq
    );
    redist41_exp_siga_uid17_block_rsrvd_fix_b_7_mem_q <= STD_LOGIC_VECTOR(redist41_exp_siga_uid17_block_rsrvd_fix_b_7_mem_iq(10 downto 0));

    -- excZ_siga_uid12_uid19_block_rsrvd_fix(LOGICAL,18)@7
    excZ_siga_uid12_uid19_block_rsrvd_fix_q <= "1" WHEN redist41_exp_siga_uid17_block_rsrvd_fix_b_7_mem_q = cstAllZWE_uid16_block_rsrvd_fix_q ELSE "0";

    -- redist38_excZ_siga_uid12_uid19_block_rsrvd_fix_q_3(DELAY,416)
    redist38_excZ_siga_uid12_uid19_block_rsrvd_fix_q_3_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist38_excZ_siga_uid12_uid19_block_rsrvd_fix_q_3_delay_0 <= STD_LOGIC_VECTOR(excZ_siga_uid12_uid19_block_rsrvd_fix_q);
                redist38_excZ_siga_uid12_uid19_block_rsrvd_fix_q_3_delay_1 <= redist38_excZ_siga_uid12_uid19_block_rsrvd_fix_q_3_delay_0;
                redist38_excZ_siga_uid12_uid19_block_rsrvd_fix_q_3_q <= STD_LOGIC_VECTOR(redist38_excZ_siga_uid12_uid19_block_rsrvd_fix_q_3_delay_1);
            END IF;
        END IF;
    END PROCESS;

    -- zMz_uid169_block_rsrvd_fix(LOGICAL,168)@10
    zMz_uid169_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(redist38_excZ_siga_uid12_uid19_block_rsrvd_fix_q_3_q and redist33_excZ_sigb_uid13_uid33_block_rsrvd_fix_q_9_q and invEffSub_uid134_block_rsrvd_fix_q);

    -- invZMZSub_uid170_block_rsrvd_fix(LOGICAL,169)@10
    invZMZSub_uid170_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (zMz_uid169_block_rsrvd_fix_q));

    -- cAmA_uid85_block_rsrvd_fix(CONSTANT,84)
    cAmA_uid85_block_rsrvd_fix_q <= "111001";

    -- zs_uid179_lzCountValAdd_uid81_block_rsrvd_fix(CONSTANT,178)
    zs_uid179_lzCountValAdd_uid81_block_rsrvd_fix_q <= "00000000000000000000000000000000";

    -- rightShiftStage2Idx3Pad48_uid284_alignmentShifter_uid65_block_rsrvd_fix(CONSTANT,283)
    rightShiftStage2Idx3Pad48_uid284_alignmentShifter_uid65_block_rsrvd_fix_q <= "000000000000000000000000000000000000000000000000";

    -- rightShiftStage2Idx3Rng48_uid283_alignmentShifter_uid65_block_rsrvd_fix(BITSELECT,282)@3
    rightShiftStage2Idx3Rng48_uid283_alignmentShifter_uid65_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(rightShiftStage1_uid276_alignmentShifter_uid65_block_rsrvd_fix_q(107 downto 48));

    -- rightShiftStage2Idx3_uid285_alignmentShifter_uid65_block_rsrvd_fix(BITJOIN,284)@3
    rightShiftStage2Idx3_uid285_alignmentShifter_uid65_block_rsrvd_fix_q <= rightShiftStage2Idx3Pad48_uid284_alignmentShifter_uid65_block_rsrvd_fix_q & rightShiftStage2Idx3Rng48_uid283_alignmentShifter_uid65_block_rsrvd_fix_b;

    -- rightShiftStage2Idx2Rng32_uid280_alignmentShifter_uid65_block_rsrvd_fix(BITSELECT,279)@3
    rightShiftStage2Idx2Rng32_uid280_alignmentShifter_uid65_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(rightShiftStage1_uid276_alignmentShifter_uid65_block_rsrvd_fix_q(107 downto 32));

    -- rightShiftStage2Idx2_uid282_alignmentShifter_uid65_block_rsrvd_fix(BITJOIN,281)@3
    rightShiftStage2Idx2_uid282_alignmentShifter_uid65_block_rsrvd_fix_q <= zs_uid179_lzCountValAdd_uid81_block_rsrvd_fix_q & rightShiftStage2Idx2Rng32_uid280_alignmentShifter_uid65_block_rsrvd_fix_b;

    -- zs_uid187_lzCountValAdd_uid81_block_rsrvd_fix(CONSTANT,186)
    zs_uid187_lzCountValAdd_uid81_block_rsrvd_fix_q <= "0000000000000000";

    -- rightShiftStage2Idx1Rng16_uid277_alignmentShifter_uid65_block_rsrvd_fix(BITSELECT,276)@3
    rightShiftStage2Idx1Rng16_uid277_alignmentShifter_uid65_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(rightShiftStage1_uid276_alignmentShifter_uid65_block_rsrvd_fix_q(107 downto 16));

    -- rightShiftStage2Idx1_uid279_alignmentShifter_uid65_block_rsrvd_fix(BITJOIN,278)@3
    rightShiftStage2Idx1_uid279_alignmentShifter_uid65_block_rsrvd_fix_q <= zs_uid187_lzCountValAdd_uid81_block_rsrvd_fix_q & rightShiftStage2Idx1Rng16_uid277_alignmentShifter_uid65_block_rsrvd_fix_b;

    -- rightShiftStage1Idx3Pad12_uid273_alignmentShifter_uid65_block_rsrvd_fix(CONSTANT,272)
    rightShiftStage1Idx3Pad12_uid273_alignmentShifter_uid65_block_rsrvd_fix_q <= "000000000000";

    -- rightShiftStage1Idx3Rng12_uid272_alignmentShifter_uid65_block_rsrvd_fix(BITSELECT,271)@3
    rightShiftStage1Idx3Rng12_uid272_alignmentShifter_uid65_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(rightShiftStage0_uid265_alignmentShifter_uid65_block_rsrvd_fix_q(107 downto 12));

    -- rightShiftStage1Idx3_uid274_alignmentShifter_uid65_block_rsrvd_fix(BITJOIN,273)@3
    rightShiftStage1Idx3_uid274_alignmentShifter_uid65_block_rsrvd_fix_q <= rightShiftStage1Idx3Pad12_uid273_alignmentShifter_uid65_block_rsrvd_fix_q & rightShiftStage1Idx3Rng12_uid272_alignmentShifter_uid65_block_rsrvd_fix_b;

    -- zs_uid193_lzCountValAdd_uid81_block_rsrvd_fix(CONSTANT,192)
    zs_uid193_lzCountValAdd_uid81_block_rsrvd_fix_q <= "00000000";

    -- rightShiftStage1Idx2Rng8_uid269_alignmentShifter_uid65_block_rsrvd_fix(BITSELECT,268)@3
    rightShiftStage1Idx2Rng8_uid269_alignmentShifter_uid65_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(rightShiftStage0_uid265_alignmentShifter_uid65_block_rsrvd_fix_q(107 downto 8));

    -- rightShiftStage1Idx2_uid271_alignmentShifter_uid65_block_rsrvd_fix(BITJOIN,270)@3
    rightShiftStage1Idx2_uid271_alignmentShifter_uid65_block_rsrvd_fix_q <= zs_uid193_lzCountValAdd_uid81_block_rsrvd_fix_q & rightShiftStage1Idx2Rng8_uid269_alignmentShifter_uid65_block_rsrvd_fix_b;

    -- zs_uid199_lzCountValAdd_uid81_block_rsrvd_fix(CONSTANT,198)
    zs_uid199_lzCountValAdd_uid81_block_rsrvd_fix_q <= "0000";

    -- rightShiftStage1Idx1Rng4_uid266_alignmentShifter_uid65_block_rsrvd_fix(BITSELECT,265)@3
    rightShiftStage1Idx1Rng4_uid266_alignmentShifter_uid65_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(rightShiftStage0_uid265_alignmentShifter_uid65_block_rsrvd_fix_q(107 downto 4));

    -- rightShiftStage1Idx1_uid268_alignmentShifter_uid65_block_rsrvd_fix(BITJOIN,267)@3
    rightShiftStage1Idx1_uid268_alignmentShifter_uid65_block_rsrvd_fix_q <= zs_uid199_lzCountValAdd_uid81_block_rsrvd_fix_q & rightShiftStage1Idx1Rng4_uid266_alignmentShifter_uid65_block_rsrvd_fix_b;

    -- zv_uid74_block_rsrvd_fix(CONSTANT,73)
    zv_uid74_block_rsrvd_fix_q <= "000";

    -- rightShiftStage0Idx3Rng3_uid261_alignmentShifter_uid65_block_rsrvd_fix(BITSELECT,260)@3
    rightShiftStage0Idx3Rng3_uid261_alignmentShifter_uid65_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(rightPaddedIn_uid66_block_rsrvd_fix_q(107 downto 3));

    -- rightShiftStage0Idx3_uid263_alignmentShifter_uid65_block_rsrvd_fix(BITJOIN,262)@3
    rightShiftStage0Idx3_uid263_alignmentShifter_uid65_block_rsrvd_fix_q <= zv_uid74_block_rsrvd_fix_q & rightShiftStage0Idx3Rng3_uid261_alignmentShifter_uid65_block_rsrvd_fix_b;

    -- zs_uid205_lzCountValAdd_uid81_block_rsrvd_fix(CONSTANT,204)
    zs_uid205_lzCountValAdd_uid81_block_rsrvd_fix_q <= "00";

    -- rightShiftStage0Idx2Rng2_uid258_alignmentShifter_uid65_block_rsrvd_fix(BITSELECT,257)@3
    rightShiftStage0Idx2Rng2_uid258_alignmentShifter_uid65_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(rightPaddedIn_uid66_block_rsrvd_fix_q(107 downto 2));

    -- rightShiftStage0Idx2_uid260_alignmentShifter_uid65_block_rsrvd_fix(BITJOIN,259)@3
    rightShiftStage0Idx2_uid260_alignmentShifter_uid65_block_rsrvd_fix_q <= zs_uid205_lzCountValAdd_uid81_block_rsrvd_fix_q & rightShiftStage0Idx2Rng2_uid258_alignmentShifter_uid65_block_rsrvd_fix_b;

    -- rightShiftStage0Idx1Rng1_uid255_alignmentShifter_uid65_block_rsrvd_fix(BITSELECT,254)@3
    rightShiftStage0Idx1Rng1_uid255_alignmentShifter_uid65_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(rightPaddedIn_uid66_block_rsrvd_fix_q(107 downto 1));

    -- rightShiftStage0Idx1_uid257_alignmentShifter_uid65_block_rsrvd_fix(BITJOIN,256)@3
    rightShiftStage0Idx1_uid257_alignmentShifter_uid65_block_rsrvd_fix_q <= GND_q & rightShiftStage0Idx1Rng1_uid255_alignmentShifter_uid65_block_rsrvd_fix_b;

    -- InvExpXIsZero_uid40_block_rsrvd_fix(LOGICAL,39)@3
    InvExpXIsZero_uid40_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (redist32_excZ_sigb_uid13_uid33_block_rsrvd_fix_q_2_q));

    -- frac_sigb_uid32_block_rsrvd_fix(BITSELECT,31)@0
    frac_sigb_uid32_block_rsrvd_fix_in <= sigb_uid13_block_rsrvd_fix_q(51 downto 0);
    frac_sigb_uid32_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(frac_sigb_uid32_block_rsrvd_fix_in(51 downto 0));

    -- redist34_frac_sigb_uid32_block_rsrvd_fix_b_3(DELAY,412)
    redist34_frac_sigb_uid32_block_rsrvd_fix_b_3_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist34_frac_sigb_uid32_block_rsrvd_fix_b_3_delay_0 <= STD_LOGIC_VECTOR(frac_sigb_uid32_block_rsrvd_fix_b);
                redist34_frac_sigb_uid32_block_rsrvd_fix_b_3_delay_1 <= redist34_frac_sigb_uid32_block_rsrvd_fix_b_3_delay_0;
                redist34_frac_sigb_uid32_block_rsrvd_fix_b_3_q <= STD_LOGIC_VECTOR(redist34_frac_sigb_uid32_block_rsrvd_fix_b_3_delay_1);
            END IF;
        END IF;
    END PROCESS;

    -- oFracB_uid62_block_rsrvd_fix(BITJOIN,61)@3
    oFracB_uid62_block_rsrvd_fix_q <= InvExpXIsZero_uid40_block_rsrvd_fix_q & redist34_frac_sigb_uid32_block_rsrvd_fix_b_3_q;

    -- padConst_uid65_block_rsrvd_fix(CONSTANT,64)
    padConst_uid65_block_rsrvd_fix_q <= "0000000000000000000000000000000000000000000000000000000";

    -- rightPaddedIn_uid66_block_rsrvd_fix(BITJOIN,65)@3
    rightPaddedIn_uid66_block_rsrvd_fix_q <= oFracB_uid62_block_rsrvd_fix_q & padConst_uid65_block_rsrvd_fix_q;

    -- rightShiftStage0_uid265_alignmentShifter_uid65_block_rsrvd_fix(MUX,264)@3
    rightShiftStage0_uid265_alignmentShifter_uid65_block_rsrvd_fix_s <= rightShiftStageSel0Dto0_uid264_alignmentShifter_uid65_block_rsrvd_fix_bit_select_merged_b;
    rightShiftStage0_uid265_alignmentShifter_uid65_block_rsrvd_fix_combproc: PROCESS (rightShiftStage0_uid265_alignmentShifter_uid65_block_rsrvd_fix_s, rightPaddedIn_uid66_block_rsrvd_fix_q, rightShiftStage0Idx1_uid257_alignmentShifter_uid65_block_rsrvd_fix_q, rightShiftStage0Idx2_uid260_alignmentShifter_uid65_block_rsrvd_fix_q, rightShiftStage0Idx3_uid263_alignmentShifter_uid65_block_rsrvd_fix_q)
    BEGIN
        CASE (rightShiftStage0_uid265_alignmentShifter_uid65_block_rsrvd_fix_s) IS
            WHEN "00" => rightShiftStage0_uid265_alignmentShifter_uid65_block_rsrvd_fix_q <= rightPaddedIn_uid66_block_rsrvd_fix_q;
            WHEN "01" => rightShiftStage0_uid265_alignmentShifter_uid65_block_rsrvd_fix_q <= rightShiftStage0Idx1_uid257_alignmentShifter_uid65_block_rsrvd_fix_q;
            WHEN "10" => rightShiftStage0_uid265_alignmentShifter_uid65_block_rsrvd_fix_q <= rightShiftStage0Idx2_uid260_alignmentShifter_uid65_block_rsrvd_fix_q;
            WHEN "11" => rightShiftStage0_uid265_alignmentShifter_uid65_block_rsrvd_fix_q <= rightShiftStage0Idx3_uid263_alignmentShifter_uid65_block_rsrvd_fix_q;
            WHEN OTHERS => rightShiftStage0_uid265_alignmentShifter_uid65_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- rightShiftStage1_uid276_alignmentShifter_uid65_block_rsrvd_fix(MUX,275)@3
    rightShiftStage1_uid276_alignmentShifter_uid65_block_rsrvd_fix_s <= rightShiftStageSel0Dto0_uid264_alignmentShifter_uid65_block_rsrvd_fix_bit_select_merged_c;
    rightShiftStage1_uid276_alignmentShifter_uid65_block_rsrvd_fix_combproc: PROCESS (rightShiftStage1_uid276_alignmentShifter_uid65_block_rsrvd_fix_s, rightShiftStage0_uid265_alignmentShifter_uid65_block_rsrvd_fix_q, rightShiftStage1Idx1_uid268_alignmentShifter_uid65_block_rsrvd_fix_q, rightShiftStage1Idx2_uid271_alignmentShifter_uid65_block_rsrvd_fix_q, rightShiftStage1Idx3_uid274_alignmentShifter_uid65_block_rsrvd_fix_q)
    BEGIN
        CASE (rightShiftStage1_uid276_alignmentShifter_uid65_block_rsrvd_fix_s) IS
            WHEN "00" => rightShiftStage1_uid276_alignmentShifter_uid65_block_rsrvd_fix_q <= rightShiftStage0_uid265_alignmentShifter_uid65_block_rsrvd_fix_q;
            WHEN "01" => rightShiftStage1_uid276_alignmentShifter_uid65_block_rsrvd_fix_q <= rightShiftStage1Idx1_uid268_alignmentShifter_uid65_block_rsrvd_fix_q;
            WHEN "10" => rightShiftStage1_uid276_alignmentShifter_uid65_block_rsrvd_fix_q <= rightShiftStage1Idx2_uid271_alignmentShifter_uid65_block_rsrvd_fix_q;
            WHEN "11" => rightShiftStage1_uid276_alignmentShifter_uid65_block_rsrvd_fix_q <= rightShiftStage1Idx3_uid274_alignmentShifter_uid65_block_rsrvd_fix_q;
            WHEN OTHERS => rightShiftStage1_uid276_alignmentShifter_uid65_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- shiftOutConst_uid52_block_rsrvd_fix(CONSTANT,51)
    shiftOutConst_uid52_block_rsrvd_fix_q <= "110111";

    -- expAmExpB_uid51_block_rsrvd_fix(SUB,50)@1 + 1
    expAmExpB_uid51_block_rsrvd_fix_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("0" & redist40_exp_siga_uid17_block_rsrvd_fix_b_1_q));
    expAmExpB_uid51_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("0" & redist35_exp_sigb_uid31_block_rsrvd_fix_b_1_q));
    expAmExpB_uid51_block_rsrvd_fix_clkproc: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                expAmExpB_uid51_block_rsrvd_fix_o <= STD_LOGIC_VECTOR(SIGNED(expAmExpB_uid51_block_rsrvd_fix_a) - SIGNED(expAmExpB_uid51_block_rsrvd_fix_b));
            END IF;
        END IF;
    END PROCESS;
    expAmExpB_uid51_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(expAmExpB_uid51_block_rsrvd_fix_o(11 downto 0));

    -- expAmExpBShiftRange_uid59_block_rsrvd_fix(BITSELECT,58)@2
    expAmExpBShiftRange_uid59_block_rsrvd_fix_in <= expAmExpB_uid51_block_rsrvd_fix_q(5 downto 0);
    expAmExpBShiftRange_uid59_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(expAmExpBShiftRange_uid59_block_rsrvd_fix_in(5 downto 0));

    -- expBIsZero_uid55_block_rsrvd_fix(LOGICAL,54)@1 + 1
    expBIsZero_uid55_block_rsrvd_fix_qi <= "1" WHEN redist35_exp_sigb_uid31_block_rsrvd_fix_b_1_q /= "00000000000" ELSE "0";
    expBIsZero_uid55_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => expBIsZero_uid55_block_rsrvd_fix_qi, xout => expBIsZero_uid55_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- expBIsZero_uid56_block_rsrvd_fix(LOGICAL,55)@2
    expBIsZero_uid56_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (expBIsZero_uid55_block_rsrvd_fix_q));

    -- alignShiftMaxM1_uid53_block_rsrvd_fix(CONSTANT,52)
    alignShiftMaxM1_uid53_block_rsrvd_fix_q <= "110110";

    -- shiftedOut1_uid57_block_rsrvd_fix(COMPARE,56)@2
    shiftedOut1_uid57_block_rsrvd_fix_a <= STD_LOGIC_VECTOR("00000000" & alignShiftMaxM1_uid53_block_rsrvd_fix_q);
    shiftedOut1_uid57_block_rsrvd_fix_b <= STD_LOGIC_VECTOR("00" & expAmExpB_uid51_block_rsrvd_fix_q);
    shiftedOut1_uid57_block_rsrvd_fix_o <= STD_LOGIC_VECTOR(UNSIGNED(shiftedOut1_uid57_block_rsrvd_fix_a) - UNSIGNED(shiftedOut1_uid57_block_rsrvd_fix_b));
    shiftedOut1_uid57_block_rsrvd_fix_c(0) <= shiftedOut1_uid57_block_rsrvd_fix_o(13);

    -- shiftedOut_uid58_block_rsrvd_fix(LOGICAL,57)@2
    shiftedOut_uid58_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(shiftedOut1_uid57_block_rsrvd_fix_c or expBIsZero_uid56_block_rsrvd_fix_q);

    -- shiftValue_uid60_block_rsrvd_fix(MUX,59)@2 + 1
    shiftValue_uid60_block_rsrvd_fix_s <= shiftedOut_uid58_block_rsrvd_fix_q;
    shiftValue_uid60_block_rsrvd_fix_clkproc: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                CASE (shiftValue_uid60_block_rsrvd_fix_s) IS
                    WHEN "0" => shiftValue_uid60_block_rsrvd_fix_q <= expAmExpBShiftRange_uid59_block_rsrvd_fix_b;
                    WHEN "1" => shiftValue_uid60_block_rsrvd_fix_q <= shiftOutConst_uid52_block_rsrvd_fix_q;
                    WHEN OTHERS => shiftValue_uid60_block_rsrvd_fix_q <= (others => '0');
                END CASE;
            END IF;
        END IF;
    END PROCESS;

    -- rightShiftStageSel0Dto0_uid264_alignmentShifter_uid65_block_rsrvd_fix_bit_select_merged(BITSELECT,365)@3
    rightShiftStageSel0Dto0_uid264_alignmentShifter_uid65_block_rsrvd_fix_bit_select_merged_b <= STD_LOGIC_VECTOR(shiftValue_uid60_block_rsrvd_fix_q(1 downto 0));
    rightShiftStageSel0Dto0_uid264_alignmentShifter_uid65_block_rsrvd_fix_bit_select_merged_c <= STD_LOGIC_VECTOR(shiftValue_uid60_block_rsrvd_fix_q(3 downto 2));
    rightShiftStageSel0Dto0_uid264_alignmentShifter_uid65_block_rsrvd_fix_bit_select_merged_d <= STD_LOGIC_VECTOR(shiftValue_uid60_block_rsrvd_fix_q(5 downto 4));

    -- rightShiftStage2_uid287_alignmentShifter_uid65_block_rsrvd_fix(MUX,286)@3 + 1
    rightShiftStage2_uid287_alignmentShifter_uid65_block_rsrvd_fix_s <= rightShiftStageSel0Dto0_uid264_alignmentShifter_uid65_block_rsrvd_fix_bit_select_merged_d;
    rightShiftStage2_uid287_alignmentShifter_uid65_block_rsrvd_fix_clkproc: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                CASE (rightShiftStage2_uid287_alignmentShifter_uid65_block_rsrvd_fix_s) IS
                    WHEN "00" => rightShiftStage2_uid287_alignmentShifter_uid65_block_rsrvd_fix_q <= rightShiftStage1_uid276_alignmentShifter_uid65_block_rsrvd_fix_q;
                    WHEN "01" => rightShiftStage2_uid287_alignmentShifter_uid65_block_rsrvd_fix_q <= rightShiftStage2Idx1_uid279_alignmentShifter_uid65_block_rsrvd_fix_q;
                    WHEN "10" => rightShiftStage2_uid287_alignmentShifter_uid65_block_rsrvd_fix_q <= rightShiftStage2Idx2_uid282_alignmentShifter_uid65_block_rsrvd_fix_q;
                    WHEN "11" => rightShiftStage2_uid287_alignmentShifter_uid65_block_rsrvd_fix_q <= rightShiftStage2Idx3_uid285_alignmentShifter_uid65_block_rsrvd_fix_q;
                    WHEN OTHERS => rightShiftStage2_uid287_alignmentShifter_uid65_block_rsrvd_fix_q <= (others => '0');
                END CASE;
            END IF;
        END IF;
    END PROCESS;

    -- stickyBits_uid68_block_rsrvd_fix_bit_select_merged(BITSELECT,377)@4
    stickyBits_uid68_block_rsrvd_fix_bit_select_merged_b <= STD_LOGIC_VECTOR(rightShiftStage2_uid287_alignmentShifter_uid65_block_rsrvd_fix_q(52 downto 0));
    stickyBits_uid68_block_rsrvd_fix_bit_select_merged_c <= STD_LOGIC_VECTOR(rightShiftStage2_uid287_alignmentShifter_uid65_block_rsrvd_fix_q(107 downto 53));

    -- redist0_stickyBits_uid68_block_rsrvd_fix_bit_select_merged_c_1(DELAY,378)
    redist0_stickyBits_uid68_block_rsrvd_fix_bit_select_merged_c_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist0_stickyBits_uid68_block_rsrvd_fix_bit_select_merged_c_1_q <= stickyBits_uid68_block_rsrvd_fix_bit_select_merged_c;
            END IF;
        END IF;
    END PROCESS;

    -- cstZeroWF_uid15_block_rsrvd_fix(CONSTANT,14)
    cstZeroWF_uid15_block_rsrvd_fix_q <= "0000000000000000000000000000000000000000000000000000";

    -- cmpStickyWZero_uid70_block_rsrvd_fix(LOGICAL,69)@4 + 1
    cmpStickyWZero_uid70_block_rsrvd_fix_b <= STD_LOGIC_VECTOR("0" & cstZeroWF_uid15_block_rsrvd_fix_q);
    cmpStickyWZero_uid70_block_rsrvd_fix_qi <= "1" WHEN stickyBits_uid68_block_rsrvd_fix_bit_select_merged_b = cmpStickyWZero_uid70_block_rsrvd_fix_b ELSE "0";
    cmpStickyWZero_uid70_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => cmpStickyWZero_uid70_block_rsrvd_fix_qi, xout => cmpStickyWZero_uid70_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- sticky_uid71_block_rsrvd_fix(LOGICAL,70)@5
    sticky_uid71_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (cmpStickyWZero_uid70_block_rsrvd_fix_q));

    -- alignFracB_uid73_block_rsrvd_fix(BITJOIN,72)@5
    alignFracB_uid73_block_rsrvd_fix_q <= redist0_stickyBits_uid68_block_rsrvd_fix_bit_select_merged_c_1_q & sticky_uid71_block_rsrvd_fix_q;

    -- fracBOp_uid76_block_rsrvd_fix(BITJOIN,75)@5
    fracBOp_uid76_block_rsrvd_fix_q <= GND_q & GND_q & alignFracB_uid73_block_rsrvd_fix_q;

    -- frac_siga_uid18_block_rsrvd_fix(BITSELECT,17)@0
    frac_siga_uid18_block_rsrvd_fix_in <= siga_uid12_block_rsrvd_fix_q(51 downto 0);
    frac_siga_uid18_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(frac_siga_uid18_block_rsrvd_fix_in(51 downto 0));

    -- redist39_frac_siga_uid18_block_rsrvd_fix_b_5_inputreg0(DELAY,421)
    redist39_frac_siga_uid18_block_rsrvd_fix_b_5_inputreg0_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist39_frac_siga_uid18_block_rsrvd_fix_b_5_inputreg0_q <= frac_siga_uid18_block_rsrvd_fix_b;
            END IF;
        END IF;
    END PROCESS;

    -- redist39_frac_siga_uid18_block_rsrvd_fix_b_5(DELAY,417)
    redist39_frac_siga_uid18_block_rsrvd_fix_b_5_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist39_frac_siga_uid18_block_rsrvd_fix_b_5_delay_0 <= STD_LOGIC_VECTOR(redist39_frac_siga_uid18_block_rsrvd_fix_b_5_inputreg0_q);
                redist39_frac_siga_uid18_block_rsrvd_fix_b_5_delay_1 <= redist39_frac_siga_uid18_block_rsrvd_fix_b_5_delay_0;
                redist39_frac_siga_uid18_block_rsrvd_fix_b_5_q <= STD_LOGIC_VECTOR(redist39_frac_siga_uid18_block_rsrvd_fix_b_5_delay_1);
            END IF;
        END IF;
    END PROCESS;

    -- redist39_frac_siga_uid18_block_rsrvd_fix_b_5_outputreg0(DELAY,422)
    redist39_frac_siga_uid18_block_rsrvd_fix_b_5_outputreg0_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist39_frac_siga_uid18_block_rsrvd_fix_b_5_outputreg0_q <= redist39_frac_siga_uid18_block_rsrvd_fix_b_5_q;
            END IF;
        END IF;
    END PROCESS;

    -- oFracA_uid63_block_rsrvd_fix(BITJOIN,62)@5
    oFracA_uid63_block_rsrvd_fix_q <= VCC_q & redist39_frac_siga_uid18_block_rsrvd_fix_b_5_outputreg0_q;

    -- fracAOp_uid75_block_rsrvd_fix(BITJOIN,74)@5
    fracAOp_uid75_block_rsrvd_fix_q <= oFracA_uid63_block_rsrvd_fix_q & zv_uid74_block_rsrvd_fix_q;

    -- fracResSub_uid78_block_rsrvd_fix(SUB,77)@5
    fracResSub_uid78_block_rsrvd_fix_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("000" & fracAOp_uid75_block_rsrvd_fix_q));
    fracResSub_uid78_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("0" & fracBOp_uid76_block_rsrvd_fix_q));
    fracResSub_uid78_block_rsrvd_fix_o <= STD_LOGIC_VECTOR(SIGNED(fracResSub_uid78_block_rsrvd_fix_a) - SIGNED(fracResSub_uid78_block_rsrvd_fix_b));
    fracResSub_uid78_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(fracResSub_uid78_block_rsrvd_fix_o(58 downto 0));

    -- fracResSubNoSignExt_uid80_block_rsrvd_fix(BITSELECT,79)@5
    fracResSubNoSignExt_uid80_block_rsrvd_fix_in <= fracResSub_uid78_block_rsrvd_fix_q(56 downto 0);
    fracResSubNoSignExt_uid80_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(fracResSubNoSignExt_uid80_block_rsrvd_fix_in(56 downto 0));

    -- redist22_fracResSubNoSignExt_uid80_block_rsrvd_fix_b_1(DELAY,400)
    redist22_fracResSubNoSignExt_uid80_block_rsrvd_fix_b_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist22_fracResSubNoSignExt_uid80_block_rsrvd_fix_b_1_q <= fracResSubNoSignExt_uid80_block_rsrvd_fix_b;
            END IF;
        END IF;
    END PROCESS;

    -- rVStage_uid217_lzCountValSub_uid83_block_rsrvd_fix(BITSELECT,216)@6
    rVStage_uid217_lzCountValSub_uid83_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(redist22_fracResSubNoSignExt_uid80_block_rsrvd_fix_b_1_q(56 downto 25));

    -- vCount_uid218_lzCountValSub_uid83_block_rsrvd_fix(LOGICAL,217)@6
    vCount_uid218_lzCountValSub_uid83_block_rsrvd_fix_q <= "1" WHEN rVStage_uid217_lzCountValSub_uid83_block_rsrvd_fix_b = zs_uid179_lzCountValAdd_uid81_block_rsrvd_fix_q ELSE "0";

    -- redist9_vCount_uid218_lzCountValSub_uid83_block_rsrvd_fix_q_2(DELAY,387)
    redist9_vCount_uid218_lzCountValSub_uid83_block_rsrvd_fix_q_2_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist9_vCount_uid218_lzCountValSub_uid83_block_rsrvd_fix_q_2_delay_0 <= STD_LOGIC_VECTOR(vCount_uid218_lzCountValSub_uid83_block_rsrvd_fix_q);
                redist9_vCount_uid218_lzCountValSub_uid83_block_rsrvd_fix_q_2_q <= STD_LOGIC_VECTOR(redist9_vCount_uid218_lzCountValSub_uid83_block_rsrvd_fix_q_2_delay_0);
            END IF;
        END IF;
    END PROCESS;

    -- vStage_uid220_lzCountValSub_uid83_block_rsrvd_fix(BITSELECT,219)@6
    vStage_uid220_lzCountValSub_uid83_block_rsrvd_fix_in <= redist22_fracResSubNoSignExt_uid80_block_rsrvd_fix_b_1_q(24 downto 0);
    vStage_uid220_lzCountValSub_uid83_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(vStage_uid220_lzCountValSub_uid83_block_rsrvd_fix_in(24 downto 0));

    -- mO_uid182_lzCountValAdd_uid81_block_rsrvd_fix(CONSTANT,181)
    mO_uid182_lzCountValAdd_uid81_block_rsrvd_fix_q <= "1111111";

    -- cStage_uid221_lzCountValSub_uid83_block_rsrvd_fix(BITJOIN,220)@6
    cStage_uid221_lzCountValSub_uid83_block_rsrvd_fix_q <= vStage_uid220_lzCountValSub_uid83_block_rsrvd_fix_b & mO_uid182_lzCountValAdd_uid81_block_rsrvd_fix_q;

    -- vStagei_uid223_lzCountValSub_uid83_block_rsrvd_fix(MUX,222)@6
    vStagei_uid223_lzCountValSub_uid83_block_rsrvd_fix_s <= vCount_uid218_lzCountValSub_uid83_block_rsrvd_fix_q;
    vStagei_uid223_lzCountValSub_uid83_block_rsrvd_fix_combproc: PROCESS (vStagei_uid223_lzCountValSub_uid83_block_rsrvd_fix_s, rVStage_uid217_lzCountValSub_uid83_block_rsrvd_fix_b, cStage_uid221_lzCountValSub_uid83_block_rsrvd_fix_q)
    BEGIN
        CASE (vStagei_uid223_lzCountValSub_uid83_block_rsrvd_fix_s) IS
            WHEN "0" => vStagei_uid223_lzCountValSub_uid83_block_rsrvd_fix_q <= rVStage_uid217_lzCountValSub_uid83_block_rsrvd_fix_b;
            WHEN "1" => vStagei_uid223_lzCountValSub_uid83_block_rsrvd_fix_q <= cStage_uid221_lzCountValSub_uid83_block_rsrvd_fix_q;
            WHEN OTHERS => vStagei_uid223_lzCountValSub_uid83_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- rVStage_uid225_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged(BITSELECT,372)@6
    rVStage_uid225_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b <= STD_LOGIC_VECTOR(vStagei_uid223_lzCountValSub_uid83_block_rsrvd_fix_q(31 downto 16));
    rVStage_uid225_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_c <= STD_LOGIC_VECTOR(vStagei_uid223_lzCountValSub_uid83_block_rsrvd_fix_q(15 downto 0));

    -- vCount_uid226_lzCountValSub_uid83_block_rsrvd_fix(LOGICAL,225)@6
    vCount_uid226_lzCountValSub_uid83_block_rsrvd_fix_q <= "1" WHEN rVStage_uid225_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b = zs_uid187_lzCountValAdd_uid81_block_rsrvd_fix_q ELSE "0";

    -- redist7_vCount_uid226_lzCountValSub_uid83_block_rsrvd_fix_q_2(DELAY,385)
    redist7_vCount_uid226_lzCountValSub_uid83_block_rsrvd_fix_q_2_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist7_vCount_uid226_lzCountValSub_uid83_block_rsrvd_fix_q_2_delay_0 <= STD_LOGIC_VECTOR(vCount_uid226_lzCountValSub_uid83_block_rsrvd_fix_q);
                redist7_vCount_uid226_lzCountValSub_uid83_block_rsrvd_fix_q_2_q <= STD_LOGIC_VECTOR(redist7_vCount_uid226_lzCountValSub_uid83_block_rsrvd_fix_q_2_delay_0);
            END IF;
        END IF;
    END PROCESS;

    -- vStagei_uid229_lzCountValSub_uid83_block_rsrvd_fix(MUX,228)@6 + 1
    vStagei_uid229_lzCountValSub_uid83_block_rsrvd_fix_s <= vCount_uid226_lzCountValSub_uid83_block_rsrvd_fix_q;
    vStagei_uid229_lzCountValSub_uid83_block_rsrvd_fix_clkproc: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                CASE (vStagei_uid229_lzCountValSub_uid83_block_rsrvd_fix_s) IS
                    WHEN "0" => vStagei_uid229_lzCountValSub_uid83_block_rsrvd_fix_q <= rVStage_uid225_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b;
                    WHEN "1" => vStagei_uid229_lzCountValSub_uid83_block_rsrvd_fix_q <= rVStage_uid225_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_c;
                    WHEN OTHERS => vStagei_uid229_lzCountValSub_uid83_block_rsrvd_fix_q <= (others => '0');
                END CASE;
            END IF;
        END IF;
    END PROCESS;

    -- rVStage_uid231_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged(BITSELECT,373)@7
    rVStage_uid231_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b <= STD_LOGIC_VECTOR(vStagei_uid229_lzCountValSub_uid83_block_rsrvd_fix_q(15 downto 8));
    rVStage_uid231_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_c <= STD_LOGIC_VECTOR(vStagei_uid229_lzCountValSub_uid83_block_rsrvd_fix_q(7 downto 0));

    -- vCount_uid232_lzCountValSub_uid83_block_rsrvd_fix(LOGICAL,231)@7
    vCount_uid232_lzCountValSub_uid83_block_rsrvd_fix_q <= "1" WHEN rVStage_uid231_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b = zs_uid193_lzCountValAdd_uid81_block_rsrvd_fix_q ELSE "0";

    -- redist6_vCount_uid232_lzCountValSub_uid83_block_rsrvd_fix_q_1(DELAY,384)
    redist6_vCount_uid232_lzCountValSub_uid83_block_rsrvd_fix_q_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist6_vCount_uid232_lzCountValSub_uid83_block_rsrvd_fix_q_1_q <= vCount_uid232_lzCountValSub_uid83_block_rsrvd_fix_q;
            END IF;
        END IF;
    END PROCESS;

    -- vStagei_uid235_lzCountValSub_uid83_block_rsrvd_fix(MUX,234)@7
    vStagei_uid235_lzCountValSub_uid83_block_rsrvd_fix_s <= vCount_uid232_lzCountValSub_uid83_block_rsrvd_fix_q;
    vStagei_uid235_lzCountValSub_uid83_block_rsrvd_fix_combproc: PROCESS (vStagei_uid235_lzCountValSub_uid83_block_rsrvd_fix_s, rVStage_uid231_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b, rVStage_uid231_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_c)
    BEGIN
        CASE (vStagei_uid235_lzCountValSub_uid83_block_rsrvd_fix_s) IS
            WHEN "0" => vStagei_uid235_lzCountValSub_uid83_block_rsrvd_fix_q <= rVStage_uid231_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b;
            WHEN "1" => vStagei_uid235_lzCountValSub_uid83_block_rsrvd_fix_q <= rVStage_uid231_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_c;
            WHEN OTHERS => vStagei_uid235_lzCountValSub_uid83_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- rVStage_uid237_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged(BITSELECT,374)@7
    rVStage_uid237_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b <= STD_LOGIC_VECTOR(vStagei_uid235_lzCountValSub_uid83_block_rsrvd_fix_q(7 downto 4));
    rVStage_uid237_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_c <= STD_LOGIC_VECTOR(vStagei_uid235_lzCountValSub_uid83_block_rsrvd_fix_q(3 downto 0));

    -- vCount_uid238_lzCountValSub_uid83_block_rsrvd_fix(LOGICAL,237)@7
    vCount_uid238_lzCountValSub_uid83_block_rsrvd_fix_q <= "1" WHEN rVStage_uid237_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b = zs_uid199_lzCountValAdd_uid81_block_rsrvd_fix_q ELSE "0";

    -- redist5_vCount_uid238_lzCountValSub_uid83_block_rsrvd_fix_q_1(DELAY,383)
    redist5_vCount_uid238_lzCountValSub_uid83_block_rsrvd_fix_q_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist5_vCount_uid238_lzCountValSub_uid83_block_rsrvd_fix_q_1_q <= vCount_uid238_lzCountValSub_uid83_block_rsrvd_fix_q;
            END IF;
        END IF;
    END PROCESS;

    -- vStagei_uid241_lzCountValSub_uid83_block_rsrvd_fix(MUX,240)@7
    vStagei_uid241_lzCountValSub_uid83_block_rsrvd_fix_s <= vCount_uid238_lzCountValSub_uid83_block_rsrvd_fix_q;
    vStagei_uid241_lzCountValSub_uid83_block_rsrvd_fix_combproc: PROCESS (vStagei_uid241_lzCountValSub_uid83_block_rsrvd_fix_s, rVStage_uid237_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b, rVStage_uid237_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_c)
    BEGIN
        CASE (vStagei_uid241_lzCountValSub_uid83_block_rsrvd_fix_s) IS
            WHEN "0" => vStagei_uid241_lzCountValSub_uid83_block_rsrvd_fix_q <= rVStage_uid237_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b;
            WHEN "1" => vStagei_uid241_lzCountValSub_uid83_block_rsrvd_fix_q <= rVStage_uid237_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_c;
            WHEN OTHERS => vStagei_uid241_lzCountValSub_uid83_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- rVStage_uid243_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged(BITSELECT,375)@7
    rVStage_uid243_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b <= STD_LOGIC_VECTOR(vStagei_uid241_lzCountValSub_uid83_block_rsrvd_fix_q(3 downto 2));
    rVStage_uid243_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_c <= STD_LOGIC_VECTOR(vStagei_uid241_lzCountValSub_uid83_block_rsrvd_fix_q(1 downto 0));

    -- vCount_uid244_lzCountValSub_uid83_block_rsrvd_fix(LOGICAL,243)@7 + 1
    vCount_uid244_lzCountValSub_uid83_block_rsrvd_fix_qi <= "1" WHEN rVStage_uid243_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b = zs_uid205_lzCountValAdd_uid81_block_rsrvd_fix_q ELSE "0";
    vCount_uid244_lzCountValSub_uid83_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => vCount_uid244_lzCountValSub_uid83_block_rsrvd_fix_qi, xout => vCount_uid244_lzCountValSub_uid83_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- redist2_rVStage_uid243_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_c_1(DELAY,380)
    redist2_rVStage_uid243_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_c_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist2_rVStage_uid243_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_c_1_q <= rVStage_uid243_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_c;
            END IF;
        END IF;
    END PROCESS;

    -- redist1_rVStage_uid243_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b_1(DELAY,379)
    redist1_rVStage_uid243_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist1_rVStage_uid243_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b_1_q <= rVStage_uid243_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b;
            END IF;
        END IF;
    END PROCESS;

    -- vStagei_uid247_lzCountValSub_uid83_block_rsrvd_fix(MUX,246)@8
    vStagei_uid247_lzCountValSub_uid83_block_rsrvd_fix_s <= vCount_uid244_lzCountValSub_uid83_block_rsrvd_fix_q;
    vStagei_uid247_lzCountValSub_uid83_block_rsrvd_fix_combproc: PROCESS (vStagei_uid247_lzCountValSub_uid83_block_rsrvd_fix_s, redist1_rVStage_uid243_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b_1_q, redist2_rVStage_uid243_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_c_1_q)
    BEGIN
        CASE (vStagei_uid247_lzCountValSub_uid83_block_rsrvd_fix_s) IS
            WHEN "0" => vStagei_uid247_lzCountValSub_uid83_block_rsrvd_fix_q <= redist1_rVStage_uid243_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_b_1_q;
            WHEN "1" => vStagei_uid247_lzCountValSub_uid83_block_rsrvd_fix_q <= redist2_rVStage_uid243_lzCountValSub_uid83_block_rsrvd_fix_bit_select_merged_c_1_q;
            WHEN OTHERS => vStagei_uid247_lzCountValSub_uid83_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- rVStage_uid249_lzCountValSub_uid83_block_rsrvd_fix(BITSELECT,248)@8
    rVStage_uid249_lzCountValSub_uid83_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(vStagei_uid247_lzCountValSub_uid83_block_rsrvd_fix_q(1 downto 1));

    -- vCount_uid250_lzCountValSub_uid83_block_rsrvd_fix(LOGICAL,249)@8
    vCount_uid250_lzCountValSub_uid83_block_rsrvd_fix_q <= "1" WHEN rVStage_uid249_lzCountValSub_uid83_block_rsrvd_fix_b = GND_q ELSE "0";

    -- r_uid251_lzCountValSub_uid83_block_rsrvd_fix(BITJOIN,250)@8
    r_uid251_lzCountValSub_uid83_block_rsrvd_fix_q <= redist9_vCount_uid218_lzCountValSub_uid83_block_rsrvd_fix_q_2_q & redist7_vCount_uid226_lzCountValSub_uid83_block_rsrvd_fix_q_2_q & redist6_vCount_uid232_lzCountValSub_uid83_block_rsrvd_fix_q_1_q & redist5_vCount_uid238_lzCountValSub_uid83_block_rsrvd_fix_q_1_q & vCount_uid244_lzCountValSub_uid83_block_rsrvd_fix_q & vCount_uid250_lzCountValSub_uid83_block_rsrvd_fix_q;

    -- aMinusA_uid86_block_rsrvd_fix(LOGICAL,85)@8 + 1
    aMinusA_uid86_block_rsrvd_fix_qi <= "1" WHEN r_uid251_lzCountValSub_uid83_block_rsrvd_fix_q = cAmA_uid85_block_rsrvd_fix_q ELSE "0";
    aMinusA_uid86_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => aMinusA_uid86_block_rsrvd_fix_qi, xout => aMinusA_uid86_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- redist21_aMinusA_uid86_block_rsrvd_fix_q_2(DELAY,399)
    redist21_aMinusA_uid86_block_rsrvd_fix_q_2_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist21_aMinusA_uid86_block_rsrvd_fix_q_2_q <= aMinusA_uid86_block_rsrvd_fix_q;
            END IF;
        END IF;
    END PROCESS;

    -- cstAllOWE_uid14_block_rsrvd_fix(CONSTANT,13)
    cstAllOWE_uid14_block_rsrvd_fix_q <= "11111111111";

    -- expXIsMax_uid34_block_rsrvd_fix(LOGICAL,33)@1 + 1
    expXIsMax_uid34_block_rsrvd_fix_qi <= "1" WHEN redist35_exp_sigb_uid31_block_rsrvd_fix_b_1_q = cstAllOWE_uid14_block_rsrvd_fix_q ELSE "0";
    expXIsMax_uid34_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => expXIsMax_uid34_block_rsrvd_fix_qi, xout => expXIsMax_uid34_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- redist31_expXIsMax_uid34_block_rsrvd_fix_q_6(DELAY,409)
    redist31_expXIsMax_uid34_block_rsrvd_fix_q_6_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist31_expXIsMax_uid34_block_rsrvd_fix_q_6_delay_0 <= STD_LOGIC_VECTOR(expXIsMax_uid34_block_rsrvd_fix_q);
                redist31_expXIsMax_uid34_block_rsrvd_fix_q_6_delay_1 <= redist31_expXIsMax_uid34_block_rsrvd_fix_q_6_delay_0;
                redist31_expXIsMax_uid34_block_rsrvd_fix_q_6_delay_2 <= redist31_expXIsMax_uid34_block_rsrvd_fix_q_6_delay_1;
                redist31_expXIsMax_uid34_block_rsrvd_fix_q_6_delay_3 <= redist31_expXIsMax_uid34_block_rsrvd_fix_q_6_delay_2;
                redist31_expXIsMax_uid34_block_rsrvd_fix_q_6_q <= STD_LOGIC_VECTOR(redist31_expXIsMax_uid34_block_rsrvd_fix_q_6_delay_3);
            END IF;
        END IF;
    END PROCESS;

    -- invExpXIsMax_uid39_block_rsrvd_fix(LOGICAL,38)@7
    invExpXIsMax_uid39_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (redist31_expXIsMax_uid34_block_rsrvd_fix_q_6_q));

    -- redist28_InvExpXIsZero_uid40_block_rsrvd_fix_q_4(DELAY,406)
    redist28_InvExpXIsZero_uid40_block_rsrvd_fix_q_4_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist28_InvExpXIsZero_uid40_block_rsrvd_fix_q_4_delay_0 <= STD_LOGIC_VECTOR(InvExpXIsZero_uid40_block_rsrvd_fix_q);
                redist28_InvExpXIsZero_uid40_block_rsrvd_fix_q_4_delay_1 <= redist28_InvExpXIsZero_uid40_block_rsrvd_fix_q_4_delay_0;
                redist28_InvExpXIsZero_uid40_block_rsrvd_fix_q_4_delay_2 <= redist28_InvExpXIsZero_uid40_block_rsrvd_fix_q_4_delay_1;
                redist28_InvExpXIsZero_uid40_block_rsrvd_fix_q_4_q <= STD_LOGIC_VECTOR(redist28_InvExpXIsZero_uid40_block_rsrvd_fix_q_4_delay_2);
            END IF;
        END IF;
    END PROCESS;

    -- excR_sigb_uid41_block_rsrvd_fix(LOGICAL,40)@7
    excR_sigb_uid41_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(redist28_InvExpXIsZero_uid40_block_rsrvd_fix_q_4_q and invExpXIsMax_uid39_block_rsrvd_fix_q);

    -- expXIsMax_uid20_block_rsrvd_fix(LOGICAL,19)@7
    expXIsMax_uid20_block_rsrvd_fix_q <= "1" WHEN redist41_exp_siga_uid17_block_rsrvd_fix_b_7_mem_q = cstAllOWE_uid14_block_rsrvd_fix_q ELSE "0";

    -- invExpXIsMax_uid25_block_rsrvd_fix(LOGICAL,24)@7
    invExpXIsMax_uid25_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (expXIsMax_uid20_block_rsrvd_fix_q));

    -- InvExpXIsZero_uid26_block_rsrvd_fix(LOGICAL,25)@7
    InvExpXIsZero_uid26_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (excZ_siga_uid12_uid19_block_rsrvd_fix_q));

    -- excR_siga_uid27_block_rsrvd_fix(LOGICAL,26)@7
    excR_siga_uid27_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(InvExpXIsZero_uid26_block_rsrvd_fix_q and invExpXIsMax_uid25_block_rsrvd_fix_q);

    -- regInputs_uid124_block_rsrvd_fix(LOGICAL,123)@7 + 1
    regInputs_uid124_block_rsrvd_fix_qi <= excR_siga_uid27_block_rsrvd_fix_q and excR_sigb_uid41_block_rsrvd_fix_q;
    regInputs_uid124_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => regInputs_uid124_block_rsrvd_fix_qi, xout => regInputs_uid124_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- redist16_regInputs_uid124_block_rsrvd_fix_q_3(DELAY,394)
    redist16_regInputs_uid124_block_rsrvd_fix_q_3_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist16_regInputs_uid124_block_rsrvd_fix_q_3_delay_0 <= STD_LOGIC_VECTOR(regInputs_uid124_block_rsrvd_fix_q);
                redist16_regInputs_uid124_block_rsrvd_fix_q_3_q <= STD_LOGIC_VECTOR(redist16_regInputs_uid124_block_rsrvd_fix_q_3_delay_0);
            END IF;
        END IF;
    END PROCESS;

    -- aMa_uid171_block_rsrvd_fix(LOGICAL,170)@10
    aMa_uid171_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(redist16_regInputs_uid124_block_rsrvd_fix_q_3_q and redist21_aMinusA_uid86_block_rsrvd_fix_q_2_q and invEffSub_uid134_block_rsrvd_fix_q);

    -- invAMASub_uid172_block_rsrvd_fix(LOGICAL,171)@10
    invAMASub_uid172_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (aMa_uid171_block_rsrvd_fix_q));

    -- swap_uid11_block_rsrvd_fix(LOGICAL,10)@0 + 1
    swap_uid11_block_rsrvd_fix_qi <= not (xGTEy_uid10_block_rsrvd_fix_n);
    swap_uid11_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => swap_uid11_block_rsrvd_fix_qi, xout => swap_uid11_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- redist42_swap_uid11_block_rsrvd_fix_q_10(DELAY,420)
    redist42_swap_uid11_block_rsrvd_fix_q_10 : dspba_delay
    GENERIC MAP ( width => 1, depth => 9, reset_kind => "NONE", phase => 0, modulus => 1024 )
    PORT MAP ( xin => swap_uid11_block_rsrvd_fix_q, xout => redist42_swap_uid11_block_rsrvd_fix_q_10_q, clk => clk, aclr => areset, ena => '1' );

    -- signASwap_uid173_block_rsrvd_fix(LOGICAL,172)@10
    signASwap_uid173_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(redist27_sigA_uid46_block_rsrvd_fix_b_10_q xor redist42_swap_uid11_block_rsrvd_fix_q_10_q);

    -- fracXIsZero_uid35_block_rsrvd_fix(LOGICAL,34)@3 + 1
    fracXIsZero_uid35_block_rsrvd_fix_qi <= "1" WHEN cstZeroWF_uid15_block_rsrvd_fix_q = redist34_frac_sigb_uid32_block_rsrvd_fix_b_3_q ELSE "0";
    fracXIsZero_uid35_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => fracXIsZero_uid35_block_rsrvd_fix_qi, xout => fracXIsZero_uid35_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- redist30_fracXIsZero_uid35_block_rsrvd_fix_q_4(DELAY,408)
    redist30_fracXIsZero_uid35_block_rsrvd_fix_q_4_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist30_fracXIsZero_uid35_block_rsrvd_fix_q_4_delay_0 <= STD_LOGIC_VECTOR(fracXIsZero_uid35_block_rsrvd_fix_q);
                redist30_fracXIsZero_uid35_block_rsrvd_fix_q_4_delay_1 <= redist30_fracXIsZero_uid35_block_rsrvd_fix_q_4_delay_0;
                redist30_fracXIsZero_uid35_block_rsrvd_fix_q_4_q <= STD_LOGIC_VECTOR(redist30_fracXIsZero_uid35_block_rsrvd_fix_q_4_delay_1);
            END IF;
        END IF;
    END PROCESS;

    -- fracXIsNotZero_uid36_block_rsrvd_fix(LOGICAL,35)@7
    fracXIsNotZero_uid36_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (redist30_fracXIsZero_uid35_block_rsrvd_fix_q_4_q));

    -- excN_sigb_uid38_block_rsrvd_fix(LOGICAL,37)@7
    excN_sigb_uid38_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(redist31_expXIsMax_uid34_block_rsrvd_fix_q_6_q and fracXIsNotZero_uid36_block_rsrvd_fix_q);

    -- fracXIsZero_uid21_block_rsrvd_fix(LOGICAL,20)@5 + 1
    fracXIsZero_uid21_block_rsrvd_fix_qi <= "1" WHEN cstZeroWF_uid15_block_rsrvd_fix_q = redist39_frac_siga_uid18_block_rsrvd_fix_b_5_outputreg0_q ELSE "0";
    fracXIsZero_uid21_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => fracXIsZero_uid21_block_rsrvd_fix_qi, xout => fracXIsZero_uid21_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- redist37_fracXIsZero_uid21_block_rsrvd_fix_q_2(DELAY,415)
    redist37_fracXIsZero_uid21_block_rsrvd_fix_q_2_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist37_fracXIsZero_uid21_block_rsrvd_fix_q_2_q <= fracXIsZero_uid21_block_rsrvd_fix_q;
            END IF;
        END IF;
    END PROCESS;

    -- fracXIsNotZero_uid22_block_rsrvd_fix(LOGICAL,21)@7
    fracXIsNotZero_uid22_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (redist37_fracXIsZero_uid21_block_rsrvd_fix_q_2_q));

    -- excN_siga_uid24_block_rsrvd_fix(LOGICAL,23)@7
    excN_siga_uid24_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(expXIsMax_uid20_block_rsrvd_fix_q and fracXIsNotZero_uid22_block_rsrvd_fix_q);

    -- oneIsNaN_uid128_block_rsrvd_fix(LOGICAL,127)@7 + 1
    oneIsNaN_uid128_block_rsrvd_fix_qi <= excN_siga_uid24_block_rsrvd_fix_q or excN_sigb_uid38_block_rsrvd_fix_q;
    oneIsNaN_uid128_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => oneIsNaN_uid128_block_rsrvd_fix_qi, xout => oneIsNaN_uid128_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- redist15_oneIsNaN_uid128_block_rsrvd_fix_q_3(DELAY,393)
    redist15_oneIsNaN_uid128_block_rsrvd_fix_q_3_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist15_oneIsNaN_uid128_block_rsrvd_fix_q_3_delay_0 <= STD_LOGIC_VECTOR(oneIsNaN_uid128_block_rsrvd_fix_q);
                redist15_oneIsNaN_uid128_block_rsrvd_fix_q_3_q <= STD_LOGIC_VECTOR(redist15_oneIsNaN_uid128_block_rsrvd_fix_q_3_delay_0);
            END IF;
        END IF;
    END PROCESS;

    -- excI_sigb_uid37_block_rsrvd_fix(LOGICAL,36)@7 + 1
    excI_sigb_uid37_block_rsrvd_fix_qi <= redist31_expXIsMax_uid34_block_rsrvd_fix_q_6_q and redist30_fracXIsZero_uid35_block_rsrvd_fix_q_4_q;
    excI_sigb_uid37_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => excI_sigb_uid37_block_rsrvd_fix_qi, xout => excI_sigb_uid37_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- redist29_excI_sigb_uid37_block_rsrvd_fix_q_3(DELAY,407)
    redist29_excI_sigb_uid37_block_rsrvd_fix_q_3_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist29_excI_sigb_uid37_block_rsrvd_fix_q_3_delay_0 <= STD_LOGIC_VECTOR(excI_sigb_uid37_block_rsrvd_fix_q);
                redist29_excI_sigb_uid37_block_rsrvd_fix_q_3_q <= STD_LOGIC_VECTOR(redist29_excI_sigb_uid37_block_rsrvd_fix_q_3_delay_0);
            END IF;
        END IF;
    END PROCESS;

    -- excI_siga_uid23_block_rsrvd_fix(LOGICAL,22)@7 + 1
    excI_siga_uid23_block_rsrvd_fix_qi <= expXIsMax_uid20_block_rsrvd_fix_q and redist37_fracXIsZero_uid21_block_rsrvd_fix_q_2_q;
    excI_siga_uid23_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => excI_siga_uid23_block_rsrvd_fix_qi, xout => excI_siga_uid23_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- redist36_excI_siga_uid23_block_rsrvd_fix_q_3(DELAY,414)
    redist36_excI_siga_uid23_block_rsrvd_fix_q_3_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist36_excI_siga_uid23_block_rsrvd_fix_q_3_delay_0 <= STD_LOGIC_VECTOR(excI_siga_uid23_block_rsrvd_fix_q);
                redist36_excI_siga_uid23_block_rsrvd_fix_q_3_q <= STD_LOGIC_VECTOR(redist36_excI_siga_uid23_block_rsrvd_fix_q_3_delay_0);
            END IF;
        END IF;
    END PROCESS;

    -- infPinfForSub_uid135_block_rsrvd_fix(LOGICAL,134)@10
    infPinfForSub_uid135_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(redist36_excI_siga_uid23_block_rsrvd_fix_q_3_q and redist29_excI_sigb_uid37_block_rsrvd_fix_q_3_q and invEffSub_uid134_block_rsrvd_fix_q);

    -- excRNaNS_uid136_block_rsrvd_fix(LOGICAL,135)@10
    excRNaNS_uid136_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(infPinfForSub_uid135_block_rsrvd_fix_q or redist15_oneIsNaN_uid128_block_rsrvd_fix_q_3_q);

    -- invExcRNaNS_uid174_block_rsrvd_fix(LOGICAL,173)@10
    invExcRNaNS_uid174_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (excRNaNS_uid136_block_rsrvd_fix_q));

    -- VCC(CONSTANT,1)
    VCC_q <= "1";

    -- signRPostExcSub_uid175_block_rsrvd_fix(LOGICAL,174)@10 + 1
    signRPostExcSub_uid175_block_rsrvd_fix_qi <= invExcRNaNS_uid174_block_rsrvd_fix_q and signASwap_uid173_block_rsrvd_fix_q and invAMASub_uid172_block_rsrvd_fix_q and invZMZSub_uid170_block_rsrvd_fix_q;
    signRPostExcSub_uid175_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => signRPostExcSub_uid175_block_rsrvd_fix_qi, xout => signRPostExcSub_uid175_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- cRBit_uid100_block_rsrvd_fix(CONSTANT,99)
    cRBit_uid100_block_rsrvd_fix_q <= "01000";

    -- leftShiftStage2Idx3Rng3_uid325_fracPostNormAdd_uid82_block_rsrvd_fix(BITSELECT,324)@8
    leftShiftStage2Idx3Rng3_uid325_fracPostNormAdd_uid82_block_rsrvd_fix_in <= leftShiftStage1_uid317_fracPostNormAdd_uid82_block_rsrvd_fix_q(53 downto 0);
    leftShiftStage2Idx3Rng3_uid325_fracPostNormAdd_uid82_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(leftShiftStage2Idx3Rng3_uid325_fracPostNormAdd_uid82_block_rsrvd_fix_in(53 downto 0));

    -- leftShiftStage2Idx3_uid326_fracPostNormAdd_uid82_block_rsrvd_fix(BITJOIN,325)@8
    leftShiftStage2Idx3_uid326_fracPostNormAdd_uid82_block_rsrvd_fix_q <= leftShiftStage2Idx3Rng3_uid325_fracPostNormAdd_uid82_block_rsrvd_fix_b & zv_uid74_block_rsrvd_fix_q;

    -- leftShiftStage2Idx2Rng2_uid322_fracPostNormAdd_uid82_block_rsrvd_fix(BITSELECT,321)@8
    leftShiftStage2Idx2Rng2_uid322_fracPostNormAdd_uid82_block_rsrvd_fix_in <= leftShiftStage1_uid317_fracPostNormAdd_uid82_block_rsrvd_fix_q(54 downto 0);
    leftShiftStage2Idx2Rng2_uid322_fracPostNormAdd_uid82_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(leftShiftStage2Idx2Rng2_uid322_fracPostNormAdd_uid82_block_rsrvd_fix_in(54 downto 0));

    -- leftShiftStage2Idx2_uid323_fracPostNormAdd_uid82_block_rsrvd_fix(BITJOIN,322)@8
    leftShiftStage2Idx2_uid323_fracPostNormAdd_uid82_block_rsrvd_fix_q <= leftShiftStage2Idx2Rng2_uid322_fracPostNormAdd_uid82_block_rsrvd_fix_b & zs_uid205_lzCountValAdd_uid81_block_rsrvd_fix_q;

    -- leftShiftStage2Idx1Rng1_uid319_fracPostNormAdd_uid82_block_rsrvd_fix(BITSELECT,318)@8
    leftShiftStage2Idx1Rng1_uid319_fracPostNormAdd_uid82_block_rsrvd_fix_in <= leftShiftStage1_uid317_fracPostNormAdd_uid82_block_rsrvd_fix_q(55 downto 0);
    leftShiftStage2Idx1Rng1_uid319_fracPostNormAdd_uid82_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(leftShiftStage2Idx1Rng1_uid319_fracPostNormAdd_uid82_block_rsrvd_fix_in(55 downto 0));

    -- leftShiftStage2Idx1_uid320_fracPostNormAdd_uid82_block_rsrvd_fix(BITJOIN,319)@8
    leftShiftStage2Idx1_uid320_fracPostNormAdd_uid82_block_rsrvd_fix_q <= leftShiftStage2Idx1Rng1_uid319_fracPostNormAdd_uid82_block_rsrvd_fix_b & GND_q;

    -- leftShiftStage1Idx3Rng12_uid314_fracPostNormAdd_uid82_block_rsrvd_fix(BITSELECT,313)@8
    leftShiftStage1Idx3Rng12_uid314_fracPostNormAdd_uid82_block_rsrvd_fix_in <= leftShiftStage0_uid306_fracPostNormAdd_uid82_block_rsrvd_fix_q(44 downto 0);
    leftShiftStage1Idx3Rng12_uid314_fracPostNormAdd_uid82_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(leftShiftStage1Idx3Rng12_uid314_fracPostNormAdd_uid82_block_rsrvd_fix_in(44 downto 0));

    -- leftShiftStage1Idx3_uid315_fracPostNormAdd_uid82_block_rsrvd_fix(BITJOIN,314)@8
    leftShiftStage1Idx3_uid315_fracPostNormAdd_uid82_block_rsrvd_fix_q <= leftShiftStage1Idx3Rng12_uid314_fracPostNormAdd_uid82_block_rsrvd_fix_b & rightShiftStage1Idx3Pad12_uid273_alignmentShifter_uid65_block_rsrvd_fix_q;

    -- leftShiftStage1Idx2Rng8_uid311_fracPostNormAdd_uid82_block_rsrvd_fix(BITSELECT,310)@8
    leftShiftStage1Idx2Rng8_uid311_fracPostNormAdd_uid82_block_rsrvd_fix_in <= leftShiftStage0_uid306_fracPostNormAdd_uid82_block_rsrvd_fix_q(48 downto 0);
    leftShiftStage1Idx2Rng8_uid311_fracPostNormAdd_uid82_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(leftShiftStage1Idx2Rng8_uid311_fracPostNormAdd_uid82_block_rsrvd_fix_in(48 downto 0));

    -- leftShiftStage1Idx2_uid312_fracPostNormAdd_uid82_block_rsrvd_fix(BITJOIN,311)@8
    leftShiftStage1Idx2_uid312_fracPostNormAdd_uid82_block_rsrvd_fix_q <= leftShiftStage1Idx2Rng8_uid311_fracPostNormAdd_uid82_block_rsrvd_fix_b & zs_uid193_lzCountValAdd_uid81_block_rsrvd_fix_q;

    -- leftShiftStage1Idx1Rng4_uid308_fracPostNormAdd_uid82_block_rsrvd_fix(BITSELECT,307)@8
    leftShiftStage1Idx1Rng4_uid308_fracPostNormAdd_uid82_block_rsrvd_fix_in <= leftShiftStage0_uid306_fracPostNormAdd_uid82_block_rsrvd_fix_q(52 downto 0);
    leftShiftStage1Idx1Rng4_uid308_fracPostNormAdd_uid82_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(leftShiftStage1Idx1Rng4_uid308_fracPostNormAdd_uid82_block_rsrvd_fix_in(52 downto 0));

    -- leftShiftStage1Idx1_uid309_fracPostNormAdd_uid82_block_rsrvd_fix(BITJOIN,308)@8
    leftShiftStage1Idx1_uid309_fracPostNormAdd_uid82_block_rsrvd_fix_q <= leftShiftStage1Idx1Rng4_uid308_fracPostNormAdd_uid82_block_rsrvd_fix_b & zs_uid199_lzCountValAdd_uid81_block_rsrvd_fix_q;

    -- leftShiftStage0Idx3Rng48_uid303_fracPostNormAdd_uid82_block_rsrvd_fix(BITSELECT,302)@8
    leftShiftStage0Idx3Rng48_uid303_fracPostNormAdd_uid82_block_rsrvd_fix_in <= redist25_fracResAddNoSignExt_uid79_block_rsrvd_fix_b_3_q(8 downto 0);
    leftShiftStage0Idx3Rng48_uid303_fracPostNormAdd_uid82_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(leftShiftStage0Idx3Rng48_uid303_fracPostNormAdd_uid82_block_rsrvd_fix_in(8 downto 0));

    -- leftShiftStage0Idx3_uid304_fracPostNormAdd_uid82_block_rsrvd_fix(BITJOIN,303)@8
    leftShiftStage0Idx3_uid304_fracPostNormAdd_uid82_block_rsrvd_fix_q <= leftShiftStage0Idx3Rng48_uid303_fracPostNormAdd_uid82_block_rsrvd_fix_b & rightShiftStage2Idx3Pad48_uid284_alignmentShifter_uid65_block_rsrvd_fix_q;

    -- fracResAdd_uid77_block_rsrvd_fix_lhsMSBs_select(BITSELECT,290)@5
    fracResAdd_uid77_block_rsrvd_fix_lhsMSBs_select_b <= STD_LOGIC_VECTOR(fracAOp_uid75_block_rsrvd_fix_q(55 downto 3));

    -- fracResAdd_uid77_block_rsrvd_fix_MSBs_sums(ADD,291)@5
    fracResAdd_uid77_block_rsrvd_fix_MSBs_sums_a <= STD_LOGIC_VECTOR("000" & fracResAdd_uid77_block_rsrvd_fix_lhsMSBs_select_b);
    fracResAdd_uid77_block_rsrvd_fix_MSBs_sums_b <= STD_LOGIC_VECTOR("0" & fracResAdd_uid77_block_rsrvd_fix_rhsMSBs_select_bit_select_merged_b);
    fracResAdd_uid77_block_rsrvd_fix_MSBs_sums_o <= STD_LOGIC_VECTOR(UNSIGNED(fracResAdd_uid77_block_rsrvd_fix_MSBs_sums_a) + UNSIGNED(fracResAdd_uid77_block_rsrvd_fix_MSBs_sums_b));
    fracResAdd_uid77_block_rsrvd_fix_MSBs_sums_q <= STD_LOGIC_VECTOR(fracResAdd_uid77_block_rsrvd_fix_MSBs_sums_o(55 downto 0));

    -- fracResAdd_uid77_block_rsrvd_fix_rhsMSBs_select_bit_select_merged(BITSELECT,366)@5
    fracResAdd_uid77_block_rsrvd_fix_rhsMSBs_select_bit_select_merged_b <= STD_LOGIC_VECTOR(fracBOp_uid76_block_rsrvd_fix_q(57 downto 3));
    fracResAdd_uid77_block_rsrvd_fix_rhsMSBs_select_bit_select_merged_c <= STD_LOGIC_VECTOR(fracBOp_uid76_block_rsrvd_fix_q(2 downto 0));

    -- fracResAdd_uid77_block_rsrvd_fix_split_join(BITJOIN,292)@5
    fracResAdd_uid77_block_rsrvd_fix_split_join_q <= fracResAdd_uid77_block_rsrvd_fix_MSBs_sums_q & fracResAdd_uid77_block_rsrvd_fix_rhsMSBs_select_bit_select_merged_c;

    -- fracResAddNoSignExt_uid79_block_rsrvd_fix(BITSELECT,78)@5
    fracResAddNoSignExt_uid79_block_rsrvd_fix_in <= fracResAdd_uid77_block_rsrvd_fix_split_join_q(56 downto 0);
    fracResAddNoSignExt_uid79_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(fracResAddNoSignExt_uid79_block_rsrvd_fix_in(56 downto 0));

    -- redist24_fracResAddNoSignExt_uid79_block_rsrvd_fix_b_1(DELAY,402)
    redist24_fracResAddNoSignExt_uid79_block_rsrvd_fix_b_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist24_fracResAddNoSignExt_uid79_block_rsrvd_fix_b_1_q <= fracResAddNoSignExt_uid79_block_rsrvd_fix_b;
            END IF;
        END IF;
    END PROCESS;

    -- vStage_uid183_lzCountValAdd_uid81_block_rsrvd_fix(BITSELECT,182)@6
    vStage_uid183_lzCountValAdd_uid81_block_rsrvd_fix_in <= redist24_fracResAddNoSignExt_uid79_block_rsrvd_fix_b_1_q(24 downto 0);
    vStage_uid183_lzCountValAdd_uid81_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(vStage_uid183_lzCountValAdd_uid81_block_rsrvd_fix_in(24 downto 0));

    -- redist13_vStage_uid183_lzCountValAdd_uid81_block_rsrvd_fix_b_2(DELAY,391)
    redist13_vStage_uid183_lzCountValAdd_uid81_block_rsrvd_fix_b_2_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist13_vStage_uid183_lzCountValAdd_uid81_block_rsrvd_fix_b_2_delay_0 <= STD_LOGIC_VECTOR(vStage_uid183_lzCountValAdd_uid81_block_rsrvd_fix_b);
                redist13_vStage_uid183_lzCountValAdd_uid81_block_rsrvd_fix_b_2_q <= STD_LOGIC_VECTOR(redist13_vStage_uid183_lzCountValAdd_uid81_block_rsrvd_fix_b_2_delay_0);
            END IF;
        END IF;
    END PROCESS;

    -- leftShiftStage0Idx2_uid301_fracPostNormAdd_uid82_block_rsrvd_fix(BITJOIN,300)@8
    leftShiftStage0Idx2_uid301_fracPostNormAdd_uid82_block_rsrvd_fix_q <= redist13_vStage_uid183_lzCountValAdd_uid81_block_rsrvd_fix_b_2_q & zs_uid179_lzCountValAdd_uid81_block_rsrvd_fix_q;

    -- leftShiftStage0Idx1Rng16_uid297_fracPostNormAdd_uid82_block_rsrvd_fix(BITSELECT,296)@8
    leftShiftStage0Idx1Rng16_uid297_fracPostNormAdd_uid82_block_rsrvd_fix_in <= redist25_fracResAddNoSignExt_uid79_block_rsrvd_fix_b_3_q(40 downto 0);
    leftShiftStage0Idx1Rng16_uid297_fracPostNormAdd_uid82_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(leftShiftStage0Idx1Rng16_uid297_fracPostNormAdd_uid82_block_rsrvd_fix_in(40 downto 0));

    -- leftShiftStage0Idx1_uid298_fracPostNormAdd_uid82_block_rsrvd_fix(BITJOIN,297)@8
    leftShiftStage0Idx1_uid298_fracPostNormAdd_uid82_block_rsrvd_fix_q <= leftShiftStage0Idx1Rng16_uid297_fracPostNormAdd_uid82_block_rsrvd_fix_b & zs_uid187_lzCountValAdd_uid81_block_rsrvd_fix_q;

    -- redist25_fracResAddNoSignExt_uid79_block_rsrvd_fix_b_3(DELAY,403)
    redist25_fracResAddNoSignExt_uid79_block_rsrvd_fix_b_3_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist25_fracResAddNoSignExt_uid79_block_rsrvd_fix_b_3_delay_0 <= STD_LOGIC_VECTOR(redist24_fracResAddNoSignExt_uid79_block_rsrvd_fix_b_1_q);
                redist25_fracResAddNoSignExt_uid79_block_rsrvd_fix_b_3_q <= STD_LOGIC_VECTOR(redist25_fracResAddNoSignExt_uid79_block_rsrvd_fix_b_3_delay_0);
            END IF;
        END IF;
    END PROCESS;

    -- leftShiftStage0_uid306_fracPostNormAdd_uid82_block_rsrvd_fix(MUX,305)@8
    leftShiftStage0_uid306_fracPostNormAdd_uid82_block_rsrvd_fix_s <= leftShiftStageSel0Dto4_uid305_fracPostNormAdd_uid82_block_rsrvd_fix_bit_select_merged_b;
    leftShiftStage0_uid306_fracPostNormAdd_uid82_block_rsrvd_fix_combproc: PROCESS (leftShiftStage0_uid306_fracPostNormAdd_uid82_block_rsrvd_fix_s, redist25_fracResAddNoSignExt_uid79_block_rsrvd_fix_b_3_q, leftShiftStage0Idx1_uid298_fracPostNormAdd_uid82_block_rsrvd_fix_q, leftShiftStage0Idx2_uid301_fracPostNormAdd_uid82_block_rsrvd_fix_q, leftShiftStage0Idx3_uid304_fracPostNormAdd_uid82_block_rsrvd_fix_q)
    BEGIN
        CASE (leftShiftStage0_uid306_fracPostNormAdd_uid82_block_rsrvd_fix_s) IS
            WHEN "00" => leftShiftStage0_uid306_fracPostNormAdd_uid82_block_rsrvd_fix_q <= redist25_fracResAddNoSignExt_uid79_block_rsrvd_fix_b_3_q;
            WHEN "01" => leftShiftStage0_uid306_fracPostNormAdd_uid82_block_rsrvd_fix_q <= leftShiftStage0Idx1_uid298_fracPostNormAdd_uid82_block_rsrvd_fix_q;
            WHEN "10" => leftShiftStage0_uid306_fracPostNormAdd_uid82_block_rsrvd_fix_q <= leftShiftStage0Idx2_uid301_fracPostNormAdd_uid82_block_rsrvd_fix_q;
            WHEN "11" => leftShiftStage0_uid306_fracPostNormAdd_uid82_block_rsrvd_fix_q <= leftShiftStage0Idx3_uid304_fracPostNormAdd_uid82_block_rsrvd_fix_q;
            WHEN OTHERS => leftShiftStage0_uid306_fracPostNormAdd_uid82_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- leftShiftStage1_uid317_fracPostNormAdd_uid82_block_rsrvd_fix(MUX,316)@8
    leftShiftStage1_uid317_fracPostNormAdd_uid82_block_rsrvd_fix_s <= leftShiftStageSel0Dto4_uid305_fracPostNormAdd_uid82_block_rsrvd_fix_bit_select_merged_c;
    leftShiftStage1_uid317_fracPostNormAdd_uid82_block_rsrvd_fix_combproc: PROCESS (leftShiftStage1_uid317_fracPostNormAdd_uid82_block_rsrvd_fix_s, leftShiftStage0_uid306_fracPostNormAdd_uid82_block_rsrvd_fix_q, leftShiftStage1Idx1_uid309_fracPostNormAdd_uid82_block_rsrvd_fix_q, leftShiftStage1Idx2_uid312_fracPostNormAdd_uid82_block_rsrvd_fix_q, leftShiftStage1Idx3_uid315_fracPostNormAdd_uid82_block_rsrvd_fix_q)
    BEGIN
        CASE (leftShiftStage1_uid317_fracPostNormAdd_uid82_block_rsrvd_fix_s) IS
            WHEN "00" => leftShiftStage1_uid317_fracPostNormAdd_uid82_block_rsrvd_fix_q <= leftShiftStage0_uid306_fracPostNormAdd_uid82_block_rsrvd_fix_q;
            WHEN "01" => leftShiftStage1_uid317_fracPostNormAdd_uid82_block_rsrvd_fix_q <= leftShiftStage1Idx1_uid309_fracPostNormAdd_uid82_block_rsrvd_fix_q;
            WHEN "10" => leftShiftStage1_uid317_fracPostNormAdd_uid82_block_rsrvd_fix_q <= leftShiftStage1Idx2_uid312_fracPostNormAdd_uid82_block_rsrvd_fix_q;
            WHEN "11" => leftShiftStage1_uid317_fracPostNormAdd_uid82_block_rsrvd_fix_q <= leftShiftStage1Idx3_uid315_fracPostNormAdd_uid82_block_rsrvd_fix_q;
            WHEN OTHERS => leftShiftStage1_uid317_fracPostNormAdd_uid82_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- rVStage_uid180_lzCountValAdd_uid81_block_rsrvd_fix(BITSELECT,179)@6
    rVStage_uid180_lzCountValAdd_uid81_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(redist24_fracResAddNoSignExt_uid79_block_rsrvd_fix_b_1_q(56 downto 25));

    -- vCount_uid181_lzCountValAdd_uid81_block_rsrvd_fix(LOGICAL,180)@6
    vCount_uid181_lzCountValAdd_uid81_block_rsrvd_fix_q <= "1" WHEN rVStage_uid180_lzCountValAdd_uid81_block_rsrvd_fix_b = zs_uid179_lzCountValAdd_uid81_block_rsrvd_fix_q ELSE "0";

    -- redist14_vCount_uid181_lzCountValAdd_uid81_block_rsrvd_fix_q_2(DELAY,392)
    redist14_vCount_uid181_lzCountValAdd_uid81_block_rsrvd_fix_q_2_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist14_vCount_uid181_lzCountValAdd_uid81_block_rsrvd_fix_q_2_delay_0 <= STD_LOGIC_VECTOR(vCount_uid181_lzCountValAdd_uid81_block_rsrvd_fix_q);
                redist14_vCount_uid181_lzCountValAdd_uid81_block_rsrvd_fix_q_2_q <= STD_LOGIC_VECTOR(redist14_vCount_uid181_lzCountValAdd_uid81_block_rsrvd_fix_q_2_delay_0);
            END IF;
        END IF;
    END PROCESS;

    -- cStage_uid184_lzCountValAdd_uid81_block_rsrvd_fix(BITJOIN,183)@6
    cStage_uid184_lzCountValAdd_uid81_block_rsrvd_fix_q <= vStage_uid183_lzCountValAdd_uid81_block_rsrvd_fix_b & mO_uid182_lzCountValAdd_uid81_block_rsrvd_fix_q;

    -- vStagei_uid186_lzCountValAdd_uid81_block_rsrvd_fix(MUX,185)@6
    vStagei_uid186_lzCountValAdd_uid81_block_rsrvd_fix_s <= vCount_uid181_lzCountValAdd_uid81_block_rsrvd_fix_q;
    vStagei_uid186_lzCountValAdd_uid81_block_rsrvd_fix_combproc: PROCESS (vStagei_uid186_lzCountValAdd_uid81_block_rsrvd_fix_s, rVStage_uid180_lzCountValAdd_uid81_block_rsrvd_fix_b, cStage_uid184_lzCountValAdd_uid81_block_rsrvd_fix_q)
    BEGIN
        CASE (vStagei_uid186_lzCountValAdd_uid81_block_rsrvd_fix_s) IS
            WHEN "0" => vStagei_uid186_lzCountValAdd_uid81_block_rsrvd_fix_q <= rVStage_uid180_lzCountValAdd_uid81_block_rsrvd_fix_b;
            WHEN "1" => vStagei_uid186_lzCountValAdd_uid81_block_rsrvd_fix_q <= cStage_uid184_lzCountValAdd_uid81_block_rsrvd_fix_q;
            WHEN OTHERS => vStagei_uid186_lzCountValAdd_uid81_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- rVStage_uid188_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged(BITSELECT,367)@6
    rVStage_uid188_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b <= STD_LOGIC_VECTOR(vStagei_uid186_lzCountValAdd_uid81_block_rsrvd_fix_q(31 downto 16));
    rVStage_uid188_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_c <= STD_LOGIC_VECTOR(vStagei_uid186_lzCountValAdd_uid81_block_rsrvd_fix_q(15 downto 0));

    -- vCount_uid189_lzCountValAdd_uid81_block_rsrvd_fix(LOGICAL,188)@6
    vCount_uid189_lzCountValAdd_uid81_block_rsrvd_fix_q <= "1" WHEN rVStage_uid188_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b = zs_uid187_lzCountValAdd_uid81_block_rsrvd_fix_q ELSE "0";

    -- redist12_vCount_uid189_lzCountValAdd_uid81_block_rsrvd_fix_q_2(DELAY,390)
    redist12_vCount_uid189_lzCountValAdd_uid81_block_rsrvd_fix_q_2_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist12_vCount_uid189_lzCountValAdd_uid81_block_rsrvd_fix_q_2_delay_0 <= STD_LOGIC_VECTOR(vCount_uid189_lzCountValAdd_uid81_block_rsrvd_fix_q);
                redist12_vCount_uid189_lzCountValAdd_uid81_block_rsrvd_fix_q_2_q <= STD_LOGIC_VECTOR(redist12_vCount_uid189_lzCountValAdd_uid81_block_rsrvd_fix_q_2_delay_0);
            END IF;
        END IF;
    END PROCESS;

    -- vStagei_uid192_lzCountValAdd_uid81_block_rsrvd_fix(MUX,191)@6 + 1
    vStagei_uid192_lzCountValAdd_uid81_block_rsrvd_fix_s <= vCount_uid189_lzCountValAdd_uid81_block_rsrvd_fix_q;
    vStagei_uid192_lzCountValAdd_uid81_block_rsrvd_fix_clkproc: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                CASE (vStagei_uid192_lzCountValAdd_uid81_block_rsrvd_fix_s) IS
                    WHEN "0" => vStagei_uid192_lzCountValAdd_uid81_block_rsrvd_fix_q <= rVStage_uid188_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b;
                    WHEN "1" => vStagei_uid192_lzCountValAdd_uid81_block_rsrvd_fix_q <= rVStage_uid188_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_c;
                    WHEN OTHERS => vStagei_uid192_lzCountValAdd_uid81_block_rsrvd_fix_q <= (others => '0');
                END CASE;
            END IF;
        END IF;
    END PROCESS;

    -- rVStage_uid194_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged(BITSELECT,368)@7
    rVStage_uid194_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b <= STD_LOGIC_VECTOR(vStagei_uid192_lzCountValAdd_uid81_block_rsrvd_fix_q(15 downto 8));
    rVStage_uid194_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_c <= STD_LOGIC_VECTOR(vStagei_uid192_lzCountValAdd_uid81_block_rsrvd_fix_q(7 downto 0));

    -- vCount_uid195_lzCountValAdd_uid81_block_rsrvd_fix(LOGICAL,194)@7
    vCount_uid195_lzCountValAdd_uid81_block_rsrvd_fix_q <= "1" WHEN rVStage_uid194_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b = zs_uid193_lzCountValAdd_uid81_block_rsrvd_fix_q ELSE "0";

    -- redist11_vCount_uid195_lzCountValAdd_uid81_block_rsrvd_fix_q_1(DELAY,389)
    redist11_vCount_uid195_lzCountValAdd_uid81_block_rsrvd_fix_q_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist11_vCount_uid195_lzCountValAdd_uid81_block_rsrvd_fix_q_1_q <= vCount_uid195_lzCountValAdd_uid81_block_rsrvd_fix_q;
            END IF;
        END IF;
    END PROCESS;

    -- vStagei_uid198_lzCountValAdd_uid81_block_rsrvd_fix(MUX,197)@7
    vStagei_uid198_lzCountValAdd_uid81_block_rsrvd_fix_s <= vCount_uid195_lzCountValAdd_uid81_block_rsrvd_fix_q;
    vStagei_uid198_lzCountValAdd_uid81_block_rsrvd_fix_combproc: PROCESS (vStagei_uid198_lzCountValAdd_uid81_block_rsrvd_fix_s, rVStage_uid194_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b, rVStage_uid194_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_c)
    BEGIN
        CASE (vStagei_uid198_lzCountValAdd_uid81_block_rsrvd_fix_s) IS
            WHEN "0" => vStagei_uid198_lzCountValAdd_uid81_block_rsrvd_fix_q <= rVStage_uid194_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b;
            WHEN "1" => vStagei_uid198_lzCountValAdd_uid81_block_rsrvd_fix_q <= rVStage_uid194_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_c;
            WHEN OTHERS => vStagei_uid198_lzCountValAdd_uid81_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- rVStage_uid200_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged(BITSELECT,369)@7
    rVStage_uid200_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b <= STD_LOGIC_VECTOR(vStagei_uid198_lzCountValAdd_uid81_block_rsrvd_fix_q(7 downto 4));
    rVStage_uid200_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_c <= STD_LOGIC_VECTOR(vStagei_uid198_lzCountValAdd_uid81_block_rsrvd_fix_q(3 downto 0));

    -- vCount_uid201_lzCountValAdd_uid81_block_rsrvd_fix(LOGICAL,200)@7
    vCount_uid201_lzCountValAdd_uid81_block_rsrvd_fix_q <= "1" WHEN rVStage_uid200_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b = zs_uid199_lzCountValAdd_uid81_block_rsrvd_fix_q ELSE "0";

    -- redist10_vCount_uid201_lzCountValAdd_uid81_block_rsrvd_fix_q_1(DELAY,388)
    redist10_vCount_uid201_lzCountValAdd_uid81_block_rsrvd_fix_q_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist10_vCount_uid201_lzCountValAdd_uid81_block_rsrvd_fix_q_1_q <= vCount_uid201_lzCountValAdd_uid81_block_rsrvd_fix_q;
            END IF;
        END IF;
    END PROCESS;

    -- vStagei_uid204_lzCountValAdd_uid81_block_rsrvd_fix(MUX,203)@7
    vStagei_uid204_lzCountValAdd_uid81_block_rsrvd_fix_s <= vCount_uid201_lzCountValAdd_uid81_block_rsrvd_fix_q;
    vStagei_uid204_lzCountValAdd_uid81_block_rsrvd_fix_combproc: PROCESS (vStagei_uid204_lzCountValAdd_uid81_block_rsrvd_fix_s, rVStage_uid200_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b, rVStage_uid200_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_c)
    BEGIN
        CASE (vStagei_uid204_lzCountValAdd_uid81_block_rsrvd_fix_s) IS
            WHEN "0" => vStagei_uid204_lzCountValAdd_uid81_block_rsrvd_fix_q <= rVStage_uid200_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b;
            WHEN "1" => vStagei_uid204_lzCountValAdd_uid81_block_rsrvd_fix_q <= rVStage_uid200_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_c;
            WHEN OTHERS => vStagei_uid204_lzCountValAdd_uid81_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- rVStage_uid206_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged(BITSELECT,370)@7
    rVStage_uid206_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b <= STD_LOGIC_VECTOR(vStagei_uid204_lzCountValAdd_uid81_block_rsrvd_fix_q(3 downto 2));
    rVStage_uid206_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_c <= STD_LOGIC_VECTOR(vStagei_uid204_lzCountValAdd_uid81_block_rsrvd_fix_q(1 downto 0));

    -- vCount_uid207_lzCountValAdd_uid81_block_rsrvd_fix(LOGICAL,206)@7 + 1
    vCount_uid207_lzCountValAdd_uid81_block_rsrvd_fix_qi <= "1" WHEN rVStage_uid206_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b = zs_uid205_lzCountValAdd_uid81_block_rsrvd_fix_q ELSE "0";
    vCount_uid207_lzCountValAdd_uid81_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => vCount_uid207_lzCountValAdd_uid81_block_rsrvd_fix_qi, xout => vCount_uid207_lzCountValAdd_uid81_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- redist4_rVStage_uid206_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_c_1(DELAY,382)
    redist4_rVStage_uid206_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_c_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist4_rVStage_uid206_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_c_1_q <= rVStage_uid206_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_c;
            END IF;
        END IF;
    END PROCESS;

    -- redist3_rVStage_uid206_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b_1(DELAY,381)
    redist3_rVStage_uid206_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist3_rVStage_uid206_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b_1_q <= rVStage_uid206_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b;
            END IF;
        END IF;
    END PROCESS;

    -- vStagei_uid210_lzCountValAdd_uid81_block_rsrvd_fix(MUX,209)@8
    vStagei_uid210_lzCountValAdd_uid81_block_rsrvd_fix_s <= vCount_uid207_lzCountValAdd_uid81_block_rsrvd_fix_q;
    vStagei_uid210_lzCountValAdd_uid81_block_rsrvd_fix_combproc: PROCESS (vStagei_uid210_lzCountValAdd_uid81_block_rsrvd_fix_s, redist3_rVStage_uid206_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b_1_q, redist4_rVStage_uid206_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_c_1_q)
    BEGIN
        CASE (vStagei_uid210_lzCountValAdd_uid81_block_rsrvd_fix_s) IS
            WHEN "0" => vStagei_uid210_lzCountValAdd_uid81_block_rsrvd_fix_q <= redist3_rVStage_uid206_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_b_1_q;
            WHEN "1" => vStagei_uid210_lzCountValAdd_uid81_block_rsrvd_fix_q <= redist4_rVStage_uid206_lzCountValAdd_uid81_block_rsrvd_fix_bit_select_merged_c_1_q;
            WHEN OTHERS => vStagei_uid210_lzCountValAdd_uid81_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- rVStage_uid212_lzCountValAdd_uid81_block_rsrvd_fix(BITSELECT,211)@8
    rVStage_uid212_lzCountValAdd_uid81_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(vStagei_uid210_lzCountValAdd_uid81_block_rsrvd_fix_q(1 downto 1));

    -- vCount_uid213_lzCountValAdd_uid81_block_rsrvd_fix(LOGICAL,212)@8
    vCount_uid213_lzCountValAdd_uid81_block_rsrvd_fix_q <= "1" WHEN rVStage_uid212_lzCountValAdd_uid81_block_rsrvd_fix_b = GND_q ELSE "0";

    -- r_uid214_lzCountValAdd_uid81_block_rsrvd_fix(BITJOIN,213)@8
    r_uid214_lzCountValAdd_uid81_block_rsrvd_fix_q <= redist14_vCount_uid181_lzCountValAdd_uid81_block_rsrvd_fix_q_2_q & redist12_vCount_uid189_lzCountValAdd_uid81_block_rsrvd_fix_q_2_q & redist11_vCount_uid195_lzCountValAdd_uid81_block_rsrvd_fix_q_1_q & redist10_vCount_uid201_lzCountValAdd_uid81_block_rsrvd_fix_q_1_q & vCount_uid207_lzCountValAdd_uid81_block_rsrvd_fix_q & vCount_uid213_lzCountValAdd_uid81_block_rsrvd_fix_q;

    -- leftShiftStageSel0Dto4_uid305_fracPostNormAdd_uid82_block_rsrvd_fix_bit_select_merged(BITSELECT,371)@8
    leftShiftStageSel0Dto4_uid305_fracPostNormAdd_uid82_block_rsrvd_fix_bit_select_merged_b <= STD_LOGIC_VECTOR(r_uid214_lzCountValAdd_uid81_block_rsrvd_fix_q(5 downto 4));
    leftShiftStageSel0Dto4_uid305_fracPostNormAdd_uid82_block_rsrvd_fix_bit_select_merged_c <= STD_LOGIC_VECTOR(r_uid214_lzCountValAdd_uid81_block_rsrvd_fix_q(3 downto 2));
    leftShiftStageSel0Dto4_uid305_fracPostNormAdd_uid82_block_rsrvd_fix_bit_select_merged_d <= STD_LOGIC_VECTOR(r_uid214_lzCountValAdd_uid81_block_rsrvd_fix_q(1 downto 0));

    -- leftShiftStage2_uid328_fracPostNormAdd_uid82_block_rsrvd_fix(MUX,327)@8
    leftShiftStage2_uid328_fracPostNormAdd_uid82_block_rsrvd_fix_s <= leftShiftStageSel0Dto4_uid305_fracPostNormAdd_uid82_block_rsrvd_fix_bit_select_merged_d;
    leftShiftStage2_uid328_fracPostNormAdd_uid82_block_rsrvd_fix_combproc: PROCESS (leftShiftStage2_uid328_fracPostNormAdd_uid82_block_rsrvd_fix_s, leftShiftStage1_uid317_fracPostNormAdd_uid82_block_rsrvd_fix_q, leftShiftStage2Idx1_uid320_fracPostNormAdd_uid82_block_rsrvd_fix_q, leftShiftStage2Idx2_uid323_fracPostNormAdd_uid82_block_rsrvd_fix_q, leftShiftStage2Idx3_uid326_fracPostNormAdd_uid82_block_rsrvd_fix_q)
    BEGIN
        CASE (leftShiftStage2_uid328_fracPostNormAdd_uid82_block_rsrvd_fix_s) IS
            WHEN "00" => leftShiftStage2_uid328_fracPostNormAdd_uid82_block_rsrvd_fix_q <= leftShiftStage1_uid317_fracPostNormAdd_uid82_block_rsrvd_fix_q;
            WHEN "01" => leftShiftStage2_uid328_fracPostNormAdd_uid82_block_rsrvd_fix_q <= leftShiftStage2Idx1_uid320_fracPostNormAdd_uid82_block_rsrvd_fix_q;
            WHEN "10" => leftShiftStage2_uid328_fracPostNormAdd_uid82_block_rsrvd_fix_q <= leftShiftStage2Idx2_uid323_fracPostNormAdd_uid82_block_rsrvd_fix_q;
            WHEN "11" => leftShiftStage2_uid328_fracPostNormAdd_uid82_block_rsrvd_fix_q <= leftShiftStage2Idx3_uid326_fracPostNormAdd_uid82_block_rsrvd_fix_q;
            WHEN OTHERS => leftShiftStage2_uid328_fracPostNormAdd_uid82_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- LSB_add_uid98_block_rsrvd_fix(BITSELECT,97)@8
    LSB_add_uid98_block_rsrvd_fix_in <= STD_LOGIC_VECTOR(leftShiftStage2_uid328_fracPostNormAdd_uid82_block_rsrvd_fix_q(4 downto 0));
    LSB_add_uid98_block_rsrvd_fix_b <= LSB_add_uid98_block_rsrvd_fix_in(4 downto 4);

    -- Guard_add_uid97_block_rsrvd_fix(BITSELECT,96)@8
    Guard_add_uid97_block_rsrvd_fix_in <= STD_LOGIC_VECTOR(leftShiftStage2_uid328_fracPostNormAdd_uid82_block_rsrvd_fix_q(3 downto 0));
    Guard_add_uid97_block_rsrvd_fix_b <= Guard_add_uid97_block_rsrvd_fix_in(3 downto 3);

    -- Round_add_uid96_block_rsrvd_fix(BITSELECT,95)@8
    Round_add_uid96_block_rsrvd_fix_in <= STD_LOGIC_VECTOR(leftShiftStage2_uid328_fracPostNormAdd_uid82_block_rsrvd_fix_q(2 downto 0));
    Round_add_uid96_block_rsrvd_fix_b <= Round_add_uid96_block_rsrvd_fix_in(2 downto 2);

    -- sticky1_add_uid95_block_rsrvd_fix(BITSELECT,94)@8
    sticky1_add_uid95_block_rsrvd_fix_in <= STD_LOGIC_VECTOR(leftShiftStage2_uid328_fracPostNormAdd_uid82_block_rsrvd_fix_q(1 downto 0));
    sticky1_add_uid95_block_rsrvd_fix_b <= sticky1_add_uid95_block_rsrvd_fix_in(1 downto 1);

    -- sticky0_add_uid94_block_rsrvd_fix(BITSELECT,93)@8
    sticky0_add_uid94_block_rsrvd_fix_in <= STD_LOGIC_VECTOR(leftShiftStage2_uid328_fracPostNormAdd_uid82_block_rsrvd_fix_q(0 downto 0));
    sticky0_add_uid94_block_rsrvd_fix_b <= sticky0_add_uid94_block_rsrvd_fix_in(0 downto 0);

    -- rndBitCond_add_uid99_block_rsrvd_fix(BITJOIN,98)@8
    rndBitCond_add_uid99_block_rsrvd_fix_q <= LSB_add_uid98_block_rsrvd_fix_b & Guard_add_uid97_block_rsrvd_fix_b & Round_add_uid96_block_rsrvd_fix_b & sticky1_add_uid95_block_rsrvd_fix_b & sticky0_add_uid94_block_rsrvd_fix_b;

    -- rBi_add_uid101_block_rsrvd_fix(LOGICAL,100)@8 + 1
    rBi_add_uid101_block_rsrvd_fix_qi <= "1" WHEN rndBitCond_add_uid99_block_rsrvd_fix_q = cRBit_uid100_block_rsrvd_fix_q ELSE "0";
    rBi_add_uid101_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => rBi_add_uid101_block_rsrvd_fix_qi, xout => rBi_add_uid101_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- roundBit_add_uid102_block_rsrvd_fix(LOGICAL,101)@9
    roundBit_add_uid102_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (rBi_add_uid101_block_rsrvd_fix_q));

    -- expInc_uid87_block_rsrvd_fix(ADD,86)@7 + 1
    expInc_uid87_block_rsrvd_fix_a <= STD_LOGIC_VECTOR("0" & redist41_exp_siga_uid17_block_rsrvd_fix_b_7_mem_q);
    expInc_uid87_block_rsrvd_fix_b <= STD_LOGIC_VECTOR("00000000000" & VCC_q);
    expInc_uid87_block_rsrvd_fix_clkproc: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                expInc_uid87_block_rsrvd_fix_o <= STD_LOGIC_VECTOR(UNSIGNED(expInc_uid87_block_rsrvd_fix_a) + UNSIGNED(expInc_uid87_block_rsrvd_fix_b));
            END IF;
        END IF;
    END PROCESS;
    expInc_uid87_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(expInc_uid87_block_rsrvd_fix_o(11 downto 0));

    -- expPostNormAdd_uid89_block_rsrvd_fix(SUB,88)@8 + 1
    expPostNormAdd_uid89_block_rsrvd_fix_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("0" & expInc_uid87_block_rsrvd_fix_q));
    expPostNormAdd_uid89_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("0000000" & r_uid214_lzCountValAdd_uid81_block_rsrvd_fix_q));
    expPostNormAdd_uid89_block_rsrvd_fix_clkproc: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                expPostNormAdd_uid89_block_rsrvd_fix_o <= STD_LOGIC_VECTOR(SIGNED(expPostNormAdd_uid89_block_rsrvd_fix_a) - SIGNED(expPostNormAdd_uid89_block_rsrvd_fix_b));
            END IF;
        END IF;
    END PROCESS;
    expPostNormAdd_uid89_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(expPostNormAdd_uid89_block_rsrvd_fix_o(12 downto 0));

    -- fracPostNormAddRndRange_uid92_block_rsrvd_fix(BITSELECT,91)@8
    fracPostNormAddRndRange_uid92_block_rsrvd_fix_in <= leftShiftStage2_uid328_fracPostNormAdd_uid82_block_rsrvd_fix_q(55 downto 0);
    fracPostNormAddRndRange_uid92_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(fracPostNormAddRndRange_uid92_block_rsrvd_fix_in(55 downto 3));

    -- redist19_fracPostNormAddRndRange_uid92_block_rsrvd_fix_b_1(DELAY,397)
    redist19_fracPostNormAddRndRange_uid92_block_rsrvd_fix_b_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist19_fracPostNormAddRndRange_uid92_block_rsrvd_fix_b_1_q <= fracPostNormAddRndRange_uid92_block_rsrvd_fix_b;
            END IF;
        END IF;
    END PROCESS;

    -- expFracRAdd_uid93_block_rsrvd_fix(BITJOIN,92)@9
    expFracRAdd_uid93_block_rsrvd_fix_q <= expPostNormAdd_uid89_block_rsrvd_fix_q & redist19_fracPostNormAddRndRange_uid92_block_rsrvd_fix_b_1_q;

    -- expFracRAddPostRound_uid103_block_rsrvd_fix(ADD,102)@9
    expFracRAddPostRound_uid103_block_rsrvd_fix_a <= STD_LOGIC_VECTOR("0" & expFracRAdd_uid93_block_rsrvd_fix_q);
    expFracRAddPostRound_uid103_block_rsrvd_fix_b <= STD_LOGIC_VECTOR("000000000000000000000000000000000000000000000000000000000000000000" & roundBit_add_uid102_block_rsrvd_fix_q);
    expFracRAddPostRound_uid103_block_rsrvd_fix_o <= STD_LOGIC_VECTOR(UNSIGNED(expFracRAddPostRound_uid103_block_rsrvd_fix_a) + UNSIGNED(expFracRAddPostRound_uid103_block_rsrvd_fix_b));
    expFracRAddPostRound_uid103_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(expFracRAddPostRound_uid103_block_rsrvd_fix_o(66 downto 0));

    -- expRPreExcAdd_uid119_block_rsrvd_fix(BITSELECT,118)@9
    expRPreExcAdd_uid119_block_rsrvd_fix_in <= expFracRAddPostRound_uid103_block_rsrvd_fix_q(63 downto 0);
    expRPreExcAdd_uid119_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(expRPreExcAdd_uid119_block_rsrvd_fix_in(63 downto 53));

    -- redist17_expRPreExcAdd_uid119_block_rsrvd_fix_b_1(DELAY,395)
    redist17_expRPreExcAdd_uid119_block_rsrvd_fix_b_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist17_expRPreExcAdd_uid119_block_rsrvd_fix_b_1_q <= expRPreExcAdd_uid119_block_rsrvd_fix_b;
            END IF;
        END IF;
    END PROCESS;

    -- leftShiftStage2Idx3Rng3_uid361_fracPostNormSub_uid84_block_rsrvd_fix(BITSELECT,360)@8
    leftShiftStage2Idx3Rng3_uid361_fracPostNormSub_uid84_block_rsrvd_fix_in <= leftShiftStage1_uid353_fracPostNormSub_uid84_block_rsrvd_fix_q(53 downto 0);
    leftShiftStage2Idx3Rng3_uid361_fracPostNormSub_uid84_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(leftShiftStage2Idx3Rng3_uid361_fracPostNormSub_uid84_block_rsrvd_fix_in(53 downto 0));

    -- leftShiftStage2Idx3_uid362_fracPostNormSub_uid84_block_rsrvd_fix(BITJOIN,361)@8
    leftShiftStage2Idx3_uid362_fracPostNormSub_uid84_block_rsrvd_fix_q <= leftShiftStage2Idx3Rng3_uid361_fracPostNormSub_uid84_block_rsrvd_fix_b & zv_uid74_block_rsrvd_fix_q;

    -- leftShiftStage2Idx2Rng2_uid358_fracPostNormSub_uid84_block_rsrvd_fix(BITSELECT,357)@8
    leftShiftStage2Idx2Rng2_uid358_fracPostNormSub_uid84_block_rsrvd_fix_in <= leftShiftStage1_uid353_fracPostNormSub_uid84_block_rsrvd_fix_q(54 downto 0);
    leftShiftStage2Idx2Rng2_uid358_fracPostNormSub_uid84_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(leftShiftStage2Idx2Rng2_uid358_fracPostNormSub_uid84_block_rsrvd_fix_in(54 downto 0));

    -- leftShiftStage2Idx2_uid359_fracPostNormSub_uid84_block_rsrvd_fix(BITJOIN,358)@8
    leftShiftStage2Idx2_uid359_fracPostNormSub_uid84_block_rsrvd_fix_q <= leftShiftStage2Idx2Rng2_uid358_fracPostNormSub_uid84_block_rsrvd_fix_b & zs_uid205_lzCountValAdd_uid81_block_rsrvd_fix_q;

    -- leftShiftStage2Idx1Rng1_uid355_fracPostNormSub_uid84_block_rsrvd_fix(BITSELECT,354)@8
    leftShiftStage2Idx1Rng1_uid355_fracPostNormSub_uid84_block_rsrvd_fix_in <= leftShiftStage1_uid353_fracPostNormSub_uid84_block_rsrvd_fix_q(55 downto 0);
    leftShiftStage2Idx1Rng1_uid355_fracPostNormSub_uid84_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(leftShiftStage2Idx1Rng1_uid355_fracPostNormSub_uid84_block_rsrvd_fix_in(55 downto 0));

    -- leftShiftStage2Idx1_uid356_fracPostNormSub_uid84_block_rsrvd_fix(BITJOIN,355)@8
    leftShiftStage2Idx1_uid356_fracPostNormSub_uid84_block_rsrvd_fix_q <= leftShiftStage2Idx1Rng1_uid355_fracPostNormSub_uid84_block_rsrvd_fix_b & GND_q;

    -- leftShiftStage1Idx3Rng12_uid350_fracPostNormSub_uid84_block_rsrvd_fix(BITSELECT,349)@8
    leftShiftStage1Idx3Rng12_uid350_fracPostNormSub_uid84_block_rsrvd_fix_in <= leftShiftStage0_uid342_fracPostNormSub_uid84_block_rsrvd_fix_q(44 downto 0);
    leftShiftStage1Idx3Rng12_uid350_fracPostNormSub_uid84_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(leftShiftStage1Idx3Rng12_uid350_fracPostNormSub_uid84_block_rsrvd_fix_in(44 downto 0));

    -- leftShiftStage1Idx3_uid351_fracPostNormSub_uid84_block_rsrvd_fix(BITJOIN,350)@8
    leftShiftStage1Idx3_uid351_fracPostNormSub_uid84_block_rsrvd_fix_q <= leftShiftStage1Idx3Rng12_uid350_fracPostNormSub_uid84_block_rsrvd_fix_b & rightShiftStage1Idx3Pad12_uid273_alignmentShifter_uid65_block_rsrvd_fix_q;

    -- leftShiftStage1Idx2Rng8_uid347_fracPostNormSub_uid84_block_rsrvd_fix(BITSELECT,346)@8
    leftShiftStage1Idx2Rng8_uid347_fracPostNormSub_uid84_block_rsrvd_fix_in <= leftShiftStage0_uid342_fracPostNormSub_uid84_block_rsrvd_fix_q(48 downto 0);
    leftShiftStage1Idx2Rng8_uid347_fracPostNormSub_uid84_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(leftShiftStage1Idx2Rng8_uid347_fracPostNormSub_uid84_block_rsrvd_fix_in(48 downto 0));

    -- leftShiftStage1Idx2_uid348_fracPostNormSub_uid84_block_rsrvd_fix(BITJOIN,347)@8
    leftShiftStage1Idx2_uid348_fracPostNormSub_uid84_block_rsrvd_fix_q <= leftShiftStage1Idx2Rng8_uid347_fracPostNormSub_uid84_block_rsrvd_fix_b & zs_uid193_lzCountValAdd_uid81_block_rsrvd_fix_q;

    -- leftShiftStage1Idx1Rng4_uid344_fracPostNormSub_uid84_block_rsrvd_fix(BITSELECT,343)@8
    leftShiftStage1Idx1Rng4_uid344_fracPostNormSub_uid84_block_rsrvd_fix_in <= leftShiftStage0_uid342_fracPostNormSub_uid84_block_rsrvd_fix_q(52 downto 0);
    leftShiftStage1Idx1Rng4_uid344_fracPostNormSub_uid84_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(leftShiftStage1Idx1Rng4_uid344_fracPostNormSub_uid84_block_rsrvd_fix_in(52 downto 0));

    -- leftShiftStage1Idx1_uid345_fracPostNormSub_uid84_block_rsrvd_fix(BITJOIN,344)@8
    leftShiftStage1Idx1_uid345_fracPostNormSub_uid84_block_rsrvd_fix_q <= leftShiftStage1Idx1Rng4_uid344_fracPostNormSub_uid84_block_rsrvd_fix_b & zs_uid199_lzCountValAdd_uid81_block_rsrvd_fix_q;

    -- leftShiftStage0Idx3Rng48_uid339_fracPostNormSub_uid84_block_rsrvd_fix(BITSELECT,338)@8
    leftShiftStage0Idx3Rng48_uid339_fracPostNormSub_uid84_block_rsrvd_fix_in <= redist23_fracResSubNoSignExt_uid80_block_rsrvd_fix_b_3_q(8 downto 0);
    leftShiftStage0Idx3Rng48_uid339_fracPostNormSub_uid84_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(leftShiftStage0Idx3Rng48_uid339_fracPostNormSub_uid84_block_rsrvd_fix_in(8 downto 0));

    -- leftShiftStage0Idx3_uid340_fracPostNormSub_uid84_block_rsrvd_fix(BITJOIN,339)@8
    leftShiftStage0Idx3_uid340_fracPostNormSub_uid84_block_rsrvd_fix_q <= leftShiftStage0Idx3Rng48_uid339_fracPostNormSub_uid84_block_rsrvd_fix_b & rightShiftStage2Idx3Pad48_uid284_alignmentShifter_uid65_block_rsrvd_fix_q;

    -- redist8_vStage_uid220_lzCountValSub_uid83_block_rsrvd_fix_b_2(DELAY,386)
    redist8_vStage_uid220_lzCountValSub_uid83_block_rsrvd_fix_b_2_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist8_vStage_uid220_lzCountValSub_uid83_block_rsrvd_fix_b_2_delay_0 <= STD_LOGIC_VECTOR(vStage_uid220_lzCountValSub_uid83_block_rsrvd_fix_b);
                redist8_vStage_uid220_lzCountValSub_uid83_block_rsrvd_fix_b_2_q <= STD_LOGIC_VECTOR(redist8_vStage_uid220_lzCountValSub_uid83_block_rsrvd_fix_b_2_delay_0);
            END IF;
        END IF;
    END PROCESS;

    -- leftShiftStage0Idx2_uid337_fracPostNormSub_uid84_block_rsrvd_fix(BITJOIN,336)@8
    leftShiftStage0Idx2_uid337_fracPostNormSub_uid84_block_rsrvd_fix_q <= redist8_vStage_uid220_lzCountValSub_uid83_block_rsrvd_fix_b_2_q & zs_uid179_lzCountValAdd_uid81_block_rsrvd_fix_q;

    -- leftShiftStage0Idx1Rng16_uid333_fracPostNormSub_uid84_block_rsrvd_fix(BITSELECT,332)@8
    leftShiftStage0Idx1Rng16_uid333_fracPostNormSub_uid84_block_rsrvd_fix_in <= redist23_fracResSubNoSignExt_uid80_block_rsrvd_fix_b_3_q(40 downto 0);
    leftShiftStage0Idx1Rng16_uid333_fracPostNormSub_uid84_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(leftShiftStage0Idx1Rng16_uid333_fracPostNormSub_uid84_block_rsrvd_fix_in(40 downto 0));

    -- leftShiftStage0Idx1_uid334_fracPostNormSub_uid84_block_rsrvd_fix(BITJOIN,333)@8
    leftShiftStage0Idx1_uid334_fracPostNormSub_uid84_block_rsrvd_fix_q <= leftShiftStage0Idx1Rng16_uid333_fracPostNormSub_uid84_block_rsrvd_fix_b & zs_uid187_lzCountValAdd_uid81_block_rsrvd_fix_q;

    -- redist23_fracResSubNoSignExt_uid80_block_rsrvd_fix_b_3(DELAY,401)
    redist23_fracResSubNoSignExt_uid80_block_rsrvd_fix_b_3_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist23_fracResSubNoSignExt_uid80_block_rsrvd_fix_b_3_delay_0 <= STD_LOGIC_VECTOR(redist22_fracResSubNoSignExt_uid80_block_rsrvd_fix_b_1_q);
                redist23_fracResSubNoSignExt_uid80_block_rsrvd_fix_b_3_q <= STD_LOGIC_VECTOR(redist23_fracResSubNoSignExt_uid80_block_rsrvd_fix_b_3_delay_0);
            END IF;
        END IF;
    END PROCESS;

    -- leftShiftStage0_uid342_fracPostNormSub_uid84_block_rsrvd_fix(MUX,341)@8
    leftShiftStage0_uid342_fracPostNormSub_uid84_block_rsrvd_fix_s <= leftShiftStageSel0Dto4_uid341_fracPostNormSub_uid84_block_rsrvd_fix_bit_select_merged_b;
    leftShiftStage0_uid342_fracPostNormSub_uid84_block_rsrvd_fix_combproc: PROCESS (leftShiftStage0_uid342_fracPostNormSub_uid84_block_rsrvd_fix_s, redist23_fracResSubNoSignExt_uid80_block_rsrvd_fix_b_3_q, leftShiftStage0Idx1_uid334_fracPostNormSub_uid84_block_rsrvd_fix_q, leftShiftStage0Idx2_uid337_fracPostNormSub_uid84_block_rsrvd_fix_q, leftShiftStage0Idx3_uid340_fracPostNormSub_uid84_block_rsrvd_fix_q)
    BEGIN
        CASE (leftShiftStage0_uid342_fracPostNormSub_uid84_block_rsrvd_fix_s) IS
            WHEN "00" => leftShiftStage0_uid342_fracPostNormSub_uid84_block_rsrvd_fix_q <= redist23_fracResSubNoSignExt_uid80_block_rsrvd_fix_b_3_q;
            WHEN "01" => leftShiftStage0_uid342_fracPostNormSub_uid84_block_rsrvd_fix_q <= leftShiftStage0Idx1_uid334_fracPostNormSub_uid84_block_rsrvd_fix_q;
            WHEN "10" => leftShiftStage0_uid342_fracPostNormSub_uid84_block_rsrvd_fix_q <= leftShiftStage0Idx2_uid337_fracPostNormSub_uid84_block_rsrvd_fix_q;
            WHEN "11" => leftShiftStage0_uid342_fracPostNormSub_uid84_block_rsrvd_fix_q <= leftShiftStage0Idx3_uid340_fracPostNormSub_uid84_block_rsrvd_fix_q;
            WHEN OTHERS => leftShiftStage0_uid342_fracPostNormSub_uid84_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- leftShiftStage1_uid353_fracPostNormSub_uid84_block_rsrvd_fix(MUX,352)@8
    leftShiftStage1_uid353_fracPostNormSub_uid84_block_rsrvd_fix_s <= leftShiftStageSel0Dto4_uid341_fracPostNormSub_uid84_block_rsrvd_fix_bit_select_merged_c;
    leftShiftStage1_uid353_fracPostNormSub_uid84_block_rsrvd_fix_combproc: PROCESS (leftShiftStage1_uid353_fracPostNormSub_uid84_block_rsrvd_fix_s, leftShiftStage0_uid342_fracPostNormSub_uid84_block_rsrvd_fix_q, leftShiftStage1Idx1_uid345_fracPostNormSub_uid84_block_rsrvd_fix_q, leftShiftStage1Idx2_uid348_fracPostNormSub_uid84_block_rsrvd_fix_q, leftShiftStage1Idx3_uid351_fracPostNormSub_uid84_block_rsrvd_fix_q)
    BEGIN
        CASE (leftShiftStage1_uid353_fracPostNormSub_uid84_block_rsrvd_fix_s) IS
            WHEN "00" => leftShiftStage1_uid353_fracPostNormSub_uid84_block_rsrvd_fix_q <= leftShiftStage0_uid342_fracPostNormSub_uid84_block_rsrvd_fix_q;
            WHEN "01" => leftShiftStage1_uid353_fracPostNormSub_uid84_block_rsrvd_fix_q <= leftShiftStage1Idx1_uid345_fracPostNormSub_uid84_block_rsrvd_fix_q;
            WHEN "10" => leftShiftStage1_uid353_fracPostNormSub_uid84_block_rsrvd_fix_q <= leftShiftStage1Idx2_uid348_fracPostNormSub_uid84_block_rsrvd_fix_q;
            WHEN "11" => leftShiftStage1_uid353_fracPostNormSub_uid84_block_rsrvd_fix_q <= leftShiftStage1Idx3_uid351_fracPostNormSub_uid84_block_rsrvd_fix_q;
            WHEN OTHERS => leftShiftStage1_uid353_fracPostNormSub_uid84_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- leftShiftStageSel0Dto4_uid341_fracPostNormSub_uid84_block_rsrvd_fix_bit_select_merged(BITSELECT,376)@8
    leftShiftStageSel0Dto4_uid341_fracPostNormSub_uid84_block_rsrvd_fix_bit_select_merged_b <= STD_LOGIC_VECTOR(r_uid251_lzCountValSub_uid83_block_rsrvd_fix_q(5 downto 4));
    leftShiftStageSel0Dto4_uid341_fracPostNormSub_uid84_block_rsrvd_fix_bit_select_merged_c <= STD_LOGIC_VECTOR(r_uid251_lzCountValSub_uid83_block_rsrvd_fix_q(3 downto 2));
    leftShiftStageSel0Dto4_uid341_fracPostNormSub_uid84_block_rsrvd_fix_bit_select_merged_d <= STD_LOGIC_VECTOR(r_uid251_lzCountValSub_uid83_block_rsrvd_fix_q(1 downto 0));

    -- leftShiftStage2_uid364_fracPostNormSub_uid84_block_rsrvd_fix(MUX,363)@8
    leftShiftStage2_uid364_fracPostNormSub_uid84_block_rsrvd_fix_s <= leftShiftStageSel0Dto4_uid341_fracPostNormSub_uid84_block_rsrvd_fix_bit_select_merged_d;
    leftShiftStage2_uid364_fracPostNormSub_uid84_block_rsrvd_fix_combproc: PROCESS (leftShiftStage2_uid364_fracPostNormSub_uid84_block_rsrvd_fix_s, leftShiftStage1_uid353_fracPostNormSub_uid84_block_rsrvd_fix_q, leftShiftStage2Idx1_uid356_fracPostNormSub_uid84_block_rsrvd_fix_q, leftShiftStage2Idx2_uid359_fracPostNormSub_uid84_block_rsrvd_fix_q, leftShiftStage2Idx3_uid362_fracPostNormSub_uid84_block_rsrvd_fix_q)
    BEGIN
        CASE (leftShiftStage2_uid364_fracPostNormSub_uid84_block_rsrvd_fix_s) IS
            WHEN "00" => leftShiftStage2_uid364_fracPostNormSub_uid84_block_rsrvd_fix_q <= leftShiftStage1_uid353_fracPostNormSub_uid84_block_rsrvd_fix_q;
            WHEN "01" => leftShiftStage2_uid364_fracPostNormSub_uid84_block_rsrvd_fix_q <= leftShiftStage2Idx1_uid356_fracPostNormSub_uid84_block_rsrvd_fix_q;
            WHEN "10" => leftShiftStage2_uid364_fracPostNormSub_uid84_block_rsrvd_fix_q <= leftShiftStage2Idx2_uid359_fracPostNormSub_uid84_block_rsrvd_fix_q;
            WHEN "11" => leftShiftStage2_uid364_fracPostNormSub_uid84_block_rsrvd_fix_q <= leftShiftStage2Idx3_uid362_fracPostNormSub_uid84_block_rsrvd_fix_q;
            WHEN OTHERS => leftShiftStage2_uid364_fracPostNormSub_uid84_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- LSB_sub_uid108_block_rsrvd_fix(BITSELECT,107)@8
    LSB_sub_uid108_block_rsrvd_fix_in <= STD_LOGIC_VECTOR(leftShiftStage2_uid364_fracPostNormSub_uid84_block_rsrvd_fix_q(4 downto 0));
    LSB_sub_uid108_block_rsrvd_fix_b <= LSB_sub_uid108_block_rsrvd_fix_in(4 downto 4);

    -- Guard_sub_uid107_block_rsrvd_fix(BITSELECT,106)@8
    Guard_sub_uid107_block_rsrvd_fix_in <= STD_LOGIC_VECTOR(leftShiftStage2_uid364_fracPostNormSub_uid84_block_rsrvd_fix_q(3 downto 0));
    Guard_sub_uid107_block_rsrvd_fix_b <= Guard_sub_uid107_block_rsrvd_fix_in(3 downto 3);

    -- Round_sub_uid106_block_rsrvd_fix(BITSELECT,105)@8
    Round_sub_uid106_block_rsrvd_fix_in <= STD_LOGIC_VECTOR(leftShiftStage2_uid364_fracPostNormSub_uid84_block_rsrvd_fix_q(2 downto 0));
    Round_sub_uid106_block_rsrvd_fix_b <= Round_sub_uid106_block_rsrvd_fix_in(2 downto 2);

    -- Sticky1_sub_uid105_block_rsrvd_fix(BITSELECT,104)@8
    Sticky1_sub_uid105_block_rsrvd_fix_in <= STD_LOGIC_VECTOR(leftShiftStage2_uid364_fracPostNormSub_uid84_block_rsrvd_fix_q(1 downto 0));
    Sticky1_sub_uid105_block_rsrvd_fix_b <= Sticky1_sub_uid105_block_rsrvd_fix_in(1 downto 1);

    -- Sticky0_sub_uid104_block_rsrvd_fix(BITSELECT,103)@8
    Sticky0_sub_uid104_block_rsrvd_fix_in <= STD_LOGIC_VECTOR(leftShiftStage2_uid364_fracPostNormSub_uid84_block_rsrvd_fix_q(0 downto 0));
    Sticky0_sub_uid104_block_rsrvd_fix_b <= Sticky0_sub_uid104_block_rsrvd_fix_in(0 downto 0);

    -- rndBitCond_sub_uid109_block_rsrvd_fix(BITJOIN,108)@8
    rndBitCond_sub_uid109_block_rsrvd_fix_q <= LSB_sub_uid108_block_rsrvd_fix_b & Guard_sub_uid107_block_rsrvd_fix_b & Round_sub_uid106_block_rsrvd_fix_b & Sticky1_sub_uid105_block_rsrvd_fix_b & Sticky0_sub_uid104_block_rsrvd_fix_b;

    -- rBi_sub_uid110_block_rsrvd_fix(LOGICAL,109)@8 + 1
    rBi_sub_uid110_block_rsrvd_fix_qi <= "1" WHEN rndBitCond_sub_uid109_block_rsrvd_fix_q = cRBit_uid100_block_rsrvd_fix_q ELSE "0";
    rBi_sub_uid110_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => rBi_sub_uid110_block_rsrvd_fix_qi, xout => rBi_sub_uid110_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- roundBit_sub_uid111_block_rsrvd_fix(LOGICAL,110)@9
    roundBit_sub_uid111_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (rBi_sub_uid110_block_rsrvd_fix_q));

    -- expPostNormSub_uid88_block_rsrvd_fix(SUB,87)@8 + 1
    expPostNormSub_uid88_block_rsrvd_fix_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("0" & expInc_uid87_block_rsrvd_fix_q));
    expPostNormSub_uid88_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("0000000" & r_uid251_lzCountValSub_uid83_block_rsrvd_fix_q));
    expPostNormSub_uid88_block_rsrvd_fix_clkproc: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                expPostNormSub_uid88_block_rsrvd_fix_o <= STD_LOGIC_VECTOR(SIGNED(expPostNormSub_uid88_block_rsrvd_fix_a) - SIGNED(expPostNormSub_uid88_block_rsrvd_fix_b));
            END IF;
        END IF;
    END PROCESS;
    expPostNormSub_uid88_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(expPostNormSub_uid88_block_rsrvd_fix_o(12 downto 0));

    -- fracPostNormSubRndRange_uid90_block_rsrvd_fix(BITSELECT,89)@8
    fracPostNormSubRndRange_uid90_block_rsrvd_fix_in <= leftShiftStage2_uid364_fracPostNormSub_uid84_block_rsrvd_fix_q(55 downto 0);
    fracPostNormSubRndRange_uid90_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(fracPostNormSubRndRange_uid90_block_rsrvd_fix_in(55 downto 3));

    -- redist20_fracPostNormSubRndRange_uid90_block_rsrvd_fix_b_1(DELAY,398)
    redist20_fracPostNormSubRndRange_uid90_block_rsrvd_fix_b_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist20_fracPostNormSubRndRange_uid90_block_rsrvd_fix_b_1_q <= fracPostNormSubRndRange_uid90_block_rsrvd_fix_b;
            END IF;
        END IF;
    END PROCESS;

    -- expFracRSub_uid91_block_rsrvd_fix(BITJOIN,90)@9
    expFracRSub_uid91_block_rsrvd_fix_q <= expPostNormSub_uid88_block_rsrvd_fix_q & redist20_fracPostNormSubRndRange_uid90_block_rsrvd_fix_b_1_q;

    -- expFracRSubPostRound_uid112_block_rsrvd_fix(ADD,111)@9 + 1
    expFracRSubPostRound_uid112_block_rsrvd_fix_a <= STD_LOGIC_VECTOR("0" & expFracRSub_uid91_block_rsrvd_fix_q);
    expFracRSubPostRound_uid112_block_rsrvd_fix_b <= STD_LOGIC_VECTOR("000000000000000000000000000000000000000000000000000000000000000000" & roundBit_sub_uid111_block_rsrvd_fix_q);
    expFracRSubPostRound_uid112_block_rsrvd_fix_clkproc: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                expFracRSubPostRound_uid112_block_rsrvd_fix_o <= STD_LOGIC_VECTOR(UNSIGNED(expFracRSubPostRound_uid112_block_rsrvd_fix_a) + UNSIGNED(expFracRSubPostRound_uid112_block_rsrvd_fix_b));
            END IF;
        END IF;
    END PROCESS;
    expFracRSubPostRound_uid112_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(expFracRSubPostRound_uid112_block_rsrvd_fix_o(66 downto 0));

    -- expRPreExcSub_uid122_block_rsrvd_fix(BITSELECT,121)@10
    expRPreExcSub_uid122_block_rsrvd_fix_in <= expFracRSubPostRound_uid112_block_rsrvd_fix_q(63 downto 0);
    expRPreExcSub_uid122_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(expRPreExcSub_uid122_block_rsrvd_fix_in(63 downto 53));

    -- expRPreExcSubtraction_uid140_block_rsrvd_fix(MUX,139)@10 + 1
    expRPreExcSubtraction_uid140_block_rsrvd_fix_s <= effSub_uid48_block_rsrvd_fix_q;
    expRPreExcSubtraction_uid140_block_rsrvd_fix_clkproc: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                CASE (expRPreExcSubtraction_uid140_block_rsrvd_fix_s) IS
                    WHEN "0" => expRPreExcSubtraction_uid140_block_rsrvd_fix_q <= expRPreExcSub_uid122_block_rsrvd_fix_b;
                    WHEN "1" => expRPreExcSubtraction_uid140_block_rsrvd_fix_q <= redist17_expRPreExcAdd_uid119_block_rsrvd_fix_b_1_q;
                    WHEN OTHERS => expRPreExcSubtraction_uid140_block_rsrvd_fix_q <= (others => '0');
                END CASE;
            END IF;
        END IF;
    END PROCESS;

    -- wEP2AllOwE_uid113_block_rsrvd_fix(CONSTANT,112)
    wEP2AllOwE_uid113_block_rsrvd_fix_q <= "0011111111111";

    -- rndExp_uid114_block_rsrvd_fix(BITSELECT,113)@9
    rndExp_uid114_block_rsrvd_fix_in <= expFracRAddPostRound_uid103_block_rsrvd_fix_q(65 downto 0);
    rndExp_uid114_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(rndExp_uid114_block_rsrvd_fix_in(65 downto 53));

    -- rOvf_uid115_block_rsrvd_fix(LOGICAL,114)@9 + 1
    rOvf_uid115_block_rsrvd_fix_qi <= "1" WHEN rndExp_uid114_block_rsrvd_fix_b = wEP2AllOwE_uid113_block_rsrvd_fix_q ELSE "0";
    rOvf_uid115_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => rOvf_uid115_block_rsrvd_fix_qi, xout => rOvf_uid115_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- excRInfVInC_uid129_block_rsrvd_fix(BITJOIN,128)@10
    excRInfVInC_uid129_block_rsrvd_fix_q <= effSub_uid48_block_rsrvd_fix_q & redist15_oneIsNaN_uid128_block_rsrvd_fix_q_3_q & rOvf_uid115_block_rsrvd_fix_q & redist16_regInputs_uid124_block_rsrvd_fix_q_3_q & redist29_excI_sigb_uid37_block_rsrvd_fix_q_3_q & redist36_excI_siga_uid23_block_rsrvd_fix_q_3_q;

    -- excRInfSub_uid131_block_rsrvd_fix(LOOKUP,130)@10
    excRInfSub_uid131_block_rsrvd_fix_combproc: PROCESS (excRInfVInC_uid129_block_rsrvd_fix_q)
    BEGIN
        -- Begin reserved scope level
        CASE (excRInfVInC_uid129_block_rsrvd_fix_q) IS
            WHEN "000000" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "000001" => excRInfSub_uid131_block_rsrvd_fix_q <= "1";
            WHEN "000010" => excRInfSub_uid131_block_rsrvd_fix_q <= "1";
            WHEN "000011" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "000100" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "000101" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "000110" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "000111" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "001000" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "001001" => excRInfSub_uid131_block_rsrvd_fix_q <= "1";
            WHEN "001010" => excRInfSub_uid131_block_rsrvd_fix_q <= "1";
            WHEN "001011" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "001100" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "001101" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "001110" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "001111" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "010000" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "010001" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "010010" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "010011" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "010100" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "010101" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "010110" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "010111" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "011000" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "011001" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "011010" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "011011" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "011100" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "011101" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "011110" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "011111" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "100000" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "100001" => excRInfSub_uid131_block_rsrvd_fix_q <= "1";
            WHEN "100010" => excRInfSub_uid131_block_rsrvd_fix_q <= "1";
            WHEN "100011" => excRInfSub_uid131_block_rsrvd_fix_q <= "1";
            WHEN "100100" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "100101" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "100110" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "100111" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "101000" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "101001" => excRInfSub_uid131_block_rsrvd_fix_q <= "1";
            WHEN "101010" => excRInfSub_uid131_block_rsrvd_fix_q <= "1";
            WHEN "101011" => excRInfSub_uid131_block_rsrvd_fix_q <= "1";
            WHEN "101100" => excRInfSub_uid131_block_rsrvd_fix_q <= "1";
            WHEN "101101" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "101110" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "101111" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "110000" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "110001" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "110010" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "110011" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "110100" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "110101" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "110110" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "110111" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "111000" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "111001" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "111010" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "111011" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "111100" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "111101" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "111110" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN "111111" => excRInfSub_uid131_block_rsrvd_fix_q <= "0";
            WHEN OTHERS => -- unreachable
                           excRInfSub_uid131_block_rsrvd_fix_q <= (others => '-');
        END CASE;
        -- End reserved scope level
    END PROCESS;

    -- signedExp_uid116_block_rsrvd_fix(BITSELECT,115)@10
    signedExp_uid116_block_rsrvd_fix_in <= STD_LOGIC_VECTOR(expFracRSubPostRound_uid112_block_rsrvd_fix_q(65 downto 0));
    signedExp_uid116_block_rsrvd_fix_b <= signedExp_uid116_block_rsrvd_fix_in(65 downto 53);

    -- rUdf_uid117_block_rsrvd_fix(COMPARE,116)@10
    rUdf_uid117_block_rsrvd_fix_a <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR("00000000000000" & GND_q));
    rUdf_uid117_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(STD_LOGIC_VECTOR((14 downto 13 => signedExp_uid116_block_rsrvd_fix_b(12)) & signedExp_uid116_block_rsrvd_fix_b));
    rUdf_uid117_block_rsrvd_fix_o <= STD_LOGIC_VECTOR(SIGNED(rUdf_uid117_block_rsrvd_fix_a) - SIGNED(rUdf_uid117_block_rsrvd_fix_b));
    rUdf_uid117_block_rsrvd_fix_n(0) <= not (rUdf_uid117_block_rsrvd_fix_o(14));

    -- excRZeroVInC_uid125_block_rsrvd_fix(BITJOIN,124)@10
    excRZeroVInC_uid125_block_rsrvd_fix_q <= effSub_uid48_block_rsrvd_fix_q & redist21_aMinusA_uid86_block_rsrvd_fix_q_2_q & rUdf_uid117_block_rsrvd_fix_n & redist16_regInputs_uid124_block_rsrvd_fix_q_3_q & redist33_excZ_sigb_uid13_uid33_block_rsrvd_fix_q_9_q & redist38_excZ_siga_uid12_uid19_block_rsrvd_fix_q_3_q;

    -- excRZeroSub_uid127_block_rsrvd_fix(LOOKUP,126)@10
    excRZeroSub_uid127_block_rsrvd_fix_combproc: PROCESS (excRZeroVInC_uid125_block_rsrvd_fix_q)
    BEGIN
        -- Begin reserved scope level
        CASE (excRZeroVInC_uid125_block_rsrvd_fix_q) IS
            WHEN "000000" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "000001" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "000010" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "000011" => excRZeroSub_uid127_block_rsrvd_fix_q <= "1";
            WHEN "000100" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "000101" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "000110" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "000111" => excRZeroSub_uid127_block_rsrvd_fix_q <= "1";
            WHEN "001000" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "001001" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "001010" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "001011" => excRZeroSub_uid127_block_rsrvd_fix_q <= "1";
            WHEN "001100" => excRZeroSub_uid127_block_rsrvd_fix_q <= "1";
            WHEN "001101" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "001110" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "001111" => excRZeroSub_uid127_block_rsrvd_fix_q <= "1";
            WHEN "010000" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "010001" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "010010" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "010011" => excRZeroSub_uid127_block_rsrvd_fix_q <= "1";
            WHEN "010100" => excRZeroSub_uid127_block_rsrvd_fix_q <= "1";
            WHEN "010101" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "010110" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "010111" => excRZeroSub_uid127_block_rsrvd_fix_q <= "1";
            WHEN "011000" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "011001" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "011010" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "011011" => excRZeroSub_uid127_block_rsrvd_fix_q <= "1";
            WHEN "011100" => excRZeroSub_uid127_block_rsrvd_fix_q <= "1";
            WHEN "011101" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "011110" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "011111" => excRZeroSub_uid127_block_rsrvd_fix_q <= "1";
            WHEN "100000" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "100001" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "100010" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "100011" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "100100" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "100101" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "100110" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "100111" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "101000" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "101001" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "101010" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "101011" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "101100" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "101101" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "101110" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "101111" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "110000" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "110001" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "110010" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "110011" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "110100" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "110101" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "110110" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "110111" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "111000" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "111001" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "111010" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "111011" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "111100" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "111101" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "111110" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN "111111" => excRZeroSub_uid127_block_rsrvd_fix_q <= "0";
            WHEN OTHERS => -- unreachable
                           excRZeroSub_uid127_block_rsrvd_fix_q <= (others => '-');
        END CASE;
        -- End reserved scope level
    END PROCESS;

    -- concExcSub_uid141_block_rsrvd_fix(BITJOIN,140)@10
    concExcSub_uid141_block_rsrvd_fix_q <= excRNaNS_uid136_block_rsrvd_fix_q & excRInfSub_uid131_block_rsrvd_fix_q & excRZeroSub_uid127_block_rsrvd_fix_q;

    -- excREncSub_uid143_block_rsrvd_fix(LOOKUP,142)@10 + 1
    excREncSub_uid143_block_rsrvd_fix_clkproc: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                CASE (concExcSub_uid141_block_rsrvd_fix_q) IS
                    WHEN "000" => excREncSub_uid143_block_rsrvd_fix_q <= "01";
                    WHEN "001" => excREncSub_uid143_block_rsrvd_fix_q <= "00";
                    WHEN "010" => excREncSub_uid143_block_rsrvd_fix_q <= "10";
                    WHEN "011" => excREncSub_uid143_block_rsrvd_fix_q <= "00";
                    WHEN "100" => excREncSub_uid143_block_rsrvd_fix_q <= "11";
                    WHEN "101" => excREncSub_uid143_block_rsrvd_fix_q <= "00";
                    WHEN "110" => excREncSub_uid143_block_rsrvd_fix_q <= "00";
                    WHEN "111" => excREncSub_uid143_block_rsrvd_fix_q <= "00";
                    WHEN OTHERS => -- unreachable
                                   excREncSub_uid143_block_rsrvd_fix_q <= (others => '-');
                END CASE;
            END IF;
        END IF;
    END PROCESS;

    -- expRPostExcSub_uid167_block_rsrvd_fix(MUX,166)@11
    expRPostExcSub_uid167_block_rsrvd_fix_s <= excREncSub_uid143_block_rsrvd_fix_q;
    expRPostExcSub_uid167_block_rsrvd_fix_combproc: PROCESS (expRPostExcSub_uid167_block_rsrvd_fix_s, cstAllZWE_uid16_block_rsrvd_fix_q, expRPreExcSubtraction_uid140_block_rsrvd_fix_q, cstAllOWE_uid14_block_rsrvd_fix_q)
    BEGIN
        CASE (expRPostExcSub_uid167_block_rsrvd_fix_s) IS
            WHEN "00" => expRPostExcSub_uid167_block_rsrvd_fix_q <= cstAllZWE_uid16_block_rsrvd_fix_q;
            WHEN "01" => expRPostExcSub_uid167_block_rsrvd_fix_q <= expRPreExcSubtraction_uid140_block_rsrvd_fix_q;
            WHEN "10" => expRPostExcSub_uid167_block_rsrvd_fix_q <= cstAllOWE_uid14_block_rsrvd_fix_q;
            WHEN "11" => expRPostExcSub_uid167_block_rsrvd_fix_q <= cstAllOWE_uid14_block_rsrvd_fix_q;
            WHEN OTHERS => expRPostExcSub_uid167_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- oneFracRPostExc2_uid145_block_rsrvd_fix(CONSTANT,144)
    oneFracRPostExc2_uid145_block_rsrvd_fix_q <= "0000000000000000000000000000000000000000000000000001";

    -- fracRPreExcAdd_uid118_block_rsrvd_fix(BITSELECT,117)@9
    fracRPreExcAdd_uid118_block_rsrvd_fix_in <= expFracRAddPostRound_uid103_block_rsrvd_fix_q(52 downto 0);
    fracRPreExcAdd_uid118_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(fracRPreExcAdd_uid118_block_rsrvd_fix_in(52 downto 1));

    -- redist18_fracRPreExcAdd_uid118_block_rsrvd_fix_b_1(DELAY,396)
    redist18_fracRPreExcAdd_uid118_block_rsrvd_fix_b_1_clkproc_0: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                redist18_fracRPreExcAdd_uid118_block_rsrvd_fix_b_1_q <= fracRPreExcAdd_uid118_block_rsrvd_fix_b;
            END IF;
        END IF;
    END PROCESS;

    -- fracRPreExcSub_uid121_block_rsrvd_fix(BITSELECT,120)@10
    fracRPreExcSub_uid121_block_rsrvd_fix_in <= expFracRSubPostRound_uid112_block_rsrvd_fix_q(52 downto 0);
    fracRPreExcSub_uid121_block_rsrvd_fix_b <= STD_LOGIC_VECTOR(fracRPreExcSub_uid121_block_rsrvd_fix_in(52 downto 1));

    -- fracRPreExcSubtraction_uid139_block_rsrvd_fix(MUX,138)@10 + 1
    fracRPreExcSubtraction_uid139_block_rsrvd_fix_s <= effSub_uid48_block_rsrvd_fix_q;
    fracRPreExcSubtraction_uid139_block_rsrvd_fix_clkproc: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                CASE (fracRPreExcSubtraction_uid139_block_rsrvd_fix_s) IS
                    WHEN "0" => fracRPreExcSubtraction_uid139_block_rsrvd_fix_q <= fracRPreExcSub_uid121_block_rsrvd_fix_b;
                    WHEN "1" => fracRPreExcSubtraction_uid139_block_rsrvd_fix_q <= redist18_fracRPreExcAdd_uid118_block_rsrvd_fix_b_1_q;
                    WHEN OTHERS => fracRPreExcSubtraction_uid139_block_rsrvd_fix_q <= (others => '0');
                END CASE;
            END IF;
        END IF;
    END PROCESS;

    -- fracRPostExcSub_uid163_block_rsrvd_fix(MUX,162)@11
    fracRPostExcSub_uid163_block_rsrvd_fix_s <= excREncSub_uid143_block_rsrvd_fix_q;
    fracRPostExcSub_uid163_block_rsrvd_fix_combproc: PROCESS (fracRPostExcSub_uid163_block_rsrvd_fix_s, cstZeroWF_uid15_block_rsrvd_fix_q, fracRPreExcSubtraction_uid139_block_rsrvd_fix_q, oneFracRPostExc2_uid145_block_rsrvd_fix_q)
    BEGIN
        CASE (fracRPostExcSub_uid163_block_rsrvd_fix_s) IS
            WHEN "00" => fracRPostExcSub_uid163_block_rsrvd_fix_q <= cstZeroWF_uid15_block_rsrvd_fix_q;
            WHEN "01" => fracRPostExcSub_uid163_block_rsrvd_fix_q <= fracRPreExcSubtraction_uid139_block_rsrvd_fix_q;
            WHEN "10" => fracRPostExcSub_uid163_block_rsrvd_fix_q <= cstZeroWF_uid15_block_rsrvd_fix_q;
            WHEN "11" => fracRPostExcSub_uid163_block_rsrvd_fix_q <= oneFracRPostExc2_uid145_block_rsrvd_fix_q;
            WHEN OTHERS => fracRPostExcSub_uid163_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- RDiff_uid176_block_rsrvd_fix(BITJOIN,175)@11
    RDiff_uid176_block_rsrvd_fix_q <= signRPostExcSub_uid175_block_rsrvd_fix_q & expRPostExcSub_uid167_block_rsrvd_fix_q & fracRPostExcSub_uid163_block_rsrvd_fix_q;

    -- out_primWireAux(GPOUT,5)@11
    out_primWireAux <= RDiff_uid176_block_rsrvd_fix_q;

    -- zMz_uid153_block_rsrvd_fix(LOGICAL,152)@10
    zMz_uid153_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(redist38_excZ_siga_uid12_uid19_block_rsrvd_fix_q_3_q and redist33_excZ_sigb_uid13_uid33_block_rsrvd_fix_q_9_q and effSub_uid48_block_rsrvd_fix_q);

    -- invZMZ_uid154_block_rsrvd_fix(LOGICAL,153)@10
    invZMZ_uid154_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (zMz_uid153_block_rsrvd_fix_q));

    -- aMa_uid155_block_rsrvd_fix(LOGICAL,154)@10
    aMa_uid155_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(redist16_regInputs_uid124_block_rsrvd_fix_q_3_q and redist21_aMinusA_uid86_block_rsrvd_fix_q_2_q and effSub_uid48_block_rsrvd_fix_q);

    -- invAMA_uid156_block_rsrvd_fix(LOGICAL,155)@10
    invAMA_uid156_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (aMa_uid155_block_rsrvd_fix_q));

    -- infMinf_uid132_block_rsrvd_fix(LOGICAL,131)@10
    infMinf_uid132_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(redist36_excI_siga_uid23_block_rsrvd_fix_q_3_q and redist29_excI_sigb_uid37_block_rsrvd_fix_q_3_q and effSub_uid48_block_rsrvd_fix_q);

    -- excRNaNA_uid133_block_rsrvd_fix(LOGICAL,132)@10
    excRNaNA_uid133_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(infMinf_uid132_block_rsrvd_fix_q or redist15_oneIsNaN_uid128_block_rsrvd_fix_q_3_q);

    -- invExcRNaNA_uid157_block_rsrvd_fix(LOGICAL,156)@10
    invExcRNaNA_uid157_block_rsrvd_fix_q <= STD_LOGIC_VECTOR(not (excRNaNA_uid133_block_rsrvd_fix_q));

    -- signRPostExcAdd_uid158_block_rsrvd_fix(LOGICAL,157)@10 + 1
    signRPostExcAdd_uid158_block_rsrvd_fix_qi <= invExcRNaNA_uid157_block_rsrvd_fix_q and redist27_sigA_uid46_block_rsrvd_fix_b_10_q and invAMA_uid156_block_rsrvd_fix_q and invZMZ_uid154_block_rsrvd_fix_q;
    signRPostExcAdd_uid158_block_rsrvd_fix_delay : dspba_delay
    GENERIC MAP ( width => 1, depth => 1, reset_kind => "NONE", phase => 0, modulus => 1 )
    PORT MAP ( xin => signRPostExcAdd_uid158_block_rsrvd_fix_qi, xout => signRPostExcAdd_uid158_block_rsrvd_fix_q, clk => clk, aclr => areset, ena => '1' );

    -- expRPreExcAddition_uid138_block_rsrvd_fix(MUX,137)@10 + 1
    expRPreExcAddition_uid138_block_rsrvd_fix_s <= effSub_uid48_block_rsrvd_fix_q;
    expRPreExcAddition_uid138_block_rsrvd_fix_clkproc: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                CASE (expRPreExcAddition_uid138_block_rsrvd_fix_s) IS
                    WHEN "0" => expRPreExcAddition_uid138_block_rsrvd_fix_q <= redist17_expRPreExcAdd_uid119_block_rsrvd_fix_b_1_q;
                    WHEN "1" => expRPreExcAddition_uid138_block_rsrvd_fix_q <= expRPreExcSub_uid122_block_rsrvd_fix_b;
                    WHEN OTHERS => expRPreExcAddition_uid138_block_rsrvd_fix_q <= (others => '0');
                END CASE;
            END IF;
        END IF;
    END PROCESS;

    -- excRInfAdd_uid130_block_rsrvd_fix(LOOKUP,129)@10
    excRInfAdd_uid130_block_rsrvd_fix_combproc: PROCESS (excRInfVInC_uid129_block_rsrvd_fix_q)
    BEGIN
        -- Begin reserved scope level
        CASE (excRInfVInC_uid129_block_rsrvd_fix_q) IS
            WHEN "000000" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "000001" => excRInfAdd_uid130_block_rsrvd_fix_q <= "1";
            WHEN "000010" => excRInfAdd_uid130_block_rsrvd_fix_q <= "1";
            WHEN "000011" => excRInfAdd_uid130_block_rsrvd_fix_q <= "1";
            WHEN "000100" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "000101" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "000110" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "000111" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "001000" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "001001" => excRInfAdd_uid130_block_rsrvd_fix_q <= "1";
            WHEN "001010" => excRInfAdd_uid130_block_rsrvd_fix_q <= "1";
            WHEN "001011" => excRInfAdd_uid130_block_rsrvd_fix_q <= "1";
            WHEN "001100" => excRInfAdd_uid130_block_rsrvd_fix_q <= "1";
            WHEN "001101" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "001110" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "001111" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "010000" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "010001" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "010010" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "010011" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "010100" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "010101" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "010110" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "010111" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "011000" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "011001" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "011010" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "011011" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "011100" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "011101" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "011110" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "011111" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "100000" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "100001" => excRInfAdd_uid130_block_rsrvd_fix_q <= "1";
            WHEN "100010" => excRInfAdd_uid130_block_rsrvd_fix_q <= "1";
            WHEN "100011" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "100100" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "100101" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "100110" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "100111" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "101000" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "101001" => excRInfAdd_uid130_block_rsrvd_fix_q <= "1";
            WHEN "101010" => excRInfAdd_uid130_block_rsrvd_fix_q <= "1";
            WHEN "101011" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "101100" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "101101" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "101110" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "101111" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "110000" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "110001" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "110010" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "110011" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "110100" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "110101" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "110110" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "110111" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "111000" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "111001" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "111010" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "111011" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "111100" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "111101" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "111110" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN "111111" => excRInfAdd_uid130_block_rsrvd_fix_q <= "0";
            WHEN OTHERS => -- unreachable
                           excRInfAdd_uid130_block_rsrvd_fix_q <= (others => '-');
        END CASE;
        -- End reserved scope level
    END PROCESS;

    -- excRZeroAdd_uid126_block_rsrvd_fix(LOOKUP,125)@10
    excRZeroAdd_uid126_block_rsrvd_fix_combproc: PROCESS (excRZeroVInC_uid125_block_rsrvd_fix_q)
    BEGIN
        -- Begin reserved scope level
        CASE (excRZeroVInC_uid125_block_rsrvd_fix_q) IS
            WHEN "000000" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "000001" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "000010" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "000011" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "1";
            WHEN "000100" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "000101" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "000110" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "000111" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "1";
            WHEN "001000" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "001001" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "001010" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "001011" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "1";
            WHEN "001100" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "001101" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "001110" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "001111" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "1";
            WHEN "010000" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "010001" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "010010" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "010011" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "1";
            WHEN "010100" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "010101" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "010110" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "010111" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "1";
            WHEN "011000" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "011001" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "011010" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "011011" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "1";
            WHEN "011100" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "011101" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "011110" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "011111" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "1";
            WHEN "100000" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "100001" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "100010" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "100011" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "100100" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "100101" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "100110" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "100111" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "101000" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "101001" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "101010" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "101011" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "101100" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "1";
            WHEN "101101" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "101110" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "101111" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "110000" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "110001" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "110010" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "110011" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "110100" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "1";
            WHEN "110101" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "110110" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "110111" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "111000" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "111001" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "111010" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "111011" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "111100" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "1";
            WHEN "111101" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "111110" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN "111111" => excRZeroAdd_uid126_block_rsrvd_fix_q <= "0";
            WHEN OTHERS => -- unreachable
                           excRZeroAdd_uid126_block_rsrvd_fix_q <= (others => '-');
        END CASE;
        -- End reserved scope level
    END PROCESS;

    -- concExcAdd_uid142_block_rsrvd_fix(BITJOIN,141)@10
    concExcAdd_uid142_block_rsrvd_fix_q <= excRNaNA_uid133_block_rsrvd_fix_q & excRInfAdd_uid130_block_rsrvd_fix_q & excRZeroAdd_uid126_block_rsrvd_fix_q;

    -- excREncAdd_uid144_block_rsrvd_fix(LOOKUP,143)@10 + 1
    excREncAdd_uid144_block_rsrvd_fix_clkproc: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                CASE (concExcAdd_uid142_block_rsrvd_fix_q) IS
                    WHEN "000" => excREncAdd_uid144_block_rsrvd_fix_q <= "01";
                    WHEN "001" => excREncAdd_uid144_block_rsrvd_fix_q <= "00";
                    WHEN "010" => excREncAdd_uid144_block_rsrvd_fix_q <= "10";
                    WHEN "011" => excREncAdd_uid144_block_rsrvd_fix_q <= "00";
                    WHEN "100" => excREncAdd_uid144_block_rsrvd_fix_q <= "11";
                    WHEN "101" => excREncAdd_uid144_block_rsrvd_fix_q <= "00";
                    WHEN "110" => excREncAdd_uid144_block_rsrvd_fix_q <= "00";
                    WHEN "111" => excREncAdd_uid144_block_rsrvd_fix_q <= "00";
                    WHEN OTHERS => -- unreachable
                                   excREncAdd_uid144_block_rsrvd_fix_q <= (others => '-');
                END CASE;
            END IF;
        END IF;
    END PROCESS;

    -- expRPostExcAdd_uid152_block_rsrvd_fix(MUX,151)@11
    expRPostExcAdd_uid152_block_rsrvd_fix_s <= excREncAdd_uid144_block_rsrvd_fix_q;
    expRPostExcAdd_uid152_block_rsrvd_fix_combproc: PROCESS (expRPostExcAdd_uid152_block_rsrvd_fix_s, cstAllZWE_uid16_block_rsrvd_fix_q, expRPreExcAddition_uid138_block_rsrvd_fix_q, cstAllOWE_uid14_block_rsrvd_fix_q)
    BEGIN
        CASE (expRPostExcAdd_uid152_block_rsrvd_fix_s) IS
            WHEN "00" => expRPostExcAdd_uid152_block_rsrvd_fix_q <= cstAllZWE_uid16_block_rsrvd_fix_q;
            WHEN "01" => expRPostExcAdd_uid152_block_rsrvd_fix_q <= expRPreExcAddition_uid138_block_rsrvd_fix_q;
            WHEN "10" => expRPostExcAdd_uid152_block_rsrvd_fix_q <= cstAllOWE_uid14_block_rsrvd_fix_q;
            WHEN "11" => expRPostExcAdd_uid152_block_rsrvd_fix_q <= cstAllOWE_uid14_block_rsrvd_fix_q;
            WHEN OTHERS => expRPostExcAdd_uid152_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- fracRPreExcAddition_uid137_block_rsrvd_fix(MUX,136)@10 + 1
    fracRPreExcAddition_uid137_block_rsrvd_fix_s <= effSub_uid48_block_rsrvd_fix_q;
    fracRPreExcAddition_uid137_block_rsrvd_fix_clkproc: PROCESS (clk)
    BEGIN
        IF (clk'EVENT AND clk = '1') THEN
            IF (false) THEN
            ELSE
                CASE (fracRPreExcAddition_uid137_block_rsrvd_fix_s) IS
                    WHEN "0" => fracRPreExcAddition_uid137_block_rsrvd_fix_q <= redist18_fracRPreExcAdd_uid118_block_rsrvd_fix_b_1_q;
                    WHEN "1" => fracRPreExcAddition_uid137_block_rsrvd_fix_q <= fracRPreExcSub_uid121_block_rsrvd_fix_b;
                    WHEN OTHERS => fracRPreExcAddition_uid137_block_rsrvd_fix_q <= (others => '0');
                END CASE;
            END IF;
        END IF;
    END PROCESS;

    -- fracRPostExcAdd_uid148_block_rsrvd_fix(MUX,147)@11
    fracRPostExcAdd_uid148_block_rsrvd_fix_s <= excREncAdd_uid144_block_rsrvd_fix_q;
    fracRPostExcAdd_uid148_block_rsrvd_fix_combproc: PROCESS (fracRPostExcAdd_uid148_block_rsrvd_fix_s, cstZeroWF_uid15_block_rsrvd_fix_q, fracRPreExcAddition_uid137_block_rsrvd_fix_q, oneFracRPostExc2_uid145_block_rsrvd_fix_q)
    BEGIN
        CASE (fracRPostExcAdd_uid148_block_rsrvd_fix_s) IS
            WHEN "00" => fracRPostExcAdd_uid148_block_rsrvd_fix_q <= cstZeroWF_uid15_block_rsrvd_fix_q;
            WHEN "01" => fracRPostExcAdd_uid148_block_rsrvd_fix_q <= fracRPreExcAddition_uid137_block_rsrvd_fix_q;
            WHEN "10" => fracRPostExcAdd_uid148_block_rsrvd_fix_q <= cstZeroWF_uid15_block_rsrvd_fix_q;
            WHEN "11" => fracRPostExcAdd_uid148_block_rsrvd_fix_q <= oneFracRPostExc2_uid145_block_rsrvd_fix_q;
            WHEN OTHERS => fracRPostExcAdd_uid148_block_rsrvd_fix_q <= (others => '0');
        END CASE;
    END PROCESS;

    -- RSum_uid159_block_rsrvd_fix(BITJOIN,158)@11
    RSum_uid159_block_rsrvd_fix_q <= signRPostExcAdd_uid158_block_rsrvd_fix_q & expRPostExcAdd_uid152_block_rsrvd_fix_q & fracRPostExcAdd_uid148_block_rsrvd_fix_q;

    -- out_primWireOut(GPOUT,6)@11
    out_primWireOut <= RSum_uid159_block_rsrvd_fix_q;

END normal;
