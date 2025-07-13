onerror {resume}

# obtain Modelsim version and extract the NN.Nc part after vsim
quietly set vsim_ver [regexp -inline {vsim (\d+\.\d+)(\w?)} [vsim -version]]
quietly set has_fixpt_radix 0
if {[lindex $vsim_ver 1] == 10.2} {
    if {[lindex $vsim_ver 2] >= "d"} {
        quietly set has_fixpt_radix 1
    }
} elseif {[lindex $vsim_ver 1] > 10.2} {
    quietly set has_fixpt_radix 1
}

proc add_fixpt_wave {name width frac_width signed} {
    global has_fixpt_radix
    if {$frac_width > 0 && $has_fixpt_radix} {
        set type "[string index $signed 0]fix${width}_En${frac_width}"
        if {[lsearch [radix names] $type] < 0} {
            if {$signed == "signed"} {
                radix define $type -fixed -signed -fraction $frac_width
            } else {
                radix define $type -fixed -fraction $frac_width
            }
        }
        add wave -noupdate -format Literal -radix $type $name
    } else {
        add wave -noupdate -format Literal -radix $signed $name
    }
}

add wave -noupdate -divider {Input Ports}
add wave -noupdate -format Logic /flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u_atb/dut/clk
add wave -noupdate -format Logic /flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u_atb/dut/areset
add_fixpt_wave /flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u_atb/sim/in_0_stm 64 0 signed
add wave -noupdate -format Literal -radix binary -group in_0_stm -label {sign} {/flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u_atb/sim/in_0_stm(63 downto 63)} -label {exp} {/flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u_atb/sim/in_0_stm(62 downto 52)} -label {frac} {/flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u_atb/sim/in_0_stm(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u_atb/checkin_0/in_0_stm_real
add_fixpt_wave /flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u_atb/sim/in_1_stm 64 0 signed
add wave -noupdate -format Literal -radix binary -group in_1_stm -label {sign} {/flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u_atb/sim/in_1_stm(63 downto 63)} -label {exp} {/flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u_atb/sim/in_1_stm(62 downto 52)} -label {frac} {/flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u_atb/sim/in_1_stm(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u_atb/checkin_1/in_1_stm_real
add wave -noupdate -divider {Output Ports}
add_fixpt_wave /flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u_atb/dut/out_primWireOut 64 0 signed
add wave -noupdate -format Literal -radix binary -group out_primWireOut  -label {sign} {/flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u_atb/dut/out_primWireOut(63 downto 63)} -label {exp} {/flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u_atb/dut/out_primWireOut(62 downto 52)} -label {frac} {/flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u_atb/dut/out_primWireOut(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u_atb/checkout_primWireOut/out_primWireOut_real
add_fixpt_wave /flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u_atb/sim/out_primWireOut_stm 64 0 signed
add wave -noupdate -format Literal -radix binary -group out_primWireOut_stm -label {sign} {/flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u_atb/sim/out_primWireOut_stm(63 downto 63)} -label {exp} {/flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u_atb/sim/out_primWireOut_stm(62 downto 52)} -label {frac} {/flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u_atb/sim/out_primWireOut_stm(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u_atb/checkout_primWireOut/out_primWireOut_stm_real
TreeUpdate [SetDefaultTree]
