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
add wave -noupdate -format Logic /fft_example_DUT_atb/dut/clk
add wave -noupdate -format Logic /fft_example_DUT_atb/dut/areset
add wave -noupdate -format Logical /fft_example_DUT_atb/sim/v_in_s_stm
add_fixpt_wave /fft_example_DUT_atb/sim/channel_in_s_stm 8 0 unsigned
add_fixpt_wave /fft_example_DUT_atb/sim/data_in_0re_stm 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_in_0re_stm -label {sign} {/fft_example_DUT_atb/sim/data_in_0re_stm(63 downto 63)} -label {exp} {/fft_example_DUT_atb/sim/data_in_0re_stm(62 downto 52)} -label {frac} {/fft_example_DUT_atb/sim/data_in_0re_stm(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelIn_vunroll_cunroll_x/data_in_0re_stm_real
add_fixpt_wave /fft_example_DUT_atb/sim/data_in_0im_stm 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_in_0im_stm -label {sign} {/fft_example_DUT_atb/sim/data_in_0im_stm(63 downto 63)} -label {exp} {/fft_example_DUT_atb/sim/data_in_0im_stm(62 downto 52)} -label {frac} {/fft_example_DUT_atb/sim/data_in_0im_stm(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelIn_vunroll_cunroll_x/data_in_0im_stm_real
add_fixpt_wave /fft_example_DUT_atb/sim/data_in_1re_stm 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_in_1re_stm -label {sign} {/fft_example_DUT_atb/sim/data_in_1re_stm(63 downto 63)} -label {exp} {/fft_example_DUT_atb/sim/data_in_1re_stm(62 downto 52)} -label {frac} {/fft_example_DUT_atb/sim/data_in_1re_stm(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelIn_vunroll_cunroll_x/data_in_1re_stm_real
add_fixpt_wave /fft_example_DUT_atb/sim/data_in_1im_stm 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_in_1im_stm -label {sign} {/fft_example_DUT_atb/sim/data_in_1im_stm(63 downto 63)} -label {exp} {/fft_example_DUT_atb/sim/data_in_1im_stm(62 downto 52)} -label {frac} {/fft_example_DUT_atb/sim/data_in_1im_stm(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelIn_vunroll_cunroll_x/data_in_1im_stm_real
add_fixpt_wave /fft_example_DUT_atb/sim/data_in_2re_stm 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_in_2re_stm -label {sign} {/fft_example_DUT_atb/sim/data_in_2re_stm(63 downto 63)} -label {exp} {/fft_example_DUT_atb/sim/data_in_2re_stm(62 downto 52)} -label {frac} {/fft_example_DUT_atb/sim/data_in_2re_stm(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelIn_vunroll_cunroll_x/data_in_2re_stm_real
add_fixpt_wave /fft_example_DUT_atb/sim/data_in_2im_stm 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_in_2im_stm -label {sign} {/fft_example_DUT_atb/sim/data_in_2im_stm(63 downto 63)} -label {exp} {/fft_example_DUT_atb/sim/data_in_2im_stm(62 downto 52)} -label {frac} {/fft_example_DUT_atb/sim/data_in_2im_stm(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelIn_vunroll_cunroll_x/data_in_2im_stm_real
add_fixpt_wave /fft_example_DUT_atb/sim/data_in_3re_stm 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_in_3re_stm -label {sign} {/fft_example_DUT_atb/sim/data_in_3re_stm(63 downto 63)} -label {exp} {/fft_example_DUT_atb/sim/data_in_3re_stm(62 downto 52)} -label {frac} {/fft_example_DUT_atb/sim/data_in_3re_stm(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelIn_vunroll_cunroll_x/data_in_3re_stm_real
add_fixpt_wave /fft_example_DUT_atb/sim/data_in_3im_stm 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_in_3im_stm -label {sign} {/fft_example_DUT_atb/sim/data_in_3im_stm(63 downto 63)} -label {exp} {/fft_example_DUT_atb/sim/data_in_3im_stm(62 downto 52)} -label {frac} {/fft_example_DUT_atb/sim/data_in_3im_stm(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelIn_vunroll_cunroll_x/data_in_3im_stm_real
add wave -noupdate -divider {Output Ports}
add wave -noupdate -format Logical /fft_example_DUT_atb/dut/v_out_s
add wave -noupdate -format Logical /fft_example_DUT_atb/sim/v_out_s_stm
add_fixpt_wave /fft_example_DUT_atb/dut/data_out_0re 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_out_0re  -label {sign} {/fft_example_DUT_atb/dut/data_out_0re(63 downto 63)} -label {exp} {/fft_example_DUT_atb/dut/data_out_0re(62 downto 52)} -label {frac} {/fft_example_DUT_atb/dut/data_out_0re(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelOut_vunroll_cunroll_x/data_out_0re_real
add_fixpt_wave /fft_example_DUT_atb/sim/data_out_0re_stm 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_out_0re_stm -label {sign} {/fft_example_DUT_atb/sim/data_out_0re_stm(63 downto 63)} -label {exp} {/fft_example_DUT_atb/sim/data_out_0re_stm(62 downto 52)} -label {frac} {/fft_example_DUT_atb/sim/data_out_0re_stm(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelOut_vunroll_cunroll_x/data_out_0re_stm_real
add_fixpt_wave /fft_example_DUT_atb/dut/data_out_0im 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_out_0im  -label {sign} {/fft_example_DUT_atb/dut/data_out_0im(63 downto 63)} -label {exp} {/fft_example_DUT_atb/dut/data_out_0im(62 downto 52)} -label {frac} {/fft_example_DUT_atb/dut/data_out_0im(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelOut_vunroll_cunroll_x/data_out_0im_real
add_fixpt_wave /fft_example_DUT_atb/sim/data_out_0im_stm 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_out_0im_stm -label {sign} {/fft_example_DUT_atb/sim/data_out_0im_stm(63 downto 63)} -label {exp} {/fft_example_DUT_atb/sim/data_out_0im_stm(62 downto 52)} -label {frac} {/fft_example_DUT_atb/sim/data_out_0im_stm(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelOut_vunroll_cunroll_x/data_out_0im_stm_real
add_fixpt_wave /fft_example_DUT_atb/dut/data_out_1re 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_out_1re  -label {sign} {/fft_example_DUT_atb/dut/data_out_1re(63 downto 63)} -label {exp} {/fft_example_DUT_atb/dut/data_out_1re(62 downto 52)} -label {frac} {/fft_example_DUT_atb/dut/data_out_1re(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelOut_vunroll_cunroll_x/data_out_1re_real
add_fixpt_wave /fft_example_DUT_atb/sim/data_out_1re_stm 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_out_1re_stm -label {sign} {/fft_example_DUT_atb/sim/data_out_1re_stm(63 downto 63)} -label {exp} {/fft_example_DUT_atb/sim/data_out_1re_stm(62 downto 52)} -label {frac} {/fft_example_DUT_atb/sim/data_out_1re_stm(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelOut_vunroll_cunroll_x/data_out_1re_stm_real
add_fixpt_wave /fft_example_DUT_atb/dut/data_out_1im 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_out_1im  -label {sign} {/fft_example_DUT_atb/dut/data_out_1im(63 downto 63)} -label {exp} {/fft_example_DUT_atb/dut/data_out_1im(62 downto 52)} -label {frac} {/fft_example_DUT_atb/dut/data_out_1im(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelOut_vunroll_cunroll_x/data_out_1im_real
add_fixpt_wave /fft_example_DUT_atb/sim/data_out_1im_stm 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_out_1im_stm -label {sign} {/fft_example_DUT_atb/sim/data_out_1im_stm(63 downto 63)} -label {exp} {/fft_example_DUT_atb/sim/data_out_1im_stm(62 downto 52)} -label {frac} {/fft_example_DUT_atb/sim/data_out_1im_stm(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelOut_vunroll_cunroll_x/data_out_1im_stm_real
add_fixpt_wave /fft_example_DUT_atb/dut/data_out_2re 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_out_2re  -label {sign} {/fft_example_DUT_atb/dut/data_out_2re(63 downto 63)} -label {exp} {/fft_example_DUT_atb/dut/data_out_2re(62 downto 52)} -label {frac} {/fft_example_DUT_atb/dut/data_out_2re(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelOut_vunroll_cunroll_x/data_out_2re_real
add_fixpt_wave /fft_example_DUT_atb/sim/data_out_2re_stm 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_out_2re_stm -label {sign} {/fft_example_DUT_atb/sim/data_out_2re_stm(63 downto 63)} -label {exp} {/fft_example_DUT_atb/sim/data_out_2re_stm(62 downto 52)} -label {frac} {/fft_example_DUT_atb/sim/data_out_2re_stm(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelOut_vunroll_cunroll_x/data_out_2re_stm_real
add_fixpt_wave /fft_example_DUT_atb/dut/data_out_2im 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_out_2im  -label {sign} {/fft_example_DUT_atb/dut/data_out_2im(63 downto 63)} -label {exp} {/fft_example_DUT_atb/dut/data_out_2im(62 downto 52)} -label {frac} {/fft_example_DUT_atb/dut/data_out_2im(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelOut_vunroll_cunroll_x/data_out_2im_real
add_fixpt_wave /fft_example_DUT_atb/sim/data_out_2im_stm 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_out_2im_stm -label {sign} {/fft_example_DUT_atb/sim/data_out_2im_stm(63 downto 63)} -label {exp} {/fft_example_DUT_atb/sim/data_out_2im_stm(62 downto 52)} -label {frac} {/fft_example_DUT_atb/sim/data_out_2im_stm(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelOut_vunroll_cunroll_x/data_out_2im_stm_real
add_fixpt_wave /fft_example_DUT_atb/dut/data_out_3re 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_out_3re  -label {sign} {/fft_example_DUT_atb/dut/data_out_3re(63 downto 63)} -label {exp} {/fft_example_DUT_atb/dut/data_out_3re(62 downto 52)} -label {frac} {/fft_example_DUT_atb/dut/data_out_3re(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelOut_vunroll_cunroll_x/data_out_3re_real
add_fixpt_wave /fft_example_DUT_atb/sim/data_out_3re_stm 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_out_3re_stm -label {sign} {/fft_example_DUT_atb/sim/data_out_3re_stm(63 downto 63)} -label {exp} {/fft_example_DUT_atb/sim/data_out_3re_stm(62 downto 52)} -label {frac} {/fft_example_DUT_atb/sim/data_out_3re_stm(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelOut_vunroll_cunroll_x/data_out_3re_stm_real
add_fixpt_wave /fft_example_DUT_atb/dut/data_out_3im 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_out_3im  -label {sign} {/fft_example_DUT_atb/dut/data_out_3im(63 downto 63)} -label {exp} {/fft_example_DUT_atb/dut/data_out_3im(62 downto 52)} -label {frac} {/fft_example_DUT_atb/dut/data_out_3im(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelOut_vunroll_cunroll_x/data_out_3im_real
add_fixpt_wave /fft_example_DUT_atb/sim/data_out_3im_stm 64 0 signed
add wave -noupdate -format Literal -radix binary -group data_out_3im_stm -label {sign} {/fft_example_DUT_atb/sim/data_out_3im_stm(63 downto 63)} -label {exp} {/fft_example_DUT_atb/sim/data_out_3im_stm(62 downto 52)} -label {frac} {/fft_example_DUT_atb/sim/data_out_3im_stm(51 downto 0)} 
add wave -noupdate -format Literal -radix decimal /fft_example_DUT_atb/checkChannelOut_vunroll_cunroll_x/data_out_3im_stm_real
TreeUpdate [SetDefaultTree]
