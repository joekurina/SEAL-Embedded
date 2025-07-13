# pass in -Gquit_at_end=true to make vsim call exit at the end. Useful for running standalone.
quietly set quit_at_end 0
if {[lsearch $argv -Gquit_at_end=true] != -1} {
    quietly set quit_at_end 1
}

if {$argc > 0} {
    quietly set base_dir $1
} else {
    quietly set base_dir "././rtl"
    echo The current directory is: [pwd]
}
quietly set base_dir [file normalize $base_dir]
echo Creating the project under $base_dir

do $base_dir/compile_modelsim_libraries.do
onerror {resume}

if { [string compare [project env] ""] != 0 } {
    quit -sim
    project close
}

if {! [file exists $base_dir/fft_example_DUT]} {
    file delete -force $base_dir/fft_example_DUT
}

project new $base_dir fft_example_DUT
if {! [file exists $base_dir/work/_info]} {
    file delete -force $base_dir/work
    vlib work
}
quietly vmap work $base_dir/work

do "$base_dir/fft_example/fft_example_DUT_fpc.do"


quietly set vcomfailed 0
onerror {
    quietly set vcomfailed 1
    resume
}

project addfile $base_dir/fft_example/fft_example_DUT_safe_path_msim.vhd vhdl
project addfile $base_dir/fft_example/fft_example_DUT.vhd vhdl
project addfile $base_dir/fft_example/flt_fft_example_DUT_addSubFusedBlock_typeSFloatIEEE_52_11_4_correctRounding_38560000x0ao30cd06cj6ok0dpzc.vhd vhdl
project addfile $base_dir/fft_example/flt_fft_example_DUT_castBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_castMode0000226123642i229742iyc5.vhd vhdl
project addfile $base_dir/fft_example/flt_fft_example_DUT_addBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa0000j6of0cd16ok0cp06hj0u.vhd vhdl
project addfile $base_dir/fft_example/flt_fft_example_DUT_multBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFlo0000of0cdj6of0cd16ol0qcz.vhd vhdl
project addfile $base_dir/fft_example/flt_fft_example_DUT_subBlock_typeSFloatIEEE_52_11_typeSFloatIEEE_52_11_typeSFloa00000cdj6of0cd16ok0ct30u.vhd vhdl
puts {Note: Process variables may be optimized out of top-level testbench. Re-compile with the following command to disable optimizations:}
puts {vcom -quiet -O0 $base_dir/fft_example/fft_example_DUT_atb.vhd}
project addfile $base_dir/fft_example/fft_example_DUT_atb.vhd vhdl
project addfile $base_dir/fft_example/fft_example_DUT_stm.vhd vhdl
project calculateorder

onerror {resume}

proc report_mismatch {signal cycle} {
    puts "Mismatch in ${signal} at system clock cycle ${cycle}"
    set modelsimvalue [examine ${signal}_dut];
    set stmvalue [examine ${signal}_stm];
    puts "\t${signal} (ModelSim):\t${modelsimvalue}"
    puts "\t${signal} (Simulink):\t${stmvalue}"
}

if {$vcomfailed == 0} {
    onbreak {
        quietly set my_tb [string trim [tb]];
        quietly set regOK [regexp {(.*) ([0-9]+) ([\[address]*) ([.]*)} $my_tb \ match atbfile linenum ignore_this];
        if {$regOK == 1} {
            quietly set simtime [expr $now - 200];
            quietly set cyclenum [expr int($simtime / 2000.000000)];
            if { [catch {exa mismatch_v_out_s} mismatch] == 0 && $mismatch } {
                report_mismatch v_out_s $cyclenum
            }
            if { [catch {exa mismatch_data_out_0re} mismatch] == 0 && $mismatch } {
                report_mismatch data_out_0re $cyclenum
            }
            if { [catch {exa mismatch_data_out_0im} mismatch] == 0 && $mismatch } {
                report_mismatch data_out_0im $cyclenum
            }
            if { [catch {exa mismatch_data_out_1re} mismatch] == 0 && $mismatch } {
                report_mismatch data_out_1re $cyclenum
            }
            if { [catch {exa mismatch_data_out_1im} mismatch] == 0 && $mismatch } {
                report_mismatch data_out_1im $cyclenum
            }
            if { [catch {exa mismatch_data_out_2re} mismatch] == 0 && $mismatch } {
                report_mismatch data_out_2re $cyclenum
            }
            if { [catch {exa mismatch_data_out_2im} mismatch] == 0 && $mismatch } {
                report_mismatch data_out_2im $cyclenum
            }
            if { [catch {exa mismatch_data_out_3re} mismatch] == 0 && $mismatch } {
                report_mismatch data_out_3re $cyclenum
            }
            if { [catch {exa mismatch_data_out_3im} mismatch] == 0 && $mismatch } {
                report_mismatch data_out_3im $cyclenum
            }
        } else {
            puts "Signal mismatch detected at $my_tb";
        }
        if {$quit_at_end == 1} {
            quit -code 1;
        }
    }
    eval vsim -quiet -suppress 14408 -error 3473 -msgmode both -voptargs="+acc" -t ps fft_example_DUT_atb $ll
    do $base_dir/fft_example/fft_example_DUT_atb.wav.do
# Disable some warnings that occur at the very start of simulation
    quietly set StdArithNoWarnings 1
    run 0ns
    quietly set StdArithNoWarnings 0
    run -all
} else {
    echo At least one module failed to compile, not starting simulation
}

if {$quit_at_end == 1} {
    exit
}
