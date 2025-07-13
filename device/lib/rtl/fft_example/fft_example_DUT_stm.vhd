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
use std.TextIO.all;
USE work.fft_example_DUT_safe_path.all;

entity fft_example_DUT_stm is
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
end fft_example_DUT_stm;

architecture normal of fft_example_DUT_stm is

    signal clk_stm_sig : std_logic := '0';
    signal clk_stm_sig_stop : std_logic := '0';
    signal areset_stm_sig : std_logic := '1';
    signal clk_ChannelIn_vunroll_cunroll_x_stm_sig_stop : std_logic := '0';
    signal clk_ChannelOut_vunroll_cunroll_x_stm_sig_stop : std_logic := '0';

    function str_to_stdvec(inp: string) return std_logic_vector is
        variable temp: std_logic_vector(inp'range) := (others => 'X');
    begin
        for i in inp'range loop
            IF ((inp(i) = '1')) THEN
                temp(i) := '1';
            elsif (inp(i) = '0') then
                temp(i) := '0';
            END IF;
            end loop;
            return temp;
        end function str_to_stdvec;
        

    begin

    clk <= clk_stm_sig;
    clk_process: process 
    begin
        wait for 200 ps;
        clk_stm_sig <= not clk_stm_sig;
        wait for 800 ps;
        if (clk_stm_sig_stop = '1') then
            assert (false)
            report "Arrived at end of stimulus data on clk clk" severity NOTE;
            wait;
        end if;
        wait for 200 ps;
        clk_stm_sig <= not clk_stm_sig;
        wait for 800 ps;
        if (clk_stm_sig_stop = '1') then
            assert (false)
            report "Arrived at end of stimulus data on clk clk" severity NOTE;
            wait;
        end if;
    end process;

    areset <= areset_stm_sig;
    areset_process: process begin
        areset_stm_sig <= '1';
        wait for 1500 ps;
        wait for 1023*2000 ps; -- additional reset delay
        areset_stm_sig <= '0';
        wait;
    end process;


        -- Generating stimulus for ChannelIn_vunroll_cunroll_x
        ChannelIn_vunroll_cunroll_x_stm_init_p: process

            variable L : line;
            variable dummy_int : Integer;
            file data_file_ChannelIn_vunroll_cunroll_x : text open read_mode is safe_path("fft_example/fft_example_DUT_ChannelIn_vunroll_cunroll_x.stm");
            variable v_in_s_int_0 : Integer;
            variable v_in_s_temp : std_logic_vector(0 downto 0);
            variable channel_in_s_int_0 : Integer;
            variable channel_in_s_temp : std_logic_vector(7 downto 0);
            variable data_in_0re_int_0 : Integer;
            variable data_in_0re_int_1 : Integer;
            variable data_in_0re_temp : std_logic_vector(63 downto 0);
            variable data_in_0im_int_0 : Integer;
            variable data_in_0im_int_1 : Integer;
            variable data_in_0im_temp : std_logic_vector(63 downto 0);
            variable data_in_1re_int_0 : Integer;
            variable data_in_1re_int_1 : Integer;
            variable data_in_1re_temp : std_logic_vector(63 downto 0);
            variable data_in_1im_int_0 : Integer;
            variable data_in_1im_int_1 : Integer;
            variable data_in_1im_temp : std_logic_vector(63 downto 0);
            variable data_in_2re_int_0 : Integer;
            variable data_in_2re_int_1 : Integer;
            variable data_in_2re_temp : std_logic_vector(63 downto 0);
            variable data_in_2im_int_0 : Integer;
            variable data_in_2im_int_1 : Integer;
            variable data_in_2im_temp : std_logic_vector(63 downto 0);
            variable data_in_3re_int_0 : Integer;
            variable data_in_3re_int_1 : Integer;
            variable data_in_3re_temp : std_logic_vector(63 downto 0);
            variable data_in_3im_int_0 : Integer;
            variable data_in_3im_int_1 : Integer;
            variable data_in_3im_temp : std_logic_vector(63 downto 0);

        begin
            -- initialize all outputs to 0
            v_in_s_stm <= (others => '0');
            channel_in_s_stm <= (others => '0');
            data_in_0re_stm <= (others => '0');
            data_in_0im_stm <= (others => '0');
            data_in_1re_stm <= (others => '0');
            data_in_1im_stm <= (others => '0');
            data_in_2re_stm <= (others => '0');
            data_in_2im_stm <= (others => '0');
            data_in_3re_stm <= (others => '0');
            data_in_3im_stm <= (others => '0');

            wait for 201 ps; -- wait delay
            
            for tick in 1 to 1023 loop
            
                wait for 2000 ps; -- additional reset delay
                
                v_in_s_stm <= (others => '0');
                channel_in_s_stm <= (others => '0');
                data_in_0re_stm <= (others => '0');
                data_in_0im_stm <= (others => '0');
                data_in_1re_stm <= (others => '0');
                data_in_1im_stm <= (others => '0');
                data_in_2re_stm <= (others => '0');
                data_in_2im_stm <= (others => '0');
                data_in_3re_stm <= (others => '0');
                data_in_3im_stm <= (others => '0');
            end loop;
            while true loop
            
                IF (endfile(data_file_ChannelIn_vunroll_cunroll_x)) THEN
                    clk_ChannelIn_vunroll_cunroll_x_stm_sig_stop <= '1';
                    wait;
                ELSE
                    readline(data_file_ChannelIn_vunroll_cunroll_x, L);
                    
                    read(L, v_in_s_int_0);
                    v_in_s_temp(0 downto 0) := std_logic_vector(to_unsigned(v_in_s_int_0, 1));
                    v_in_s_stm <= v_in_s_temp;
                    read(L, channel_in_s_int_0);
                    channel_in_s_temp(7 downto 0) := std_logic_vector(to_unsigned(channel_in_s_int_0, 8));
                    channel_in_s_stm <= channel_in_s_temp;
                    read(L, data_in_0re_int_0);
                    data_in_0re_temp(31 downto 0) := std_logic_vector(to_signed(data_in_0re_int_0, 32));
                    read(L, data_in_0re_int_1);
                    data_in_0re_temp(63 downto 32) := std_logic_vector(to_signed(data_in_0re_int_1, 32));
                    data_in_0re_stm <= data_in_0re_temp;
                    read(L, data_in_0im_int_0);
                    data_in_0im_temp(31 downto 0) := std_logic_vector(to_signed(data_in_0im_int_0, 32));
                    read(L, data_in_0im_int_1);
                    data_in_0im_temp(63 downto 32) := std_logic_vector(to_signed(data_in_0im_int_1, 32));
                    data_in_0im_stm <= data_in_0im_temp;
                    read(L, data_in_1re_int_0);
                    data_in_1re_temp(31 downto 0) := std_logic_vector(to_signed(data_in_1re_int_0, 32));
                    read(L, data_in_1re_int_1);
                    data_in_1re_temp(63 downto 32) := std_logic_vector(to_signed(data_in_1re_int_1, 32));
                    data_in_1re_stm <= data_in_1re_temp;
                    read(L, data_in_1im_int_0);
                    data_in_1im_temp(31 downto 0) := std_logic_vector(to_signed(data_in_1im_int_0, 32));
                    read(L, data_in_1im_int_1);
                    data_in_1im_temp(63 downto 32) := std_logic_vector(to_signed(data_in_1im_int_1, 32));
                    data_in_1im_stm <= data_in_1im_temp;
                    read(L, data_in_2re_int_0);
                    data_in_2re_temp(31 downto 0) := std_logic_vector(to_signed(data_in_2re_int_0, 32));
                    read(L, data_in_2re_int_1);
                    data_in_2re_temp(63 downto 32) := std_logic_vector(to_signed(data_in_2re_int_1, 32));
                    data_in_2re_stm <= data_in_2re_temp;
                    read(L, data_in_2im_int_0);
                    data_in_2im_temp(31 downto 0) := std_logic_vector(to_signed(data_in_2im_int_0, 32));
                    read(L, data_in_2im_int_1);
                    data_in_2im_temp(63 downto 32) := std_logic_vector(to_signed(data_in_2im_int_1, 32));
                    data_in_2im_stm <= data_in_2im_temp;
                    read(L, data_in_3re_int_0);
                    data_in_3re_temp(31 downto 0) := std_logic_vector(to_signed(data_in_3re_int_0, 32));
                    read(L, data_in_3re_int_1);
                    data_in_3re_temp(63 downto 32) := std_logic_vector(to_signed(data_in_3re_int_1, 32));
                    data_in_3re_stm <= data_in_3re_temp;
                    read(L, data_in_3im_int_0);
                    data_in_3im_temp(31 downto 0) := std_logic_vector(to_signed(data_in_3im_int_0, 32));
                    read(L, data_in_3im_int_1);
                    data_in_3im_temp(63 downto 32) := std_logic_vector(to_signed(data_in_3im_int_1, 32));
                    data_in_3im_stm <= data_in_3im_temp;

                    deallocate(L);
                END IF;
                -- -- wait for rising edge to pass (assert signals just after rising edge)
                wait until clk_stm_sig'EVENT and clk_stm_sig = '1';
                wait for 1 ps; -- wait delay
                
                end loop;
            wait;
        END PROCESS;

        -- Generating stimulus for ChannelOut_vunroll_cunroll_x
        ChannelOut_vunroll_cunroll_x_stm_init_p: process

            variable L : line;
            variable dummy_int : Integer;
            file data_file_ChannelOut_vunroll_cunroll_x : text open read_mode is safe_path("fft_example/fft_example_DUT_ChannelOut_vunroll_cunroll_x.stm");
            variable v_out_s_int_0 : Integer;
            variable v_out_s_temp : std_logic_vector(0 downto 0);
            variable data_out_0re_int_0 : Integer;
            variable data_out_0re_int_1 : Integer;
            variable data_out_0re_temp : std_logic_vector(63 downto 0);
            variable data_out_0im_int_0 : Integer;
            variable data_out_0im_int_1 : Integer;
            variable data_out_0im_temp : std_logic_vector(63 downto 0);
            variable data_out_1re_int_0 : Integer;
            variable data_out_1re_int_1 : Integer;
            variable data_out_1re_temp : std_logic_vector(63 downto 0);
            variable data_out_1im_int_0 : Integer;
            variable data_out_1im_int_1 : Integer;
            variable data_out_1im_temp : std_logic_vector(63 downto 0);
            variable data_out_2re_int_0 : Integer;
            variable data_out_2re_int_1 : Integer;
            variable data_out_2re_temp : std_logic_vector(63 downto 0);
            variable data_out_2im_int_0 : Integer;
            variable data_out_2im_int_1 : Integer;
            variable data_out_2im_temp : std_logic_vector(63 downto 0);
            variable data_out_3re_int_0 : Integer;
            variable data_out_3re_int_1 : Integer;
            variable data_out_3re_temp : std_logic_vector(63 downto 0);
            variable data_out_3im_int_0 : Integer;
            variable data_out_3im_int_1 : Integer;
            variable data_out_3im_temp : std_logic_vector(63 downto 0);

        begin
            -- initialize all outputs to 0
            v_out_s_stm <= (others => '0');
            data_out_0re_stm <= (others => '0');
            data_out_0im_stm <= (others => '0');
            data_out_1re_stm <= (others => '0');
            data_out_1im_stm <= (others => '0');
            data_out_2re_stm <= (others => '0');
            data_out_2im_stm <= (others => '0');
            data_out_3re_stm <= (others => '0');
            data_out_3im_stm <= (others => '0');

            wait for 201 ps; -- wait delay
            
            wait for 1023*2000 ps; -- additional reset delay
            
            while true loop
            
                IF (endfile(data_file_ChannelOut_vunroll_cunroll_x)) THEN
                    clk_ChannelOut_vunroll_cunroll_x_stm_sig_stop <= '1';
                    wait;
                ELSE
                    readline(data_file_ChannelOut_vunroll_cunroll_x, L);
                    
                    read(L, v_out_s_int_0);
                    v_out_s_temp(0 downto 0) := std_logic_vector(to_unsigned(v_out_s_int_0, 1));
                    v_out_s_stm <= v_out_s_temp;
                    read(L, dummy_int);
                    read(L, data_out_0re_int_0);
                    data_out_0re_temp(31 downto 0) := std_logic_vector(to_signed(data_out_0re_int_0, 32));
                    read(L, data_out_0re_int_1);
                    data_out_0re_temp(63 downto 32) := std_logic_vector(to_signed(data_out_0re_int_1, 32));
                    data_out_0re_stm <= data_out_0re_temp;
                    read(L, data_out_0im_int_0);
                    data_out_0im_temp(31 downto 0) := std_logic_vector(to_signed(data_out_0im_int_0, 32));
                    read(L, data_out_0im_int_1);
                    data_out_0im_temp(63 downto 32) := std_logic_vector(to_signed(data_out_0im_int_1, 32));
                    data_out_0im_stm <= data_out_0im_temp;
                    read(L, data_out_1re_int_0);
                    data_out_1re_temp(31 downto 0) := std_logic_vector(to_signed(data_out_1re_int_0, 32));
                    read(L, data_out_1re_int_1);
                    data_out_1re_temp(63 downto 32) := std_logic_vector(to_signed(data_out_1re_int_1, 32));
                    data_out_1re_stm <= data_out_1re_temp;
                    read(L, data_out_1im_int_0);
                    data_out_1im_temp(31 downto 0) := std_logic_vector(to_signed(data_out_1im_int_0, 32));
                    read(L, data_out_1im_int_1);
                    data_out_1im_temp(63 downto 32) := std_logic_vector(to_signed(data_out_1im_int_1, 32));
                    data_out_1im_stm <= data_out_1im_temp;
                    read(L, data_out_2re_int_0);
                    data_out_2re_temp(31 downto 0) := std_logic_vector(to_signed(data_out_2re_int_0, 32));
                    read(L, data_out_2re_int_1);
                    data_out_2re_temp(63 downto 32) := std_logic_vector(to_signed(data_out_2re_int_1, 32));
                    data_out_2re_stm <= data_out_2re_temp;
                    read(L, data_out_2im_int_0);
                    data_out_2im_temp(31 downto 0) := std_logic_vector(to_signed(data_out_2im_int_0, 32));
                    read(L, data_out_2im_int_1);
                    data_out_2im_temp(63 downto 32) := std_logic_vector(to_signed(data_out_2im_int_1, 32));
                    data_out_2im_stm <= data_out_2im_temp;
                    read(L, data_out_3re_int_0);
                    data_out_3re_temp(31 downto 0) := std_logic_vector(to_signed(data_out_3re_int_0, 32));
                    read(L, data_out_3re_int_1);
                    data_out_3re_temp(63 downto 32) := std_logic_vector(to_signed(data_out_3re_int_1, 32));
                    data_out_3re_stm <= data_out_3re_temp;
                    read(L, data_out_3im_int_0);
                    data_out_3im_temp(31 downto 0) := std_logic_vector(to_signed(data_out_3im_int_0, 32));
                    read(L, data_out_3im_int_1);
                    data_out_3im_temp(63 downto 32) := std_logic_vector(to_signed(data_out_3im_int_1, 32));
                    data_out_3im_stm <= data_out_3im_temp;

                    deallocate(L);
                END IF;
                -- -- wait for rising edge to pass (assert signals just after rising edge)
                wait until clk_stm_sig'EVENT and clk_stm_sig = '1';
                wait for 1 ps; -- wait delay
                
                end loop;
            wait;
        END PROCESS;

    clk_stm_sig_stop <= clk_ChannelIn_vunroll_cunroll_x_stm_sig_stop OR clk_ChannelOut_vunroll_cunroll_x_stm_sig_stop OR '0';


    END normal;
