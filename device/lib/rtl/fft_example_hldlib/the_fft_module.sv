typedef struct packed {
    logic[63:0] ChannelIn_port_data_in_3im;
    logic[63:0] ChannelIn_port_data_in_3re;
    logic[63:0] ChannelIn_port_data_in_2im;
    logic[63:0] ChannelIn_port_data_in_2re;
    logic[63:0] ChannelIn_port_data_in_1im;
    logic[63:0] ChannelIn_port_data_in_1re;
    logic[63:0] ChannelIn_port_data_in_0im;
    logic[63:0] ChannelIn_port_data_in_0re;
    logic[7:0] ChannelIn_port_channel_in_s;
    logic[7:0] ChannelIn_port_v_in_s;
} input_t;

typedef struct packed {
    logic[63:0] ChannelOut_port_data_out_3im;
    logic[63:0] ChannelOut_port_data_out_3re;
    logic[63:0] ChannelOut_port_data_out_2im;
    logic[63:0] ChannelOut_port_data_out_2re;
    logic[63:0] ChannelOut_port_data_out_1im;
    logic[63:0] ChannelOut_port_data_out_1re;
    logic[63:0] ChannelOut_port_data_out_0im;
    logic[63:0] ChannelOut_port_data_out_0re;
    logic[7:0] ChannelOut_port_;
    logic[7:0] ChannelOut_port_v_out_s;
} output_t;

module the_fft_module
(
    input  clock,
    input  resetn,

    input  ivalid,
    output ovalid,

    input  iready,
    output oready,

    input[$bits(input_t) - 1:0] idata,
    output[$bits(output_t) - 1:0] odata
);

input_t dataIn;
output_t dataOut;

assign oready = 1'b1;
assign ovalid = dataOut.ChannelOut_port_v_out_s;

assign dataIn = idata;
assign odata = dataOut;

logic valid_in;
assign valid_in = ivalid & dataIn.ChannelIn_port_v_in_s;

logic areset;
assign areset = ~resetn;

fft_example_DUT
fft_example_DUT_inst (
    .v_in_s (valid_in),
    .channel_in_s (dataIn.ChannelIn_port_channel_in_s),
    .data_in_0re (dataIn.ChannelIn_port_data_in_0re),
    .data_in_0im (dataIn.ChannelIn_port_data_in_0im),
    .data_in_1re (dataIn.ChannelIn_port_data_in_1re),
    .data_in_1im (dataIn.ChannelIn_port_data_in_1im),
    .data_in_2re (dataIn.ChannelIn_port_data_in_2re),
    .data_in_2im (dataIn.ChannelIn_port_data_in_2im),
    .data_in_3re (dataIn.ChannelIn_port_data_in_3re),
    .data_in_3im (dataIn.ChannelIn_port_data_in_3im),
    .v_out_s (dataOut.ChannelOut_port_v_out_s),
    .data_out_0re (dataOut.ChannelOut_port_data_out_0re),
    .data_out_0im (dataOut.ChannelOut_port_data_out_0im),
    .data_out_1re (dataOut.ChannelOut_port_data_out_1re),
    .data_out_1im (dataOut.ChannelOut_port_data_out_1im),
    .data_out_2re (dataOut.ChannelOut_port_data_out_2re),
    .data_out_2im (dataOut.ChannelOut_port_data_out_2im),
    .data_out_3re (dataOut.ChannelOut_port_data_out_3re),
    .data_out_3im (dataOut.ChannelOut_port_data_out_3im),
    .clk (clock),
    .areset (areset)
);
endmodule;
