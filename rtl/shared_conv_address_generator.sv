// shared_conv_address_generator.sv
// Layer-selectable address/padding helper for the shared convolution engine.
//
// Trained sizes (obsolete 8/16 prompt sketches are not used):
//   Conv1 (layer_is_conv2=0): 3->16 on 32x32, 27 weights/oc
//   Conv2 (layer_is_conv2=1): 16->32 on 16x16, 144 weights/oc
//
// Reuses the verified Conv1/Conv2 address generators bit-exactly.

`timescale 1ns / 1ps

module shared_conv_address_generator (
    input  logic        layer_is_conv2,

    input  logic [4:0]  output_channel,   // 0..15 Conv1, 0..31 Conv2
    input  logic [4:0]  output_row,       // 0..31 Conv1, 0..15 Conv2
    input  logic [4:0]  output_column,
    input  logic [3:0]  input_channel,    // 0..2 Conv1, 0..15 Conv2
    input  logic [1:0]  kernel_row,
    input  logic [1:0]  kernel_column,

    output logic signed [6:0] input_row,
    output logic signed [6:0] input_column,
    output logic              padding,
    output logic [11:0]       activation_address,
    output logic [12:0]       weight_address,
    output logic [4:0]        bias_address,
    output logic [13:0]       output_address
);

    logic signed [6:0] c1_in_row, c1_in_col;
    logic              c1_pad;
    logic [11:0]       c1_act;
    logic [8:0]        c1_wgt;
    logic [3:0]        c1_bias;

    logic signed [5:0] c2_in_row, c2_in_col;
    logic              c2_pad;
    logic [11:0]       c2_act;
    logic [12:0]       c2_wgt;
    logic [4:0]        c2_bias;
    logic [12:0]       c2_out;

    conv_address_generator u_c1 (
        .output_channel     (output_channel[3:0]),
        .output_row         (output_row),
        .output_column      (output_column),
        .input_channel      (input_channel[1:0]),
        .kernel_row         (kernel_row),
        .kernel_column      (kernel_column),
        .input_row          (c1_in_row),
        .input_column       (c1_in_col),
        .padding            (c1_pad),
        .activation_address (c1_act),
        .weight_address     (c1_wgt),
        .bias_address       (c1_bias)
    );

    conv2_address_generator u_c2 (
        .output_channel       (output_channel),
        .output_row           (output_row[3:0]),
        .output_column        (output_column[3:0]),
        .input_channel        (input_channel),
        .kernel_row           (kernel_row),
        .kernel_column        (kernel_column),
        .input_row            (c2_in_row),
        .input_column         (c2_in_col),
        .padding              (c2_pad),
        .pool1_read_address   (c2_act),
        .conv2_weight_address (c2_wgt),
        .conv2_bias_address   (c2_bias),
        .conv2_output_address (c2_out)
    );

    // Conv1 output address: oc*1024 + row*32 + col
    logic [13:0] c1_out;
    assign c1_out = ({output_channel[3:0], 10'd0})
                  + ({5'd0, output_row, 5'd0})
                  + {9'd0, output_column};

    always_comb begin
        if (!layer_is_conv2) begin
            input_row           = c1_in_row;
            input_column        = c1_in_col;
            padding             = c1_pad;
            activation_address  = c1_act;
            weight_address      = {4'd0, c1_wgt};
            bias_address        = {1'b0, c1_bias};
            output_address      = c1_out;
        end else begin
            input_row           = {c2_in_row[5], c2_in_row};
            input_column        = {c2_in_col[5], c2_in_col};
            padding             = c2_pad;
            activation_address  = c2_act;
            weight_address      = c2_wgt;
            bias_address        = c2_bias;
            output_address      = {1'b0, c2_out};
        end
    end

endmodule
