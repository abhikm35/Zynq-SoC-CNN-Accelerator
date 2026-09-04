// conv_address_generator.sv
// Combinational Conv1 address / padding helper for one kernel tap.
//
// Geometry (trained model): input 3x32x32, kernel 3x3, stride 1, pad 1.
// Output channels: 16 (authoritative trained model; not the obsolete 8-ch sketch).
//
// Address equations (match software.hardware.memory_layout):
//   activation_address = input_channel * 1024 + input_row * 32 + input_column
//   weight_address     = output_channel * 27 + input_channel * 9
//                        + kernel_row * 3 + kernel_column
//   bias_address       = output_channel
//
// Padded taps (input_row/col outside [0,31]):
//   padding = 1, activation_address is driven to 0 but must not be used for a
//   memory read by the caller (activation value forced to 0).

`timescale 1ns / 1ps

module conv_address_generator (
    input  logic        [3:0]  output_channel,   // 0 .. 15
    input  logic        [4:0]  output_row,       // 0 .. 31
    input  logic        [4:0]  output_column,    // 0 .. 31
    input  logic        [1:0]  input_channel,    // 0 .. 2
    input  logic        [1:0]  kernel_row,       // 0 .. 2
    input  logic        [1:0]  kernel_column,    // 0 .. 2

    output logic signed [6:0]  input_row,        // may be -1 .. 32
    output logic signed [6:0]  input_column,
    output logic               padding,
    output logic        [11:0] activation_address, // 0 .. 3071 when valid
    output logic        [8:0]  weight_address,     // 0 .. 431
    output logic        [3:0]  bias_address        // 0 .. 15
);

    // Signed temporaries: out + k - pad1
    logic signed [6:0] row_s;
    logic signed [6:0] col_s;
    logic              in_bounds;
    logic signed [6:0] out_row_s;
    logic signed [6:0] out_col_s;
    logic signed [6:0] kr_s;
    logic signed [6:0] kc_s;

    always_comb begin
        out_row_s = {2'b0, output_row};
        out_col_s = {2'b0, output_column};
        kr_s      = {5'b0, kernel_row};
        kc_s      = {5'b0, kernel_column};
        row_s = out_row_s + kr_s - 7'sd1;
        col_s = out_col_s + kc_s - 7'sd1;

        input_row    = row_s;
        input_column = col_s;

        in_bounds = (row_s >= 7'sd0) && (row_s < 7'sd32)
                 && (col_s >= 7'sd0) && (col_s < 7'sd32);
        padding = ~in_bounds;

        // Valid activation address only when in bounds. When padded, hold 0;
        // the engine must gate the activation ROM read_enable with ~padding.
        if (in_bounds) begin
            // activation_address = ic*1024 + row*32 + col
            activation_address = ({10'd0, input_channel} << 10)
                               + ({7'd0, row_s[4:0]} << 5)
                               + {7'd0, col_s[4:0]};
        end else begin
            activation_address = 12'd0;
        end

        // weight_address = oc*27 + ic*9 + kr*3 + kc
        weight_address = ({5'd0, output_channel} * 9'd27)
                       + ({7'd0, input_channel} * 9'd9)
                       + ({7'd0, kernel_row} * 9'd3)
                       + {7'd0, kernel_column};

        bias_address = output_channel;
    end

endmodule
