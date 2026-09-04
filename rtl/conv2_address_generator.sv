// conv2_address_generator.sv
// Combinational address helper for Conv2 (trained: 16->32 on 16x16 Pool1).
//
// input_row    = output_row + kernel_row - 1
// input_column = output_column + kernel_column - 1
// padding when input coords outside [0, 16)
//
// pool1_read_address    = ic * 256 + row * 16 + col
// conv2_weight_address  = oc * 144 + ic * 9 + kr * 3 + kc
// conv2_bias_address    = oc
// conv2_output_address  = oc * 256 + out_row * 16 + out_col

`timescale 1ns / 1ps

module conv2_address_generator (
    input  logic [4:0]  output_channel,   // 0 .. 31
    input  logic [3:0]  output_row,       // 0 .. 15
    input  logic [3:0]  output_column,    // 0 .. 15
    input  logic [3:0]  input_channel,    // 0 .. 15
    input  logic [1:0]  kernel_row,       // 0 .. 2
    input  logic [1:0]  kernel_column,    // 0 .. 2

    output logic signed [5:0] input_row,      // may be -1 .. 17
    output logic signed [5:0] input_column,
    output logic              padding,
    output logic [11:0]       pool1_read_address,    // 0 .. 4095
    output logic [12:0]       conv2_weight_address,  // 0 .. 4607
    output logic [4:0]        conv2_bias_address,    // 0 .. 31
    output logic [12:0]       conv2_output_address   // 0 .. 8191
);

    logic signed [5:0] row_s;
    logic signed [5:0] col_s;
    logic in_bounds;

    always_comb begin
        // stride 1, padding 1
        row_s = $signed({1'b0, output_row}) + $signed({4'd0, kernel_row}) - 6'sd1;
        col_s = $signed({1'b0, output_column}) + $signed({4'd0, kernel_column}) - 6'sd1;
        input_row = row_s;
        input_column = col_s;

        in_bounds = (row_s >= 6'sd0) && (row_s < 6'sd16) &&
                    (col_s >= 6'sd0) && (col_s < 6'sd16);
        padding = ~in_bounds;

        // ic*256 + row*16 + col  (only meaningful when !padding)
        if (in_bounds) begin
            pool1_read_address = 12'(
                ({input_channel, 8'd0}) +
                ({4'd0, row_s[3:0], 4'd0}) +
                {8'd0, col_s[3:0]}
            );
        end else begin
            pool1_read_address = 12'd0;
        end

        // oc*144 + ic*9 + kr*3 + kc
        // 144 = 128+16 = {oc,7'd0} + {oc,4'd0}
        conv2_weight_address = 13'(
            ({output_channel, 7'd0}) + ({output_channel, 4'd0}) +
            ({input_channel, 3'd0}) + input_channel +
            ({kernel_row, 1'b0}) + kernel_row +
            kernel_column
        );

        conv2_bias_address = output_channel;

        // oc*256 + out_row*16 + out_col
        conv2_output_address = 13'(
            ({output_channel, 8'd0}) +
            ({4'd0, output_row, 4'd0}) +
            {9'd0, output_column}
        );
    end

endmodule
