// maxpool2_address_generator.sv
// Combinational address helper for 2x2 stride-2 MaxPool2.
//
// Trained model: NUM_CHANNELS = 32 (prompt sketches with 16 are obsolete).
// Input:  Conv2 32 x 16 x 16
// Output: Pool2 32 x  8 x  8
//
// conv2_read_address  = channel * 256 + input_row * 16 + input_column
// pool2_write_address = channel * 64  + pool_row * 8   + pool_column

`timescale 1ns / 1ps

module maxpool2_address_generator (
    input  logic [4:0]  channel,       // 0 .. 31
    input  logic [2:0]  pool_row,      // 0 .. 7
    input  logic [2:0]  pool_column,   // 0 .. 7
    input  logic [1:0]  window_index,  // 0 .. 3

    output logic [3:0]  input_row,     // 0 .. 15
    output logic [3:0]  input_column,  // 0 .. 15
    output logic [12:0] conv2_read_address,  // 0 .. 8191
    output logic [10:0] pool2_write_address  // 0 .. 2047
);

    logic [3:0] base_row;
    logic [3:0] base_col;

    always_comb begin
        base_row = {pool_row, 1'b0};      // *2
        base_col = {pool_column, 1'b0};   // *2

        unique case (window_index)
            2'd0: begin
                input_row    = base_row;
                input_column = base_col;
            end
            2'd1: begin
                input_row    = base_row;
                input_column = base_col + 4'd1;
            end
            2'd2: begin
                input_row    = base_row + 4'd1;
                input_column = base_col;
            end
            default: begin
                input_row    = base_row + 4'd1;
                input_column = base_col + 4'd1;
            end
        endcase

        // channel*256 + row*16 + col
        conv2_read_address = 13'(
            ({channel, 8'd0}) + ({4'd0, input_row, 4'd0}) + {9'd0, input_column}
        );

        // channel*64 + pool_row*8 + pool_column
        pool2_write_address = 11'(
            ({channel, 6'd0}) + ({3'd0, pool_row, 3'd0}) + {8'd0, pool_column}
        );
    end

endmodule
