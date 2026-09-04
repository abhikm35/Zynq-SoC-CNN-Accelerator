// maxpool2x2_address_generator.sv
// Combinational address helper for 2x2 stride-2 MaxPool1.
//
// Trained model: NUM_CHANNELS = 16 (prompt sketches with 8 are obsolete).
//
// base_row = pool_row * 2
// base_col = pool_column * 2
//
// window_index:
//   0: (base_row,     base_col)      top-left
//   1: (base_row,     base_col + 1)  top-right
//   2: (base_row + 1, base_col)      bottom-left
//   3: (base_row + 1, base_col + 1)  bottom-right
//
// conv1_read_address  = channel * 1024 + input_row * 32 + input_column
// pool1_write_address = channel * 256  + pool_row * 16  + pool_column

`timescale 1ns / 1ps

module maxpool2x2_address_generator (
    input  logic [3:0]  channel,       // 0 .. 15
    input  logic [3:0]  pool_row,      // 0 .. 15
    input  logic [3:0]  pool_column,   // 0 .. 15
    input  logic [1:0]  window_index,  // 0 .. 3

    output logic [4:0]  input_row,     // 0 .. 31
    output logic [4:0]  input_column,  // 0 .. 31
    output logic [13:0] conv1_read_address,  // 0 .. 16383
    output logic [11:0] pool1_write_address  // 0 .. 4095
);

    logic [4:0] base_row;
    logic [4:0] base_col;

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
                input_column = base_col + 5'd1;
            end
            2'd2: begin
                input_row    = base_row + 5'd1;
                input_column = base_col;
            end
            default: begin // 2'd3
                input_row    = base_row + 5'd1;
                input_column = base_col + 5'd1;
            end
        endcase

        // channel*1024 + row*32 + col
        conv1_read_address = 14'(
            ({channel, 10'd0}) + ({5'd0, input_row, 5'd0}) + {9'd0, input_column}
        );

        // channel*256 + pool_row*16 + pool_column
        pool1_write_address = 12'(
            ({channel, 8'd0}) + ({4'd0, pool_row, 4'd0}) + {8'd0, pool_column}
        );
    end

endmodule
