// shared_maxpool_address_generator.sv
// Layer-selectable address helper for shared 2x2 stride-2 max-pool.
//
// Trained sizes (obsolete 8/16 prompt sketches are not used):
//   Pool1 (layer_is_pool2=0): 16 ch, 32x32 -> 16x16
//   Pool2 (layer_is_pool2=1): 32 ch, 16x16 ->  8x8
//
// Reuses verified MaxPool1/MaxPool2 address generators bit-exactly.

`timescale 1ns / 1ps

module shared_maxpool_address_generator (
    input  logic        layer_is_pool2,

    input  logic [4:0]  channel,       // 0..15 Pool1, 0..31 Pool2
    input  logic [3:0]  pool_row,      // 0..15 Pool1, 0..7  Pool2
    input  logic [3:0]  pool_column,
    input  logic [1:0]  window_index,

    output logic [4:0]  input_row,
    output logic [4:0]  input_column,
    output logic [13:0] activation_read_address,
    output logic [11:0] pool_write_address
);

    logic [4:0] p1_in_row, p1_in_col;
    logic [13:0] p1_rd;
    logic [11:0] p1_wr;

    logic [3:0] p2_in_row, p2_in_col;
    logic [12:0] p2_rd;
    logic [10:0] p2_wr;

    maxpool2x2_address_generator u_p1 (
        .channel             (channel[3:0]),
        .pool_row            (pool_row),
        .pool_column         (pool_column),
        .window_index        (window_index),
        .input_row           (p1_in_row),
        .input_column        (p1_in_col),
        .conv1_read_address  (p1_rd),
        .pool1_write_address (p1_wr)
    );

    maxpool2_address_generator u_p2 (
        .channel             (channel),
        .pool_row            (pool_row[2:0]),
        .pool_column         (pool_column[2:0]),
        .window_index        (window_index),
        .input_row           (p2_in_row),
        .input_column        (p2_in_col),
        .conv2_read_address  (p2_rd),
        .pool2_write_address (p2_wr)
    );

    always_comb begin
        if (!layer_is_pool2) begin
            input_row                = p1_in_row;
            input_column             = p1_in_col;
            activation_read_address  = p1_rd;
            pool_write_address       = p1_wr;
        end else begin
            input_row                = {1'b0, p2_in_row};
            input_column             = {1'b0, p2_in_col};
            activation_read_address  = {1'b0, p2_rd};
            pool_write_address       = {1'b0, p2_wr};
        end
    end

endmodule
