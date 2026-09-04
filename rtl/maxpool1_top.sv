// maxpool1_top.sv
// MaxPool1 over the verified Conv1 output tensor.
// Conv1 source: sync ROM loaded from vectors (golden post-ReLU map).
// Pool1 sink: 4096-entry sync RAM (16 x 16 x 16).

`timescale 1ns / 1ps

module maxpool1_top #(
    parameter int NUM_CHANNELS = 16,
    parameter CONV1_MEM_FILE = "vectors/pool1/conv1_input_for_pool.mem"
) (
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    start,

    output logic                    busy,
    output logic                    pool1_done,
    output logic [12:0]             output_count,
    output logic [3:0]              current_channel,
    output logic [3:0]              current_pool_row,
    output logic [3:0]              current_pool_column,
    output logic [1:0]              current_window_index,

    output logic                    conv1_read_enable,
    output logic [13:0]             conv1_read_address,
    output logic signed [7:0]       conv1_read_data,

    output logic                    pool1_write_enable,
    output logic [11:0]             pool1_write_address,
    output logic signed [7:0]       pool1_write_data,

    output logic signed [7:0]       value_a,
    output logic signed [7:0]       value_b,
    output logic signed [7:0]       value_c,
    output logic signed [7:0]       value_d,
    output logic signed [7:0]       maximum_value,

    // Pool1 read port for TB / future Conv2
    input  logic                    pool1_read_enable,
    input  logic [11:0]             pool1_read_address,
    output logic signed [7:0]       pool1_read_data
);

    localparam int CONV1_DEPTH = NUM_CHANNELS * 1024;
    localparam int POOL1_DEPTH = NUM_CHANNELS * 256;

    logic signed [7:0] conv1_data_w;

    assign conv1_read_data = conv1_data_w;

    int8_sync_rom #(
        .DEPTH(CONV1_DEPTH),
        .ADDR_WIDTH(14),
        .MEM_FILE(CONV1_MEM_FILE)
    ) u_conv1_rom (
        .clk          (clk),
        .read_enable  (conv1_read_enable),
        .read_address (conv1_read_address),
        .read_data    (conv1_data_w)
    );

    maxpool1_controller #(
        .NUM_CHANNELS(NUM_CHANNELS)
    ) u_ctrl (
        .clk                   (clk),
        .rst                   (rst),
        .start                 (start),
        .busy                  (busy),
        .pool1_done            (pool1_done),
        .conv1_read_enable     (conv1_read_enable),
        .conv1_read_address    (conv1_read_address),
        .conv1_read_data       (conv1_data_w),
        .pool1_write_enable    (pool1_write_enable),
        .pool1_write_address   (pool1_write_address),
        .pool1_write_data      (pool1_write_data),
        .current_channel       (current_channel),
        .current_pool_row      (current_pool_row),
        .current_pool_column   (current_pool_column),
        .current_window_index  (current_window_index),
        .output_count          (output_count),
        .value_a               (value_a),
        .value_b               (value_b),
        .value_c               (value_c),
        .value_d               (value_d),
        .maximum_value         (maximum_value)
    );

    int8_sync_ram #(
        .DEPTH(POOL1_DEPTH),
        .ADDR_WIDTH(12)
    ) u_pool1_ram (
        .clk           (clk),
        .write_enable  (pool1_write_enable),
        .write_address (pool1_write_address),
        .write_data    (pool1_write_data),
        .read_enable   (pool1_read_enable),
        .read_address  (pool1_read_address),
        .read_data     (pool1_read_data)
    );

endmodule
