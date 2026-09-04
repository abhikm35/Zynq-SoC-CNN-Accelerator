// maxpool2_top.sv
// MaxPool2 over the verified Conv2/ReLU2 tensor.
// Conv2 source: sync ROM from vectors. Pool2 sink: 2048-entry sync RAM.

`timescale 1ns / 1ps

module maxpool2_top #(
    parameter int NUM_CHANNELS = 32,
    parameter CONV2_MEM_FILE = "vectors/pool2/conv2_input.mem"
) (
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    start,

    output logic                    busy,
    output logic                    pool2_done,
    output logic [11:0]             output_count,
    output logic [13:0]             read_count,
    output logic [4:0]              current_channel,
    output logic [2:0]              current_pool_row,
    output logic [2:0]              current_pool_column,
    output logic [1:0]              current_window_index,

    output logic                    conv2_read_enable,
    output logic [12:0]             conv2_read_address,
    output logic signed [7:0]       conv2_read_data,

    output logic                    pool2_write_enable,
    output logic [10:0]             pool2_write_address,
    output logic signed [7:0]       pool2_write_data,

    output logic signed [7:0]       value_a,
    output logic signed [7:0]       value_b,
    output logic signed [7:0]       value_c,
    output logic signed [7:0]       value_d,
    output logic signed [7:0]       maximum_value,

    input  logic                    pool2_read_enable,
    input  logic [10:0]             pool2_read_address,
    output logic signed [7:0]       pool2_read_data
);

    localparam int CONV2_DEPTH = NUM_CHANNELS * 256; // 8192
    localparam int POOL2_DEPTH = NUM_CHANNELS * 64;  // 2048

    logic signed [7:0] conv2_data_w;
    assign conv2_read_data = conv2_data_w;

    int8_sync_rom #(
        .DEPTH(CONV2_DEPTH),
        .ADDR_WIDTH(13),
        .MEM_FILE(CONV2_MEM_FILE)
    ) u_conv2_rom (
        .clk          (clk),
        .read_enable  (conv2_read_enable),
        .read_address (conv2_read_address),
        .read_data    (conv2_data_w)
    );

    maxpool2_controller #(
        .NUM_CHANNELS(NUM_CHANNELS)
    ) u_ctrl (
        .clk                   (clk),
        .rst                   (rst),
        .start                 (start),
        .busy                  (busy),
        .pool2_done            (pool2_done),
        .conv2_read_enable     (conv2_read_enable),
        .conv2_read_address    (conv2_read_address),
        .conv2_read_data       (conv2_data_w),
        .pool2_write_enable    (pool2_write_enable),
        .pool2_write_address   (pool2_write_address),
        .pool2_write_data      (pool2_write_data),
        .current_channel       (current_channel),
        .current_pool_row      (current_pool_row),
        .current_pool_column   (current_pool_column),
        .current_window_index  (current_window_index),
        .output_count          (output_count),
        .read_count            (read_count),
        .value_a               (value_a),
        .value_b               (value_b),
        .value_c               (value_c),
        .value_d               (value_d),
        .maximum_value         (maximum_value)
    );

    int8_sync_ram #(
        .DEPTH(POOL2_DEPTH),
        .ADDR_WIDTH(11)
    ) u_pool2_ram (
        .clk           (clk),
        .write_enable  (pool2_write_enable),
        .write_address (pool2_write_address),
        .write_data    (pool2_write_data),
        .read_enable   (pool2_read_enable),
        .read_address  (pool2_read_address),
        .read_data     (pool2_read_data)
    );

endmodule
