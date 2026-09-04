// global_average_pool_top.sv
// Global Average Pooling over verified Pool2 (32 x 8 x 8).
// Pool2 source: sync ROM from vectors. GAP sink: 32-entry storage.

`timescale 1ns / 1ps

module global_average_pool_top #(
    parameter int NUM_CHANNELS = 32,
    parameter POOL2_MEM_FILE = "vectors/gap/pool2_input.mem",
    parameter logic signed [31:0] GAP_MULTIPLIER = 32'sd1759306569,
    parameter logic [5:0]         GAP_SHIFT = 6'd29
) (
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    start,

    output logic                    busy,
    output logic                    gap_done,
    output logic [11:0]             read_count,
    output logic [5:0]              output_count,
    output logic [4:0]              current_channel,
    output logic [5:0]              current_element,
    output logic signed [31:0]      running_sum,
    output logic signed [31:0]      final_sum,
    output logic signed [7:0]       averaged_int8,
    output logic signed [7:0]       gap_int8,

    output logic                    pool2_read_enable,
    output logic [10:0]             pool2_read_address,
    output logic signed [7:0]       pool2_read_data,

    output logic                    gap_write_enable,
    output logic [4:0]              gap_write_address,
    output logic signed [7:0]       gap_write_data,

    input  logic                    gap_read_enable,
    input  logic [4:0]              gap_read_address,
    output logic signed [7:0]       gap_read_data
);

    localparam int POOL2_DEPTH = NUM_CHANNELS * 64; // 2048

    logic signed [7:0] pool2_data_w;
    assign pool2_read_data = pool2_data_w;

    int8_sync_rom #(
        .DEPTH(POOL2_DEPTH),
        .ADDR_WIDTH(11),
        .MEM_FILE(POOL2_MEM_FILE)
    ) u_pool2_rom (
        .clk          (clk),
        .read_enable  (pool2_read_enable),
        .read_address (pool2_read_address),
        .read_data    (pool2_data_w)
    );

    global_average_pool_controller #(
        .NUM_CHANNELS  (NUM_CHANNELS),
        .GAP_MULTIPLIER(GAP_MULTIPLIER),
        .GAP_SHIFT     (GAP_SHIFT)
    ) u_ctrl (
        .clk                (clk),
        .rst                (rst),
        .start              (start),
        .busy               (busy),
        .gap_done           (gap_done),
        .pool2_read_enable  (pool2_read_enable),
        .pool2_read_address (pool2_read_address),
        .pool2_read_data    (pool2_data_w),
        .gap_write_enable   (gap_write_enable),
        .gap_write_address  (gap_write_address),
        .gap_write_data     (gap_write_data),
        .current_channel    (current_channel),
        .current_element    (current_element),
        .running_sum        (running_sum),
        .read_count         (read_count),
        .output_count       (output_count),
        .final_sum          (final_sum),
        .averaged_int8      (averaged_int8),
        .gap_int8           (gap_int8)
    );

    gap_output_storage #(
        .DEPTH(NUM_CHANNELS),
        .ADDR_WIDTH(5)
    ) u_gap_store (
        .clk           (clk),
        .write_enable  (gap_write_enable),
        .write_address (gap_write_address),
        .write_data    (gap_write_data),
        .read_enable   (gap_read_enable),
        .read_address  (gap_read_address),
        .read_data     (gap_read_data)
    );

endmodule
