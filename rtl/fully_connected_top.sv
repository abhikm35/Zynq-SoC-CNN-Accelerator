// fully_connected_top.sv
// Fully connected classifier: 32 GAP INT8 features -> 5 INT32 logits.
//
// Memories:
//   GAP feature ROM (verified GAP outputs)
//   Weight ROM [5 x 32]
//   Bias / multiplier / shift ROMs
//   Logit storage (5 x INT32)

`timescale 1ns / 1ps

module fully_connected_top #(
    parameter int NUM_FEATURES = 32,
    parameter int NUM_CLASSES = 5,
    parameter GAP_MEM_FILE   = "vectors/fc/gap_input.mem",
    parameter WGT_MEM_FILE   = "vectors/fc/fc_weights.mem",
    parameter BIAS_MEM_FILE  = "vectors/fc/fc_biases.mem",
    parameter MULT_MEM_FILE  = "vectors/fc/fc_multipliers.mem",
    parameter SHIFT_MEM_FILE = "vectors/fc/fc_shifts.mem"
) (
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    start,

    output logic                    busy,
    output logic                    fc_done,
    output logic [2:0]              class_index,
    output logic [3:0]              class_count,
    output logic [7:0]              total_mac_count,

    output logic signed [31:0]      engine_logit,
    output logic signed [31:0]      engine_accumulator,
    output logic [5:0]              engine_mac_count,
    output logic [4:0]              engine_input_index,
    output logic signed [15:0]      engine_product,
    output logic signed [31:0]      engine_running_acc,

    output logic                    logit_write_enable,
    output logic [2:0]              logit_write_address,
    output logic signed [31:0]      logit_write_data,

    input  logic                    logit_read_enable,
    input  logic [2:0]              logit_read_address,
    output logic signed [31:0]      logit_read_data
);

    localparam int WGT_DEPTH = NUM_CLASSES * NUM_FEATURES; // 160

    logic engine_start;
    logic [2:0] engine_class;
    logic engine_busy;
    logic engine_done;

    logic gap_re;
    logic [4:0] gap_addr;
    logic signed [7:0] gap_data;

    logic wgt_re;
    logic [7:0] wgt_addr;
    logic signed [7:0] wgt_data;

    logic bias_re, mult_re, shift_re;
    logic [2:0] bias_addr, mult_addr, shift_addr;
    logic signed [31:0] bias_data, mult_data, shift_data;

    int8_sync_rom #(
        .DEPTH(NUM_FEATURES),
        .ADDR_WIDTH(5),
        .MEM_FILE(GAP_MEM_FILE)
    ) u_gap_rom (
        .clk          (clk),
        .read_enable  (gap_re),
        .read_address (gap_addr),
        .read_data    (gap_data)
    );

    int8_sync_rom #(
        .DEPTH(WGT_DEPTH),
        .ADDR_WIDTH(8),
        .MEM_FILE(WGT_MEM_FILE)
    ) u_wgt_rom (
        .clk          (clk),
        .read_enable  (wgt_re),
        .read_address (wgt_addr),
        .read_data    (wgt_data)
    );

    int32_sync_rom #(
        .DEPTH(NUM_CLASSES),
        .ADDR_WIDTH(3),
        .MEM_FILE(BIAS_MEM_FILE)
    ) u_bias_rom (
        .clk          (clk),
        .read_enable  (bias_re),
        .read_address (bias_addr),
        .read_data    (bias_data)
    );

    int32_sync_rom #(
        .DEPTH(NUM_CLASSES),
        .ADDR_WIDTH(3),
        .MEM_FILE(MULT_MEM_FILE)
    ) u_mult_rom (
        .clk          (clk),
        .read_enable  (mult_re),
        .read_address (mult_addr),
        .read_data    (mult_data)
    );

    int32_sync_rom #(
        .DEPTH(NUM_CLASSES),
        .ADDR_WIDTH(3),
        .MEM_FILE(SHIFT_MEM_FILE)
    ) u_shift_rom (
        .clk          (clk),
        .read_enable  (shift_re),
        .read_address (shift_addr),
        .read_data    (shift_data)
    );

    logic engine_logit_valid;

    fully_connected_class_engine #(
        .NUM_FEATURES(NUM_FEATURES)
    ) u_engine (
        .clk                 (clk),
        .rst                 (rst),
        .start               (engine_start),
        .class_index         (engine_class),
        .busy                (engine_busy),
        .done                (engine_done),
        .logit_valid         (engine_logit_valid),
        .logit_value         (engine_logit),
        .final_accumulator   (engine_accumulator),
        .mac_count           (engine_mac_count),
        .debug_input_index   (engine_input_index),
        .debug_product       (engine_product),
        .debug_accumulator   (engine_running_acc),
        .gap_read_enable     (gap_re),
        .gap_read_address    (gap_addr),
        .gap_read_data       (gap_data),
        .weight_read_enable  (wgt_re),
        .weight_read_address (wgt_addr),
        .weight_read_data    (wgt_data),
        .bias_read_enable    (bias_re),
        .bias_read_address   (bias_addr),
        .bias_read_data      (bias_data),
        .mult_read_enable    (mult_re),
        .mult_read_address   (mult_addr),
        .mult_read_data      (mult_data),
        .shift_read_enable   (shift_re),
        .shift_read_address  (shift_addr),
        .shift_read_data     (shift_data)
    );

    /* verilator lint_off UNUSED */
    wire unused_logit_valid = engine_logit_valid;
    /* verilator lint_on UNUSED */

    fully_connected_layer_controller #(
        .NUM_CLASSES (NUM_CLASSES),
        .NUM_FEATURES(NUM_FEATURES)
    ) u_ctrl (
        .clk                   (clk),
        .rst                   (rst),
        .start                 (start),
        .busy                  (busy),
        .fc_done               (fc_done),
        .class_index           (class_index),
        .class_count           (class_count),
        .total_mac_count       (total_mac_count),
        .engine_start          (engine_start),
        .engine_class_index    (engine_class),
        .engine_busy           (engine_busy),
        .engine_done           (engine_done),
        .engine_logit          (engine_logit),
        .engine_accumulator    (engine_accumulator),
        .engine_mac_count      (engine_mac_count),
        .logit_write_enable    (logit_write_enable),
        .logit_write_address   (logit_write_address),
        .logit_write_data      (logit_write_data)
    );

    logit_storage #(
        .DEPTH(NUM_CLASSES),
        .ADDR_WIDTH(3)
    ) u_logits (
        .clk           (clk),
        .write_enable  (logit_write_enable),
        .write_address (logit_write_address),
        .write_data    (logit_write_data),
        .read_enable   (logit_read_enable),
        .read_address  (logit_read_address),
        .read_data     (logit_read_data)
    );

endmodule
