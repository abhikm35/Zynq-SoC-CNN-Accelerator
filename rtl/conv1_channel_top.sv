// conv1_channel_top.sv
// One Conv1 output feature-map channel (oc=0, 32x32):
//   full input/weight/bias memories
//   -> memory-driven single-output engine (27 MACs, all 3 input channels)
//   -> channel controller iterates all spatial coords
//   -> 1024-entry output activation RAM

`timescale 1ns / 1ps

module conv1_channel_top #(
    parameter ACT_MEM_FILE   = "vectors/conv1_memory/input_image.mem",
    parameter WGT_MEM_FILE   = "vectors/conv1_memory/conv1_weights.mem",
    parameter BIAS_MEM_FILE  = "vectors/conv1_memory/conv1_biases.mem",
    parameter MULT_MEM_FILE  = "vectors/conv1_memory/conv1_multipliers.mem",
    parameter SHIFT_MEM_FILE = "vectors/conv1_memory/conv1_shifts.mem"
) (
    input  logic                   clk,
    input  logic                   rst,
    input  logic                   start,

    output logic                   busy,
    output logic                   channel_done,
    output logic [10:0]            output_count,
    output logic [4:0]             current_output_row,
    output logic [4:0]             current_output_column,

    // Output RAM read port (TB / future stages)
    input  logic                   out_read_enable,
    input  logic [9:0]             out_read_address,
    output logic signed [7:0]      out_read_data,

    // Debug: write side + engine status
    output logic                   output_write_enable,
    output logic [9:0]             output_write_address,
    output logic signed [7:0]      output_write_data,
    output logic                   engine_start,
    output logic                   engine_busy,
    output logic                   engine_done,
    output logic [4:0]             engine_mac_count,
    output logic signed [31:0]     engine_final_accumulator,
    output logic signed [7:0]      engine_requantized_output,
    output logic signed [7:0]      engine_relu_output,
    output logic [3:0]             engine_output_channel,
    // Selected-pixel MAC debug
    output logic                   engine_operand_valid,
    output logic                   engine_mac_enable,
    output logic                   engine_load_bias,
    output logic [1:0]             engine_ic,
    output logic [1:0]             engine_kr,
    output logic [1:0]             engine_kc,
    output logic [11:0]            engine_act_addr,
    output logic [8:0]             engine_wgt_addr,
    output logic                   engine_padding,
    output logic signed [7:0]      engine_act_value,
    output logic signed [7:0]      engine_wgt_value,
    output logic signed [15:0]     engine_product,
    output logic signed [31:0]     engine_accumulator,
    output logic signed [15:0]     engine_last_product
);

    logic engine_start_w;
    logic [4:0] eng_row;
    logic [4:0] eng_col;

    assign engine_start = engine_start_w;

    conv1_channel_controller u_ctrl (
        .clk                     (clk),
        .rst                     (rst),
        .start                   (start),
        .engine_start            (engine_start_w),
        .engine_output_channel   (engine_output_channel),
        .engine_output_row       (eng_row),
        .engine_output_column    (eng_col),
        .engine_busy             (engine_busy),
        .engine_done             (engine_done),
        .engine_relu_output      (engine_relu_output),
        .engine_mac_count        (engine_mac_count),
        .busy                    (busy),
        .channel_done            (channel_done),
        .output_count            (output_count),
        .current_output_row      (current_output_row),
        .current_output_column   (current_output_column),
        .output_write_enable     (output_write_enable),
        .output_write_address    (output_write_address),
        .output_write_data       (output_write_data)
    );

    logic [3:0] bias_addr_unused;
    logic       act_re_unused;
    logic       wgt_re_unused;

    conv1_memory_single_output #(
        .ACT_MEM_FILE  (ACT_MEM_FILE),
        .WGT_MEM_FILE  (WGT_MEM_FILE),
        .BIAS_MEM_FILE (BIAS_MEM_FILE),
        .MULT_MEM_FILE (MULT_MEM_FILE),
        .SHIFT_MEM_FILE(SHIFT_MEM_FILE)
    ) u_engine (
        .clk                         (clk),
        .rst                         (rst),
        .start                       (engine_start_w),
        .output_channel              (engine_output_channel),
        .output_row                  (eng_row),
        .output_column               (eng_col),
        .external_act_data           (8'sd0),
        .busy                        (engine_busy),
        .done                        (engine_done),
        .final_accumulator           (engine_final_accumulator),
        .requantized_output          (engine_requantized_output),
        .relu_output                 (engine_relu_output),
        .mac_count                   (engine_mac_count),
        .current_input_channel       (engine_ic),
        .current_kernel_row          (engine_kr),
        .current_kernel_column       (engine_kc),
        .current_activation_address  (engine_act_addr),
        .current_weight_address      (engine_wgt_addr),
        .current_bias_address        (bias_addr_unused),
        .current_padding             (engine_padding),
        .current_activation_value    (engine_act_value),
        .current_weight_value        (engine_wgt_value),
        .current_product             (engine_product),
        .current_accumulator         (engine_accumulator),
        .operand_valid               (engine_operand_valid),
        .mac_enable                  (engine_mac_enable),
        .load_bias                   (engine_load_bias),
        .act_read_enable             (act_re_unused),
        .wgt_read_enable             (wgt_re_unused),
        .last_product                (engine_last_product)
    );

    int8_sync_ram #(
        .DEPTH(1024),
        .ADDR_WIDTH(10)
    ) u_out_ram (
        .clk           (clk),
        .write_enable  (output_write_enable),
        .write_address (output_write_address),
        .write_data    (output_write_data),
        .read_enable   (out_read_enable),
        .read_address  (out_read_address),
        .read_data     (out_read_data)
    );

endmodule
