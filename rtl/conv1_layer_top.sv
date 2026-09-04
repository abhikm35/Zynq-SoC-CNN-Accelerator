// conv1_layer_top.sv
// Complete Conv1 (trained 3->16): all spatial outputs of all output channels.
// Reuses one memory-driven single-output engine; stores 16*32*32 INT8 results.

`timescale 1ns / 1ps

module conv1_layer_top #(
    parameter int NUM_OUT_CHANNELS = 16,
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
    output logic                   conv1_done,
    output logic [14:0]            output_count,
    output logic [3:0]             current_output_channel,
    output logic [4:0]             current_output_row,
    output logic [4:0]             current_output_column,

    input  logic                   out_read_enable,
    input  logic [13:0]            out_read_address,
    output logic signed [7:0]      out_read_data,

    output logic                   output_write_enable,
    output logic [13:0]            output_write_address,
    output logic signed [7:0]      output_write_data,
    output logic                   engine_start,
    output logic                   engine_busy,
    output logic                   engine_done,
    output logic [4:0]             engine_mac_count,
    output logic signed [31:0]     engine_final_accumulator,
    output logic signed [7:0]      engine_requantized_output,
    output logic signed [7:0]      engine_relu_output,
    output logic [3:0]             engine_output_channel,
    output logic [3:0]             engine_bias_address,
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

    localparam int OUT_DEPTH = NUM_OUT_CHANNELS * 1024;
    localparam int OUT_ADDR_W = 14;

    logic engine_start_w;
    logic [4:0] eng_row;
    logic [4:0] eng_col;
    logic act_re_unused;
    logic wgt_re_unused;

    assign engine_start = engine_start_w;

    conv1_layer_controller #(
        .NUM_OUT_CHANNELS(NUM_OUT_CHANNELS)
    ) u_ctrl (
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
        .conv1_done              (conv1_done),
        .output_count            (output_count),
        .current_output_channel  (current_output_channel),
        .current_output_row      (current_output_row),
        .current_output_column   (current_output_column),
        .output_write_enable     (output_write_enable),
        .output_write_address    (output_write_address),
        .output_write_data       (output_write_data)
    );

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
        .current_bias_address        (engine_bias_address),
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
        .DEPTH(OUT_DEPTH),
        .ADDR_WIDTH(OUT_ADDR_W)
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
