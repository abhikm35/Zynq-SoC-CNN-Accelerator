// shared_conv_engine.sv
// Full reusable convolution layer: controller + single-output datapath.
//
// layer_is_conv2 = 0 -> Conv1 (3->16, 32x32, 27 MACs, 16384 outputs)
// layer_is_conv2 = 1 -> Conv2 (16->32, 16x16, 144 MACs, 8192 outputs)
//
// Activations are EXTERNAL (1-cycle sync). Parameter ROMs stay separate
// banks inside the single-output engine, muxed by layer.

`timescale 1ns / 1ps

module shared_conv_engine #(
    parameter CONV1_WGT_MEM   = "vectors/conv1_memory/conv1_weights.mem",
    parameter CONV1_BIAS_MEM  = "vectors/conv1_memory/conv1_biases.mem",
    parameter CONV1_MULT_MEM  = "vectors/conv1_memory/conv1_multipliers.mem",
    parameter CONV1_SHIFT_MEM = "vectors/conv1_memory/conv1_shifts.mem",
    parameter CONV2_WGT_MEM   = "vectors/conv2/conv2_weights.mem",
    parameter CONV2_BIAS_MEM  = "vectors/conv2/conv2_biases.mem",
    parameter CONV2_MULT_MEM  = "vectors/conv2/conv2_multipliers.mem",
    parameter CONV2_SHIFT_MEM = "vectors/conv2/conv2_shifts.mem"
) (
    input  logic                   clk,
    input  logic                   rst,
    input  logic                   start,
    input  logic                   layer_is_conv2,

    input  logic signed [7:0]      external_act_data,

    output logic                   busy,
    output logic                   done,
    output logic [14:0]            output_count,
    output logic [4:0]             current_output_channel,
    output logic [4:0]             current_output_row,
    output logic [4:0]             current_output_column,
    output logic                   dbg_layer_is_conv2,

    output logic                   act_read_enable,
    output logic [11:0]            act_read_address,

    output logic                   output_write_enable,
    output logic [13:0]            output_write_address,
    output logic signed [7:0]      output_write_data,

    output logic [7:0]             dbg_engine_mac_count,
    output logic                   dbg_engine_done
);

    logic eng_start, eng_busy, eng_done;
    logic [4:0] eng_oc, eng_row, eng_col;
    logic signed [7:0] eng_relu;
    logic [7:0] eng_macs;

    logic signed [31:0] eng_final;
    logic signed [7:0] eng_rq;
    logic [3:0] eng_ic;
    logic [1:0] eng_kr, eng_kc;
    logic [11:0] eng_act_addr;
    logic [12:0] eng_wgt_addr;
    logic [4:0] eng_bias_addr;
    logic [13:0] eng_out_addr;
    logic eng_pad, eng_op_valid, eng_mac_en, eng_load_bias;
    logic signed [7:0] eng_act_v, eng_wgt_v;
    logic signed [15:0] eng_prod, eng_last_prod;
    logic signed [31:0] eng_acc;
    logic eng_wgt_re;
    logic eng_layer;
    logic ctrl_layer;
    logic eng_act_re_w;

    assign act_read_enable  = eng_act_re_w;
    assign act_read_address = eng_act_addr;
    assign dbg_engine_mac_count = eng_macs;
    assign dbg_engine_done = eng_done;
    assign dbg_layer_is_conv2 = ctrl_layer;

    shared_conv_layer_controller u_ctrl (
        .clk(clk), .rst(rst), .start(start),
        .layer_is_conv2(layer_is_conv2),
        .engine_start(eng_start),
        .engine_output_channel(eng_oc),
        .engine_output_row(eng_row),
        .engine_output_column(eng_col),
        .engine_busy(eng_busy),
        .engine_done(eng_done),
        .engine_relu_output(eng_relu),
        .engine_mac_count(eng_macs),
        .busy(busy),
        .layer_done(done),
        .output_count(output_count),
        .current_output_channel(current_output_channel),
        .current_output_row(current_output_row),
        .current_output_column(current_output_column),
        .dbg_layer_is_conv2(ctrl_layer),
        .output_write_enable(output_write_enable),
        .output_write_address(output_write_address),
        .output_write_data(output_write_data)
    );

    // Pixel engine uses controller-latched layer (stable for whole layer)
    shared_conv_single_output #(
        .CONV1_WGT_MEM(CONV1_WGT_MEM),
        .CONV1_BIAS_MEM(CONV1_BIAS_MEM),
        .CONV1_MULT_MEM(CONV1_MULT_MEM),
        .CONV1_SHIFT_MEM(CONV1_SHIFT_MEM),
        .CONV2_WGT_MEM(CONV2_WGT_MEM),
        .CONV2_BIAS_MEM(CONV2_BIAS_MEM),
        .CONV2_MULT_MEM(CONV2_MULT_MEM),
        .CONV2_SHIFT_MEM(CONV2_SHIFT_MEM)
    ) u_eng (
        .clk(clk), .rst(rst), .start(eng_start),
        .layer_is_conv2(ctrl_layer),
        .output_channel(eng_oc),
        .output_row(eng_row),
        .output_column(eng_col),
        .external_act_data(external_act_data),
        .busy(eng_busy), .done(eng_done),
        .final_accumulator(eng_final),
        .requantized_output(eng_rq),
        .relu_output(eng_relu),
        .mac_count(eng_macs),
        .current_input_channel(eng_ic),
        .current_kernel_row(eng_kr),
        .current_kernel_column(eng_kc),
        .current_activation_address(eng_act_addr),
        .current_weight_address(eng_wgt_addr),
        .current_bias_address(eng_bias_addr),
        .current_output_address(eng_out_addr),
        .current_padding(eng_pad),
        .current_activation_value(eng_act_v),
        .current_weight_value(eng_wgt_v),
        .current_product(eng_prod),
        .current_accumulator(eng_acc),
        .operand_valid(eng_op_valid),
        .mac_enable(eng_mac_en),
        .load_bias(eng_load_bias),
        .act_read_enable(eng_act_re_w),
        .wgt_read_enable(eng_wgt_re),
        .last_product(eng_last_prod),
        .dbg_layer_is_conv2(eng_layer)
    );

    // synopsys translate_off
    always_ff @(posedge clk) begin
        if (!rst && busy && start)
            $error("shared_conv_engine start while busy");
    end
    // synopsys translate_on

    /* verilator lint_off UNUSED */
    wire unused_dbg = |{eng_final, eng_rq, eng_ic, eng_kr, eng_kc, eng_wgt_addr,
        eng_bias_addr, eng_out_addr, eng_pad, eng_op_valid, eng_mac_en,
        eng_load_bias, eng_act_v, eng_wgt_v, eng_prod, eng_acc, eng_last_prod,
        eng_wgt_re, eng_layer};
    /* verilator lint_on UNUSED */

endmodule
