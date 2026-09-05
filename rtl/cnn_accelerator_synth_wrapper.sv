// cnn_accelerator_synth_wrapper.sv
// Minimal top for FPGA place/route characterization.
//
// The full cnn_accelerator_shared_compute_top exposes many TB/debug/host
// ports (~1000 IOs). xc7z010clg400-1 only has ~230 user IOs, so that top
// cannot be placed as-is. This wrapper keeps a small pinout and leaves the
// verified accelerator intact underneath (no arithmetic changes).
//
// Exposed pins:
//   clk, rst, start, busy, done, predicted_class, maximum_logit
//   logit_0..4, cycle_count
//   input_write_enable/address/data  (Activation RAM A load; for future PS path)
//
// NOTE: This module has string file-path parameters. Vivado IP Integrator
// Module Reference typically treats that as incompatible. For Block Design,
// use cnn_accelerator_bd_wrapper (no string parameters on its boundary).

`timescale 1ns / 1ps

module cnn_accelerator_synth_wrapper #(
    parameter int LOGIT_WIDTH = 32,
    parameter CONV1_WGT_MEM  = "vectors/conv1_memory/conv1_weights.mem",
    parameter CONV1_BIAS_MEM = "vectors/conv1_memory/conv1_biases.mem",
    parameter CONV1_MULT_MEM = "vectors/conv1_memory/conv1_multipliers.mem",
    parameter CONV1_SHIFT_MEM= "vectors/conv1_memory/conv1_shifts.mem",
    parameter CONV2_WGT_MEM  = "vectors/conv2/conv2_weights.mem",
    parameter CONV2_BIAS_MEM = "vectors/conv2/conv2_biases.mem",
    parameter CONV2_MULT_MEM = "vectors/conv2/conv2_multipliers.mem",
    parameter CONV2_SHIFT_MEM= "vectors/conv2/conv2_shifts.mem",
    parameter FC_WGT_MEM     = "vectors/fc/fc_weights.mem",
    parameter FC_BIAS_MEM    = "vectors/fc/fc_biases.mem",
    parameter FC_MULT_MEM    = "vectors/fc/fc_multipliers.mem",
    parameter FC_SHIFT_MEM   = "vectors/fc/fc_shifts.mem"
) (
    input  logic                          clk,
    input  logic                          rst,
    input  logic                          start,
    output logic                          busy,
    output logic                          done,
    output logic [2:0]                    predicted_class,
    output logic signed [LOGIT_WIDTH-1:0] maximum_logit,
    output logic signed [LOGIT_WIDTH-1:0] logit_0,
    output logic signed [LOGIT_WIDTH-1:0] logit_1,
    output logic signed [LOGIT_WIDTH-1:0] logit_2,
    output logic signed [LOGIT_WIDTH-1:0] logit_3,
    output logic signed [LOGIT_WIDTH-1:0] logit_4,
    output logic [63:0]                   cycle_count,
    // Optional host load into Activation RAM A (input tensor). Tied off by
    // characterization flows that do not need runtime image load.
    input  logic                          input_write_enable,
    input  logic [11:0]                   input_write_address,
    input  logic signed [7:0]             input_write_data
);

    // Keep the full accelerator from being optimized away during synth.
    (* DONT_TOUCH = "true" *)
    cnn_accelerator_shared_compute_top #(
        .LOGIT_WIDTH(LOGIT_WIDTH),
        .CONV1_WGT_MEM(CONV1_WGT_MEM),
        .CONV1_BIAS_MEM(CONV1_BIAS_MEM),
        .CONV1_MULT_MEM(CONV1_MULT_MEM),
        .CONV1_SHIFT_MEM(CONV1_SHIFT_MEM),
        .CONV2_WGT_MEM(CONV2_WGT_MEM),
        .CONV2_BIAS_MEM(CONV2_BIAS_MEM),
        .CONV2_MULT_MEM(CONV2_MULT_MEM),
        .CONV2_SHIFT_MEM(CONV2_SHIFT_MEM),
        .FC_WGT_MEM(FC_WGT_MEM),
        .FC_BIAS_MEM(FC_BIAS_MEM),
        .FC_MULT_MEM(FC_MULT_MEM),
        .FC_SHIFT_MEM(FC_SHIFT_MEM)
    ) u_cnn (
        .clk(clk),
        .rst(rst),
        .start(start),
        .busy(busy),
        .done(done),
        .predicted_class(predicted_class),
        .maximum_logit(maximum_logit),
        .logit_0(logit_0),
        .logit_1(logit_1),
        .logit_2(logit_2),
        .logit_3(logit_3),
        .logit_4(logit_4),
        .cycle_count(cycle_count),
        .state_id(),
        .stage_id(),
        .dbg_conv1_done(),
        .dbg_pool1_done(),
        .dbg_conv2_done(),
        .dbg_pool2_done(),
        .dbg_gap_done(),
        .dbg_fc_done(),
        .dbg_argmax_done(),
        .dbg_conv1_output_count(),
        .dbg_pool1_output_count(),
        .dbg_conv2_output_count(),
        .dbg_pool2_output_count(),
        .dbg_gap_output_count(),
        .dbg_fc_mac_count(),
        .dbg_conv1_cycles(),
        .dbg_pool1_cycles(),
        .dbg_conv2_cycles(),
        .dbg_pool2_cycles(),
        .dbg_gap_cycles(),
        .dbg_fc_cycles(),
        .dbg_argmax_cycles(),
        .own_conv1(),
        .own_pool1(),
        .own_conv2(),
        .own_pool2(),
        .own_gap(),
        .dbg_act_a_we(),
        .dbg_act_a_waddr(),
        .dbg_act_a_wdata(),
        .dbg_act_b_we(),
        .dbg_act_b_waddr(),
        .dbg_act_b_wdata(),
        .dbg_act_a_re(),
        .dbg_act_a_raddr(),
        .dbg_act_b_re(),
        .dbg_act_b_raddr(),

        .input_write_enable(input_write_enable),
        .input_write_address(input_write_address),
        .input_write_data(input_write_data),
        .act_a_read_enable(1'b0),
        .act_a_read_address(14'd0),
        .act_a_read_data(),
        .act_b_read_enable(1'b0),
        .act_b_read_address(14'd0),
        .act_b_read_data(),
        .gap_read_enable(1'b0),
        .gap_read_address(5'd0),
        .gap_read_data(),
        .logit_read_enable(1'b0),
        .logit_read_address(3'd0),
        .logit_read_data()
    );

endmodule
