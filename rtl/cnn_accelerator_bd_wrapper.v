// cnn_accelerator_bd_wrapper.v
//
// Plain-Verilog compatibility wrapper for Vivado IP Integrator
// Module Reference.
//
// The timing-closed CNN implementation remains in
// cnn_accelerator_synth_wrapper.sv.
//
// This wrapper exists because the current Vivado Module Reference flow
// does not accept the SystemVerilog file as the reference top
// (error: top file of type SystemVerilog is not allowed in the reference).
//
// Behavior:
//   - exposes only simple scalar / packed wire ports (no string parameters)
//   - instantiates cnn_accelerator_synth_wrapper exactly once
//   - no CNN arithmetic, AXI, sticky-DONE, or START-pulse logic
//
// Hierarchy:
//   Block Design -> cnn_accelerator_bd_wrapper.v
//                -> cnn_accelerator_synth_wrapper.sv
//                -> cnn_accelerator_shared_compute_top

`timescale 1ns / 1ps

module cnn_accelerator_bd_wrapper (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME clk, ASSOCIATED_RESET rst, FREQ_HZ 83333336" *)
    input  wire               clk,

    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rst RST" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME rst, POLARITY ACTIVE_HIGH" *)
    input  wire               rst,   // active-high (same polarity as synth wrapper)

    input  wire               start,
    output wire               busy,
    output wire               done,

    output wire [2:0]         predicted_class,
    output wire signed [31:0] maximum_logit,
    output wire signed [31:0] logit_0,
    output wire signed [31:0] logit_1,
    output wire signed [31:0] logit_2,
    output wire signed [31:0] logit_3,
    output wire signed [31:0] logit_4,
    output wire [63:0]        cycle_count,

    // Activation RAM A input-tensor load (plain RTL; not AXI)
    input  wire               input_write_enable,
    input  wire [11:0]        input_write_address,
    input  wire signed [7:0]  input_write_data
);

    // Use synth-wrapper default parameters (LOGIT_WIDTH, *.mem paths).
    // Do not expose model-memory path parameters on the BD boundary.
    cnn_accelerator_synth_wrapper u_cnn (
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
        .input_write_enable(input_write_enable),
        .input_write_address(input_write_address),
        .input_write_data(input_write_data)
    );

endmodule
