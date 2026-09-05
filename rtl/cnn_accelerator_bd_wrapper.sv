// cnn_accelerator_bd_wrapper.sv
// Block Design / Module Reference-compatible shell around the timing-closed
// cnn_accelerator_synth_wrapper.
//
// Why this exists:
//   Vivado IP Integrator "Add Module" hides cnn_accelerator_synth_wrapper when
//   "Hide incompatible modules" is checked, primarily because that module
//   declares many **string file-path parameters** (weight/bias .mem paths) and
//   a parameter-dependent port width. Module Reference cannot map those cleanly.
//
// This wrapper:
//   - exposes only simple scalar / packed ports (no string parameters)
//   - instantiates cnn_accelerator_synth_wrapper exactly once
//   - does not duplicate CNN datapath, AXI, sticky-DONE, or START pulsing
//
// Hierarchy:
//   Block Design -> cnn_accelerator_bd_wrapper -> cnn_accelerator_synth_wrapper
//                -> cnn_accelerator_shared_compute_top

`timescale 1ns / 1ps

module cnn_accelerator_bd_wrapper (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME clk, ASSOCIATED_RESET rst, FREQ_HZ 83333333" *)
    input  wire               clk,

    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rst RST" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME rst, POLARITY ACTIVE_HIGH" *)
    input  wire               rst,   // active-high (CNN core polarity)

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

    // Memory paths stay inside the hierarchy (not on the BD boundary).
    cnn_accelerator_synth_wrapper #(
        .LOGIT_WIDTH(32)
    ) u_synth (
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
