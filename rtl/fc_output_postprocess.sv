// fc_output_postprocess.sv
// Classifier score requantization: same rounding as requantize.sv, but
// saturates to full signed INT32 (Python qmin/qmax = INT32 limits).
//
//   logit = saturate_int32(round(acc * multiplier / 2^shift) + zp)
//
// Combinational for unit testing; pipeline externally for timing
// (register wide_product, then fc_logit_from_product).
// No ReLU — classifier logits may be negative.
// No floating-point.

`timescale 1ns / 1ps

module fc_output_postprocess (
    input  logic signed [31:0]      accumulator,
    input  logic signed [31:0]      multiplier,
    input  logic        [5:0]       shift,
    input  logic signed [7:0]       output_zero_point,  // 0 for this project

    output logic signed [63:0]      wide_product,
    output logic signed [63:0]      rounding_offset,
    output logic signed [63:0]      rounded_product,
    output logic signed [63:0]      shifted_value,
    output logic signed [31:0]      zero_point_adjusted,
    output logic signed [31:0]      logit_value
);

    logic signed [63:0] acc_sext;
    logic signed [63:0] mult_sext;

    always_comb begin
        acc_sext  = {{32{accumulator[31]}}, accumulator};
        mult_sext = {{32{multiplier[31]}}, multiplier};
        wide_product = acc_sext * mult_sext;
    end

    fc_logit_from_product u_post (
        .wide_product        (wide_product),
        .shift               (shift),
        .output_zero_point   (output_zero_point),
        .rounding_offset     (rounding_offset),
        .rounded_product     (rounded_product),
        .shifted_value       (shifted_value),
        .zero_point_adjusted (zero_point_adjusted),
        .logit_value         (logit_value)
    );

endmodule
