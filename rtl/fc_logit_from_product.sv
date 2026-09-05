// fc_logit_from_product.sv
// Second half of FC logit requantization (combinational wrapper).
// Pipeline with rounding_right_shift64 then saturate_shifted_int32 in the FSM.

`timescale 1ns / 1ps

module fc_logit_from_product (
    input  logic signed [63:0]      wide_product,
    input  logic        [5:0]       shift,
    input  logic signed [7:0]       output_zero_point,

    output logic signed [63:0]      rounding_offset,
    output logic signed [63:0]      rounded_product,
    output logic signed [63:0]      shifted_value,
    output logic signed [31:0]      zero_point_adjusted,
    output logic signed [31:0]      logit_value
);

    rounding_right_shift64 u_shift (
        .wide_product    (wide_product),
        .shift           (shift),
        .rounding_offset (rounding_offset),
        .rounded_product (rounded_product),
        .shifted_value   (shifted_value)
    );

    saturate_shifted_int32 u_sat (
        .shifted_value       (shifted_value),
        .output_zero_point   (output_zero_point),
        .zero_point_adjusted (zero_point_adjusted),
        .saturated_value     (logit_value)
    );

endmodule
