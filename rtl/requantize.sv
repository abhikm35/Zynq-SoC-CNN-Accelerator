// requantize.sv
// Bit-exact match to software.quantization.fixed_point.requantize_int32
// with rounding_right_shift (ties away from zero):
//
//   wide_product = int64(acc) * int64(multiplier)
//   if shift == 0:
//       shifted = wide_product
//   else:
//       half = 1 << (shift-1)
//       if wide_product >= 0:
//           shifted = (wide_product + half) >> shift
//       else:
//           shifted = -(((-wide_product) + half) >> shift)
//   zero_point_adjusted = shifted + output_zero_point
//   saturated_value = clip(zero_point_adjusted, -128, 127)
//
// Multiplier/shift are module inputs so per-output-channel Conv1 params work.
// Combinational for unit testing; pipeline externally for timing
// (register wide_product, then requantize_from_product).

`timescale 1ns / 1ps

module requantize (
    input  logic signed [31:0]      accumulator,
    input  logic signed [31:0]      multiplier,
    input  logic        [5:0]       shift,              // observed Conv1 shifts ~38..41
    input  logic signed [7:0]       output_zero_point,  // 0 for this project
    output logic signed [63:0]      wide_product,
    output logic signed [63:0]      rounding_offset,
    output logic signed [63:0]      rounded_product,
    output logic signed [63:0]      shifted_value,
    output logic signed [31:0]      zero_point_adjusted,
    output logic signed [7:0]       saturated_value
);

    logic signed [63:0] acc_sext;
    logic signed [63:0] mult_sext;

    always_comb begin
        acc_sext  = {{32{accumulator[31]}}, accumulator};
        mult_sext = {{32{multiplier[31]}}, multiplier};
        wide_product = acc_sext * mult_sext;
    end

    requantize_from_product u_post (
        .wide_product        (wide_product),
        .shift               (shift),
        .output_zero_point   (output_zero_point),
        .rounding_offset     (rounding_offset),
        .rounded_product     (rounded_product),
        .shifted_value       (shifted_value),
        .zero_point_adjusted (zero_point_adjusted),
        .saturated_value     (saturated_value)
    );

endmodule
