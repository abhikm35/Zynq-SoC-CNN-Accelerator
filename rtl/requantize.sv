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
// No floating-point. Combinational for unit testing; register externally if needed.

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
    logic signed [63:0] neg_wide;
    logic signed [63:0] tmp;

    always_comb begin
        acc_sext  = {{32{accumulator[31]}}, accumulator};
        mult_sext = {{32{multiplier[31]}}, multiplier};
        wide_product = acc_sext * mult_sext;

        if (shift == 6'd0) begin
            rounding_offset = 64'sd0;
            rounded_product = wide_product;
            shifted_value   = wide_product;
        end else begin
            // half = 2^(shift-1)
            rounding_offset = 64'sd1 <<< (shift - 6'd1);
            if (wide_product >= 64'sd0) begin
                rounded_product = wide_product + rounding_offset;
                shifted_value   = rounded_product >>> shift;
            end else begin
                neg_wide = -wide_product;
                tmp = neg_wide + rounding_offset;
                rounded_product = -(tmp); // debug: negated rounded magnitude before shift
                shifted_value   = -(tmp >>> shift);
            end
        end

        zero_point_adjusted = shifted_value[31:0] + {{24{output_zero_point[7]}}, output_zero_point};

        if (zero_point_adjusted > 32'sd127)
            saturated_value = 8'sd127;
        else if (zero_point_adjusted < -32'sd128)
            saturated_value = -8'sd128;
        else
            saturated_value = zero_point_adjusted[7:0];
    end

endmodule
