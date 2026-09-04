// fc_output_postprocess.sv
// Classifier score requantization: same rounding as requantize.sv, but
// saturates to full signed INT32 (Python qmin/qmax = INT32 limits).
//
//   logit = saturate_int32(round(acc * multiplier / 2^shift) + zp)
//
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
    logic signed [63:0] neg_wide;
    logic signed [63:0] tmp;
    logic signed [63:0] with_zp;

    always_comb begin
        acc_sext  = {{32{accumulator[31]}}, accumulator};
        mult_sext = {{32{multiplier[31]}}, multiplier};
        wide_product = acc_sext * mult_sext;

        if (shift == 6'd0) begin
            rounding_offset = 64'sd0;
            rounded_product = wide_product;
            shifted_value   = wide_product;
        end else begin
            rounding_offset = 64'sd1 <<< (shift - 6'd1);
            if (wide_product >= 64'sd0) begin
                rounded_product = wide_product + rounding_offset;
                shifted_value   = rounded_product >>> shift;
            end else begin
                neg_wide = -wide_product;
                tmp = neg_wide + rounding_offset;
                rounded_product = -(tmp);
                shifted_value   = -(tmp >>> shift);
            end
        end

        with_zp = shifted_value + {{56{output_zero_point[7]}}, output_zero_point};
        zero_point_adjusted = with_zp[31:0];

        // Saturate to signed INT32 full range (matches Python classifier scores).
        if (with_zp > 64'sd2147483647)
            logit_value = 32'sd2147483647;
        else if (with_zp < -64'sd2147483648)
            logit_value = -32'sd2147483648;
        else
            logit_value = with_zp[31:0];
    end

endmodule
