// fc_logit_from_product.sv
// Second half of FC logit requantization: round / shift / zero-point /
// saturate to signed INT32. Used to pipeline the 64-bit multiply away from
// this fabric logic. Bit-exact with the post-multiply path in
// fc_output_postprocess.sv.

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

    logic signed [63:0] neg_wide;
    logic signed [63:0] tmp;
    logic signed [63:0] with_zp;

    always_comb begin
        neg_wide = 64'sd0;
        tmp = 64'sd0;

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

        if (with_zp > 64'sd2147483647)
            logit_value = 32'sd2147483647;
        else if (with_zp < -64'sd2147483648)
            logit_value = -32'sd2147483648;
        else
            logit_value = with_zp[31:0];
    end

endmodule
