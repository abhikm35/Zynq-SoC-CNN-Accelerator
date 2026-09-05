// requantize_from_product.sv
// Second half of requantize_int32: round / shift / zero-point / saturate.
// Used to pipeline the 64-bit multiply away from this fabric logic.
// Bit-exact with the post-multiply path in requantize.sv.

`timescale 1ns / 1ps

module requantize_from_product (
    input  logic signed [63:0]      wide_product,
    input  logic        [5:0]       shift,
    input  logic signed [7:0]       output_zero_point,

    output logic signed [63:0]      rounding_offset,
    output logic signed [63:0]      rounded_product,
    output logic signed [63:0]      shifted_value,
    output logic signed [31:0]      zero_point_adjusted,
    output logic signed [7:0]       saturated_value
);

    logic signed [63:0] neg_wide;
    logic signed [63:0] tmp;

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

        zero_point_adjusted = shifted_value[31:0] + {{24{output_zero_point[7]}}, output_zero_point};

        if (zero_point_adjusted > 32'sd127)
            saturated_value = 8'sd127;
        else if (zero_point_adjusted < -32'sd128)
            saturated_value = -8'sd128;
        else
            saturated_value = zero_point_adjusted[7:0];
    end

endmodule
