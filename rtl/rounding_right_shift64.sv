// rounding_right_shift64.sv
// Ties-away-from-zero rounding right shift for signed INT64 products.
// Matches software.quantization.fixed_point.rounding_right_shift / requantize_int32
// pre-saturate path. Combinational; register externally for timing.

`timescale 1ns / 1ps

module rounding_right_shift64 (
    input  logic signed [63:0]      wide_product,
    input  logic        [5:0]       shift,

    output logic signed [63:0]      rounding_offset,
    output logic signed [63:0]      rounded_product,
    output logic signed [63:0]      shifted_value
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
    end

endmodule
