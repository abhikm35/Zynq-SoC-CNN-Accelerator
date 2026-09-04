// gap_average.sv
// Exact match to software.quantization.fixed_point.round_divide_int(sum, 64)
// followed by saturate to signed INT8 (same as integer_global_average_pool_nchw
// with zero_point == 0).
//
// denom = 64 = 2^6 -> rounding_right_shift(sum, 6), ties away from zero:
//   half = 32
//   if sum >= 0:  avg = (sum + 32) >>> 6
//   else:         avg = -(((-sum) + 32) >>> 6)
// then saturate to [-128, 127].
//
// No floating-point. Combinational.

`timescale 1ns / 1ps

module gap_average (
    input  logic signed [31:0] sum_value,

    output logic signed [31:0] rounding_adjustment,
    output logic signed [31:0] adjusted_sum,
    output logic signed [31:0] shifted_or_divided_value,
    output logic signed [7:0]  saturated_value,
    output logic signed [7:0]  final_output
);

    localparam logic signed [31:0] HALF = 32'sd32;

    logic signed [31:0] neg_sum;
    logic signed [31:0] mag_adj;
    logic signed [31:0] avg_w;
    logic signed [7:0]  sat_w;

    always_comb begin
        neg_sum = -sum_value;
        if (sum_value >= 32'sd0) begin
            rounding_adjustment = HALF;
            adjusted_sum = sum_value + HALF;
            avg_w = adjusted_sum >>> 6;
        end else begin
            rounding_adjustment = -HALF;
            mag_adj = neg_sum + HALF;
            adjusted_sum = -mag_adj; // debug: signed adjusted magnitude before shift
            avg_w = -(mag_adj >>> 6);
        end
        shifted_or_divided_value = avg_w;
    end

    saturate_int8 u_sat (
        .value     (avg_w),
        .saturated (sat_w)
    );

    assign saturated_value = sat_w;
    assign final_output    = sat_w;

endmodule
