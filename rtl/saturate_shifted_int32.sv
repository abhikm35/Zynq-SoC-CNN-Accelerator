// saturate_shifted_int32.sv
// Add output zero-point and saturate to signed INT32 (FC logits).
// Input is the already round-shifted INT64 value from rounding_right_shift64.

`timescale 1ns / 1ps

module saturate_shifted_int32 (
    input  logic signed [63:0]      shifted_value,
    input  logic signed [7:0]       output_zero_point,

    output logic signed [31:0]      zero_point_adjusted,
    output logic signed [31:0]      saturated_value
);

    logic signed [63:0] with_zp;

    always_comb begin
        with_zp = shifted_value + {{56{output_zero_point[7]}}, output_zero_point};
        zero_point_adjusted = with_zp[31:0];

        if (with_zp > 64'sd2147483647)
            saturated_value = 32'sd2147483647;
        else if (with_zp < -64'sd2147483648)
            saturated_value = -32'sd2147483648;
        else
            saturated_value = with_zp[31:0];
    end

endmodule
