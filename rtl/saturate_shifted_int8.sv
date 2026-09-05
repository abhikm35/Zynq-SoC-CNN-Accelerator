// saturate_shifted_int8.sv
// Add output zero-point and saturate to signed INT8.
// Input is the already round-shifted INT64 value from rounding_right_shift64.

`timescale 1ns / 1ps

module saturate_shifted_int8 (
    input  logic signed [63:0]      shifted_value,
    input  logic signed [7:0]       output_zero_point,

    output logic signed [31:0]      zero_point_adjusted,
    output logic signed [7:0]       saturated_value
);

    always_comb begin
        zero_point_adjusted = shifted_value[31:0] + {{24{output_zero_point[7]}}, output_zero_point};

        if (zero_point_adjusted > 32'sd127)
            saturated_value = 8'sd127;
        else if (zero_point_adjusted < -32'sd128)
            saturated_value = -8'sd128;
        else
            saturated_value = zero_point_adjusted[7:0];
    end

endmodule
