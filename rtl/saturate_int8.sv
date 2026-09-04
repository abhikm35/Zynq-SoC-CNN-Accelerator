// saturate_int8.sv
// Signed saturation to int8 range before any wraparound cast.
//   > 127  -> 127
//   < -128 -> -128

`timescale 1ns / 1ps

module saturate_int8 (
    input  logic signed [31:0] value,
    output logic signed [7:0]  saturated
);

    always_comb begin
        if (value > 32'sd127)
            saturated = 8'sd127;
        else if (value < -32'sd128)
            saturated = -8'sd128;
        else
            saturated = value[7:0];
    end

endmodule
