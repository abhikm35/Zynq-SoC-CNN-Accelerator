// relu_int8.sv
// Quantized-domain ReLU matching software.inference.integer_inference.integer_relu
// with zero_point = 0 (symmetric scheme):
//   output = max(input, zero_point) = max(input, 0)

`timescale 1ns / 1ps

module relu_int8 (
    input  logic signed [7:0]  in_value,
    input  logic signed [7:0]  zero_point,  // 0 for this project
    output logic signed [7:0]  out_value
);

    always_comb begin
        if (in_value > zero_point)
            out_value = in_value;
        else
            out_value = zero_point;
    end

endmodule
