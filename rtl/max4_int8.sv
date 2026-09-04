// max4_int8.sv
// Signed INT8 maximum of four values (combinational).
// Uses signed comparisons only — never treat operands as unsigned.

`timescale 1ns / 1ps

module max4_int8 (
    input  logic signed [7:0] value_a,
    input  logic signed [7:0] value_b,
    input  logic signed [7:0] value_c,
    input  logic signed [7:0] value_d,
    output logic signed [7:0] maximum
);

    logic signed [7:0] max_ab;
    logic signed [7:0] max_cd;

    always_comb begin
        max_ab = (value_a > value_b) ? value_a : value_b;
        max_cd = (value_c > value_d) ? value_c : value_d;
        maximum = (max_ab > max_cd) ? max_ab : max_cd;
    end

endmodule
