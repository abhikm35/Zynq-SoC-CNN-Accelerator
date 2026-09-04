// int8_mac.sv
// Signed INT8 x INT8 multiply-accumulate into INT32.
//
// Width notes:
//   activation, weight : signed 8-bit two's complement
//   product            : signed 16-bit (max | -128 * -128 | = 16384 < 32767)
//   accumulator        : signed 32-bit
// Product is sign-extended to 32 bits before addition (no unsigned widening).
//
// Controls:
//   load_bias : when high with enable, load bias into accumulator (clears prior sum)
//   enable    : when high without load_bias, perform one MAC
// Synchronous active-high rst clears accumulator to 0.

`timescale 1ns / 1ps

module int8_mac (
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    load_bias,
    input  logic                    enable,
    input  logic signed [7:0]       activation,
    input  logic signed [7:0]       weight,
    input  logic signed [31:0]      bias,
    output logic signed [15:0]      product,
    output logic signed [31:0]      accumulator
);

    // Combinational signed multiply; result held for TB visibility.
    logic signed [15:0] product_c;
    logic signed [31:0] product_sext;

    always_comb begin
        product_c   = activation * weight;
        product_sext = {{16{product_c[15]}}, product_c};
    end

    assign product = product_c;

    always_ff @(posedge clk) begin
        if (rst) begin
            accumulator <= 32'sd0;
        end else if (enable && load_bias) begin
            accumulator <= bias;
        end else if (enable) begin
            accumulator <= accumulator + product_sext;
        end
    end

endmodule
