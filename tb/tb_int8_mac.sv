// tb_int8_mac.sv — Verilator-friendly (negedge stimulus / posedge check)
`timescale 1ns / 1ps

module tb_int8_mac (
    input logic clk
);
    logic rst;
    logic load_bias;
    logic enable;
    logic signed [7:0] activation;
    logic signed [7:0] weight;
    logic signed [31:0] bias;
    logic signed [15:0] product;
    logic signed [31:0] accumulator;

    int8_mac dut (
        .clk(clk), .rst(rst), .load_bias(load_bias), .enable(enable),
        .activation(activation), .weight(weight), .bias(bias),
        .product(product), .accumulator(accumulator)
    );

    integer cycle;
    integer errors;

    initial begin
        cycle = 0;
        errors = 0;
        rst = 1'b1;
        load_bias = 1'b0;
        enable = 1'b0;
        activation = 0;
        weight = 0;
        bias = 0;
    end

    always @(negedge clk) begin
        cycle = cycle + 1;
        case (cycle)
            1, 2: begin rst = 1; enable = 0; load_bias = 0; end
            3: begin rst = 0; end
            4: begin bias = 100; load_bias = 1; enable = 1; end
            5: begin load_bias = 0; enable = 0; end
            6: begin activation = 10; weight = 5; enable = 1; end
            7: begin enable = 0; end
            8: begin activation = -10; weight = 5; enable = 1; end
            9: begin enable = 0; end
            10: begin activation = -10; weight = -5; enable = 1; end
            11: begin enable = 0; end
            12: begin bias = 0; load_bias = 1; enable = 1; end
            13: begin load_bias = 0; activation = 127; weight = 127; enable = 1; end
            14: begin enable = 0; end
            15: begin bias = 0; load_bias = 1; enable = 1; end
            16: begin load_bias = 0; activation = 8'h80; weight = 8'sd127; enable = 1; end
            17: begin enable = 0; end
            18: begin bias = 0; load_bias = 1; enable = 1; end
            19: begin load_bias = 0; activation = 8'h80; weight = 8'h80; enable = 1; end
            20: begin enable = 0; end
            21: begin activation = 3; weight = 3; end
            default: ;
        endcase
    end

    always @(posedge clk) begin
        case (cycle)
            3: if (accumulator !== 0) begin $display("FAIL reset"); errors = errors + 1; end
            5: if (accumulator !== 100) begin $display("FAIL bias %0d", accumulator); errors = errors + 1; end
            7: begin
                if (product !== 50) begin $display("FAIL 10*5 prod"); errors = errors + 1; end
                if (accumulator !== 150) begin $display("FAIL 10*5 acc"); errors = errors + 1; end
            end
            9: begin
                if (product !== -50) begin $display("FAIL -10*5 prod"); errors = errors + 1; end
                if (accumulator !== 100) begin $display("FAIL -10*5 acc"); errors = errors + 1; end
            end
            11: begin
                if (product !== 50) begin $display("FAIL -10*-5 prod"); errors = errors + 1; end
                if (accumulator !== 150) begin $display("FAIL -10*-5 acc"); errors = errors + 1; end
            end
            14: begin
                if (product !== 16129) begin $display("FAIL 127*127 prod %0d", product); errors = errors + 1; end
                if (accumulator !== 16129) begin $display("FAIL 127*127 acc"); errors = errors + 1; end
            end
            17: begin
                if (product !== -16256) begin $display("FAIL -128*127 prod %0d", product); errors = errors + 1; end
                if (accumulator !== -16256) begin $display("FAIL -128*127 acc"); errors = errors + 1; end
            end
            20: begin
                if (product !== 16384) begin $display("FAIL -128*-128 prod %0d", product); errors = errors + 1; end
                if (accumulator !== 16384) begin $display("FAIL -128*-128 acc"); errors = errors + 1; end
            end
            22: begin
                if (accumulator !== 16384) begin $display("FAIL hold"); errors = errors + 1; end
                if (errors != 0) begin $display("FAIL tb_int8_mac"); $fatal(1); end
                $display("PASS tb_int8_mac");
                $finish;
            end
            default: ;
        endcase
    end
endmodule
