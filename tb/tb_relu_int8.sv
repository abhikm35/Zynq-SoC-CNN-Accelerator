// tb_relu_int8.sv — combinational checks on first evals via cycle counter
`timescale 1ns / 1ps

module tb_relu_int8 (
    input logic clk
);
    logic signed [7:0] in_value;
    logic signed [7:0] zero_point;
    logic signed [7:0] out_value;
    integer cycle;
    integer errors;

    relu_int8 dut (
        .in_value(in_value),
        .zero_point(zero_point),
        .out_value(out_value)
    );

    initial begin
        cycle = 0;
        errors = 0;
        in_value = 0;
        zero_point = 0;
    end

    always @(posedge clk) begin
        cycle = cycle + 1;
        case (cycle)
            1: begin
                in_value = 5; zero_point = 0;
            end
            2: begin
                if (out_value !== 8'sd5) begin $display("FAIL pos"); errors = errors + 1; end
                in_value = -3; zero_point = 0;
            end
            3: begin
                if (out_value !== 8'sd0) begin $display("FAIL neg"); errors = errors + 1; end
                in_value = -128; zero_point = 0;
            end
            4: begin
                if (out_value !== 8'sd0) begin $display("FAIL min"); errors = errors + 1; end
                in_value = 127; zero_point = 0;
            end
            5: begin
                if (out_value !== 8'sd127) begin $display("FAIL max"); errors = errors + 1; end
                if (errors) $fatal(1);
                $display("PASS tb_relu_int8");
                $finish;
            end
            default: ;
        endcase
    end
endmodule
