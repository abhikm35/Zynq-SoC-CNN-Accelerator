// tb_saturate_int8.sv
`timescale 1ns / 1ps

module tb_saturate_int8 (
    input logic clk
);
    logic signed [31:0] value;
    logic signed [7:0] saturated;
    integer cycle;
    integer errors;

    saturate_int8 dut (.value(value), .saturated(saturated));

    initial begin
        cycle = 0;
        errors = 0;
        value = 0;
    end

    always @(negedge clk) begin
        cycle = cycle + 1;
        case (cycle)
            1: value = 0;
            2: value = 127;
            3: value = 128;
            4: value = -128;
            5: value = -129;
            6: value = 1000;
            default: ;
        endcase
    end

    always @(posedge clk) begin
        case (cycle)
            1: if (saturated !== 0) errors = errors + 1;
            2: if (saturated !== 127) errors = errors + 1;
            3: if (saturated !== 127) errors = errors + 1;
            4: if (saturated !== -128) errors = errors + 1;
            5: if (saturated !== -128) errors = errors + 1;
            6: begin
                if (saturated !== 127) errors = errors + 1;
                if (errors != 0) begin $display("FAIL tb_saturate_int8"); $fatal(1); end
                $display("PASS tb_saturate_int8");
                $finish;
            end
            default: ;
        endcase
    end
endmodule
