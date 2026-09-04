// tb_requantize.sv
`timescale 1ns / 1ps

module tb_requantize (
    input logic clk
);
    logic signed [31:0] accumulator;
    logic signed [31:0] multiplier;
    logic [5:0] shift;
    logic signed [7:0] output_zero_point;
    logic signed [63:0] wide_product;
    logic signed [63:0] rounding_offset;
    logic signed [63:0] rounded_product;
    logic signed [63:0] shifted_value;
    logic signed [31:0] zero_point_adjusted;
    logic signed [7:0] saturated_value;

    requantize dut (
        .accumulator(accumulator),
        .multiplier(multiplier),
        .shift(shift),
        .output_zero_point(output_zero_point),
        .wide_product(wide_product),
        .rounding_offset(rounding_offset),
        .rounded_product(rounded_product),
        .shifted_value(shifted_value),
        .zero_point_adjusted(zero_point_adjusted),
        .saturated_value(saturated_value)
    );

    integer cycle;
    integer errors;
    integer fd;
    integer exp_sat;
    reg [31:0] hexv;

    task automatic check(input integer expv, input string tag);
        begin
            if (saturated_value !== expv) begin
                $display("FAIL %s: got %0d expected %0d (acc=%0d mult=%0d sh=%0d)",
                         tag, saturated_value, expv, accumulator, multiplier, shift);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        cycle = 0;
        errors = 0;
        accumulator = 0;
        multiplier = 1;
        shift = 0;
        output_zero_point = 0;
    end

    // Drive on negedge so comb DUT settles before posedge check
    always @(negedge clk) begin
        cycle = cycle + 1;
        case (cycle)
            1: begin accumulator = 0; multiplier = 1; shift = 0; end
            2: begin accumulator = 10; multiplier = 1; shift = 0; end
            3: begin accumulator = -5; multiplier = 1; shift = 0; end
            4: begin accumulator = 3; multiplier = 1; shift = 1; end
            5: begin accumulator = 2; multiplier = 1; shift = 1; end
            6: begin accumulator = 1; multiplier = 1; shift = 1; end
            7: begin accumulator = -3; multiplier = 1; shift = 1; end
            8: begin accumulator = -2; multiplier = 1; shift = 1; end
            9: begin accumulator = 100000; multiplier = 1; shift = 0; end
            10: begin accumulator = -100000; multiplier = 1; shift = 0; end
            11: begin
                fd = $fopen("vectors/conv1_single_output/center_expected_acc.mem", "r");
                if (fd != 0) begin
                    void'($fscanf(fd, "%h", hexv));
                    $fclose(fd);
                    accumulator = hexv;
                    fd = $fopen("vectors/conv1_single_output/center_multiplier.mem", "r");
                    void'($fscanf(fd, "%h", hexv));
                    $fclose(fd);
                    multiplier = hexv;
                    fd = $fopen("vectors/conv1_single_output/center_shift.mem", "r");
                    void'($fscanf(fd, "%h", hexv));
                    $fclose(fd);
                    shift = hexv[5:0];
                end
            end
            default: ;
        endcase
    end

    always @(posedge clk) begin
        case (cycle)
            1: check(0, "zero");
            2: check(10, "pos");
            3: check(-5, "neg");
            4: check(2, "3>>1");
            5: check(1, "2>>1");
            6: check(1, "1>>1");
            7: check(-2, "-3>>1");
            8: check(-1, "-2>>1");
            9: check(127, "sat+");
            10: check(-128, "sat-");
            11: begin
                fd = $fopen("vectors/conv1_single_output/center_expected_requantized.mem", "r");
                if (fd != 0) begin
                    void'($fscanf(fd, "%h", hexv));
                    $fclose(fd);
                    exp_sat = hexv[7:0];
                    if (hexv[7]) exp_sat = exp_sat - 256;
                    check(exp_sat, "center_file");
                end else begin
                    $display("WARN skip center file vector");
                end
                if (errors) begin
                    $display("FAIL tb_requantize errors=%0d", errors);
                    $fatal(1);
                end
                $display("PASS tb_requantize");
                $finish;
            end
            default: ;
        endcase
    end
endmodule
