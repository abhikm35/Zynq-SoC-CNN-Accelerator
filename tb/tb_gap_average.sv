// tb_gap_average.sv
// Unit test gap_average divide-by-64 (ties away from zero) + saturate.
// Expected values from vectors/gap/gap_average_cases.mem

`timescale 1ns / 1ps

module tb_gap_average (
    input logic clk
);
    logic signed [31:0] sum_value;
    logic signed [31:0] rounding_adjustment;
    logic signed [31:0] adjusted_sum;
    logic signed [31:0] shifted_or_divided_value;
    logic signed [7:0] saturated_value;
    logic signed [7:0] final_output;

    gap_average dut (
        .sum_value(sum_value),
        .rounding_adjustment(rounding_adjustment),
        .adjusted_sum(adjusted_sum),
        .shifted_or_divided_value(shifted_or_divided_value),
        .saturated_value(saturated_value),
        .final_output(final_output)
    );

    integer cycle;
    integer errors;
    integer fd;
    integer code;
    reg [31:0] sum_hex;
    reg [7:0] exp_sat_hex;
    reg [7:0] exp_gap_hex; // ignored here (post-requant); kept for mem format
    integer exp_sat;
    integer got;

    initial begin
        cycle = 0;
        errors = 0;
        sum_value = 0;
    end

    always @(negedge clk) begin
        cycle = cycle + 1;
        if (cycle == 1) begin
            fd = $fopen("vectors/gap/gap_average_cases.mem", "r");
            if (fd == 0)
                $fatal(1, "cannot open gap_average_cases.mem");
        end
        if (cycle >= 1 && cycle <= 17) begin
            code = $fscanf(fd, "%h %h %h", sum_hex, exp_sat_hex, exp_gap_hex);
            if (code != 3)
                $fatal(1, "bad mem line at cycle %0d", cycle);
            sum_value = sum_hex;
        end
    end

    always @(posedge clk) begin
        if (cycle >= 1 && cycle <= 17) begin
            exp_sat = $signed(exp_sat_hex);
            got = $signed(final_output);
            if (got !== exp_sat) begin
                $display("FAIL sum=%0d got=%0d expected_sat=%0d adj=%0d shifted=%0d",
                         $signed(sum_value), got, exp_sat,
                         rounding_adjustment, shifted_or_divided_value);
                errors = errors + 1;
                $fatal(1, "gap_average mismatch");
            end else begin
                $display("PASS sum=%0d -> avg=%0d (adj=%0d shifted=%0d)",
                         $signed(sum_value), got,
                         rounding_adjustment, shifted_or_divided_value);
            end
        end
        if (cycle == 18) begin
            $fclose(fd);
            if (errors != 0)
                $fatal(1, "tb_gap_average failures=%0d", errors);
            $display("PASS: tb_gap_average cases=17");
            $finish;
        end
    end

endmodule
