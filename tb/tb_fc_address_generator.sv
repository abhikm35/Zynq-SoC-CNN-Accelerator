// tb_fc_address_generator.sv
// Drive on negedge, check on posedge (Verilator-friendly).

`timescale 1ns / 1ps

module tb_fc_address_generator (
    input logic clk
);
    logic [2:0] class_index;
    logic [4:0] input_index;
    logic [4:0] gap_read_address;
    logic [7:0] fc_weight_address;
    logic [2:0] fc_bias_address;
    logic [2:0] logit_write_address;

    integer local_c, phase, errors;
    integer c, i, exp_gap, exp_wgt, exp_bias, exp_logit;
    integer cases_c [0:4];
    integer cases_i [0:4];

    fc_address_generator dut (
        .class_index(class_index),
        .input_index(input_index),
        .gap_read_address(gap_read_address),
        .fc_weight_address(fc_weight_address),
        .fc_bias_address(fc_bias_address),
        .logit_write_address(logit_write_address)
    );

    initial begin
        local_c=0; phase=0; errors=0;
        class_index=0; input_index=0;
        cases_c[0]=0; cases_i[0]=0;
        cases_c[1]=0; cases_i[1]=31;
        cases_c[2]=1; cases_i[2]=0;
        cases_c[3]=2; cases_i[3]=7;
        cases_c[4]=4; cases_i[4]=31;
    end

    always @(negedge clk) begin
        local_c = local_c + 1;
        if (phase < 5) begin
            c = cases_c[phase];
            i = cases_i[phase];
            class_index = c[2:0];
            input_index = i[4:0];
        end
    end

    always @(posedge clk) begin
        if (local_c >= 1 && phase < 5) begin
            c = cases_c[phase];
            i = cases_i[phase];
            exp_gap = i;
            exp_wgt = c * 32 + i;
            exp_bias = c;
            exp_logit = c;
            if (gap_read_address !== exp_gap[4:0] ||
                fc_weight_address !== exp_wgt[7:0] ||
                fc_bias_address !== exp_bias[2:0] ||
                logit_write_address !== exp_logit[2:0]) begin
                $display("FAIL c=%0d i=%0d gap=%0d/%0d wgt=%0d/%0d bias=%0d/%0d logit=%0d/%0d",
                         c, i, gap_read_address, exp_gap, fc_weight_address, exp_wgt,
                         fc_bias_address, exp_bias, logit_write_address, exp_logit);
                errors = errors + 1;
                $fatal(1, "fc address mismatch");
            end
            $display("PASS c=%0d i=%0d gap=%0d wgt=%0d bias=%0d logit=%0d",
                     c, i, gap_read_address, fc_weight_address,
                     fc_bias_address, logit_write_address);
            phase = phase + 1;
            if (phase == 5) begin
                if (errors != 0)
                    $fatal(1, "errors=%0d", errors);
                $display("PASS: tb_fc_address_generator");
                $finish;
            end
        end
    end

endmodule
