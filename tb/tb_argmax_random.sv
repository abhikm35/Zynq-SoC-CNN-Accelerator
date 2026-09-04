// tb_argmax_random.sv
// 512 randomized five-logit vectors vs Python (combinational signed_argmax5).

`timescale 1ns / 1ps

module tb_argmax_random (
    input logic clk
);
    localparam int NUM_RANDOM = 512;

    logic signed [31:0] logit_0, logit_1, logit_2, logit_3, logit_4;
    logic [2:0] predicted_class;
    logic signed [31:0] maximum_logit;

    signed_argmax5 dut (
        .logit_0(logit_0),
        .logit_1(logit_1),
        .logit_2(logit_2),
        .logit_3(logit_3),
        .logit_4(logit_4),
        .predicted_class(predicted_class),
        .maximum_logit(maximum_logit)
    );

    integer cycle, case_i, errors;
    integer fd, code;
    integer exp_max, exp_class;
    reg [31:0] hexv;
    reg [3:0] hexc;

    initial begin
        cycle = 0; case_i = 0; errors = 0;
        logit_0 = 0; logit_1 = 0; logit_2 = 0; logit_3 = 0; logit_4 = 0;
        fd = $fopen("vectors/argmax/test_cases/random_cases.mem", "r");
        if (fd == 0)
            $fatal(1, "cannot open random_cases.mem");
    end

    always @(negedge clk) begin
        cycle = cycle + 1;
        if (case_i < NUM_RANDOM) begin
            code = $fscanf(fd, "%h", hexv); if (code != 1) $fatal(1, "eof early");
            logit_0 = hexv;
            code = $fscanf(fd, "%h", hexv); logit_1 = hexv;
            code = $fscanf(fd, "%h", hexv); logit_2 = hexv;
            code = $fscanf(fd, "%h", hexv); logit_3 = hexv;
            code = $fscanf(fd, "%h", hexv); logit_4 = hexv;
            code = $fscanf(fd, "%h", hexv); exp_max = $signed(hexv);
            code = $fscanf(fd, "%h", hexc); exp_class = hexc;
        end
    end

    always @(posedge clk) begin
        if (cycle >= 1 && case_i < NUM_RANDOM) begin
            if (($signed(maximum_logit) !== exp_max) ||
                (predicted_class !== exp_class[2:0])) begin
                $display("FAIL random[%0d] logits=%0d %0d %0d %0d %0d got max=%0d class=%0d exp %0d/%0d",
                         case_i,
                         $signed(logit_0), $signed(logit_1), $signed(logit_2),
                         $signed(logit_3), $signed(logit_4),
                         $signed(maximum_logit), predicted_class, exp_max, exp_class);
                errors = errors + 1;
                $fatal(1, "random mismatch");
            end
            case_i = case_i + 1;
            if (case_i == NUM_RANDOM) begin
                $fclose(fd);
                if (errors != 0)
                    $fatal(1, "errors=%0d", errors);
                $display("PASS: tb_argmax_random cases=%0d", NUM_RANDOM);
                $finish;
            end
        end
    end

endmodule
