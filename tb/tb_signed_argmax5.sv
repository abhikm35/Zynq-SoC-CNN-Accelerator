// tb_signed_argmax5.sv
// Directed + tie-breaking tests for combinational signed_argmax5.
// Drive on negedge, check on posedge.

`timescale 1ns / 1ps

module tb_signed_argmax5 (
    input logic clk
);
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

    integer local_c, phase, errors, case_i;
    integer exp_class, exp_max;
    integer fd, code;
    reg [31:0] hexv;
    reg [3:0] hexc;

    localparam int NUM_HARDCODED = 12;
    localparam int NUM_DIRECTED = 16;

    // Hardcoded prompt + extreme cases: logits[5], exp_max, exp_class
    integer hc [0:11][0:6];

    initial begin
        local_c=0; phase=0; errors=0; case_i=0;
        logit_0=0; logit_1=0; logit_2=0; logit_3=0; logit_4=0;
        // case1..8 + extremes
        hc[0][0]=500; hc[0][1]=20; hc[0][2]=10; hc[0][3]=0; hc[0][4]=-1; hc[0][5]=500; hc[0][6]=0;
        hc[1][0]=-10; hc[1][1]=200; hc[1][2]=30; hc[1][3]=40; hc[1][4]=50; hc[1][5]=200; hc[1][6]=1;
        hc[2][0]=-10; hc[2][1]=-20; hc[2][2]=300; hc[2][3]=-40; hc[2][4]=-50; hc[2][5]=300; hc[2][6]=2;
        hc[3][0]=-100; hc[3][1]=-200; hc[3][2]=-300; hc[3][3]=-1; hc[3][4]=-400; hc[3][5]=-1; hc[3][6]=3;
        hc[4][0]=-100; hc[4][1]=-200; hc[4][2]=-300; hc[4][3]=-400; hc[4][4]=0; hc[4][5]=0; hc[4][6]=4;
        hc[5][0]=-10; hc[5][1]=-5; hc[5][2]=-20; hc[5][3]=-8; hc[5][4]=-30; hc[5][5]=-5; hc[5][6]=1;
        hc[6][0]=100; hc[6][1]=500; hc[6][2]=20; hc[6][3]=500; hc[6][4]=-10; hc[6][5]=500; hc[6][6]=1;
        hc[7][0]=7; hc[7][1]=7; hc[7][2]=7; hc[7][3]=7; hc[7][4]=7; hc[7][5]=7; hc[7][6]=0;
        hc[8][0]=-2147483648; hc[8][1]=-1; hc[8][2]=-2; hc[8][3]=-3; hc[8][4]=-4; hc[8][5]=-1; hc[8][6]=1;
        hc[9][0]=0; hc[9][1]=1; hc[9][2]=2147483647; hc[9][3]=2; hc[9][4]=3; hc[9][5]=2147483647; hc[9][6]=2;
        hc[10][0]=9; hc[10][1]=1; hc[10][2]=2; hc[10][3]=3; hc[10][4]=9; hc[10][5]=9; hc[10][6]=0;
        hc[11][0]=1; hc[11][1]=1; hc[11][2]=5; hc[11][3]=5; hc[11][4]=5; hc[11][5]=5; hc[11][6]=2;
        fd = $fopen("vectors/argmax/test_cases/directed_cases.mem", "r");
        if (fd == 0)
            $fatal(1, "cannot open directed_cases.mem");
    end

    always @(negedge clk) begin
        local_c = local_c + 1;
        if (phase == 0 && case_i < NUM_HARDCODED) begin
            logit_0 = hc[case_i][0];
            logit_1 = hc[case_i][1];
            logit_2 = hc[case_i][2];
            logit_3 = hc[case_i][3];
            logit_4 = hc[case_i][4];
            exp_max = hc[case_i][5];
            exp_class = hc[case_i][6];
        end else if (phase == 1 && case_i < NUM_DIRECTED) begin
            code = $fscanf(fd, "%h", hexv); logit_0 = hexv;
            code = $fscanf(fd, "%h", hexv); logit_1 = hexv;
            code = $fscanf(fd, "%h", hexv); logit_2 = hexv;
            code = $fscanf(fd, "%h", hexv); logit_3 = hexv;
            code = $fscanf(fd, "%h", hexv); logit_4 = hexv;
            code = $fscanf(fd, "%h", hexv); exp_max = $signed(hexv);
            code = $fscanf(fd, "%h", hexc); exp_class = hexc;
        end
    end

    always @(posedge clk) begin
        if (local_c >= 1) begin
            if (phase == 0 && case_i < NUM_HARDCODED) begin
                if (($signed(maximum_logit) !== exp_max) ||
                    (predicted_class !== exp_class[2:0])) begin
                    $display("FAIL hc[%0d] got max=%0d class=%0d exp %0d/%0d",
                             case_i, $signed(maximum_logit), predicted_class,
                             exp_max, exp_class);
                    $fatal(1, "hardcoded mismatch");
                end
                $display("PASS hc[%0d] max=%0d class=%0d", case_i, exp_max, exp_class);
                case_i = case_i + 1;
                if (case_i == NUM_HARDCODED) begin
                    phase = 1;
                    case_i = 0;
                end
            end else if (phase == 1 && case_i < NUM_DIRECTED) begin
                if (($signed(maximum_logit) !== exp_max) ||
                    (predicted_class !== exp_class[2:0])) begin
                    $display("FAIL directed[%0d] got max=%0d class=%0d exp %0d/%0d",
                             case_i, $signed(maximum_logit), predicted_class,
                             exp_max, exp_class);
                    $fatal(1, "directed mismatch");
                end
                case_i = case_i + 1;
                if (case_i == NUM_DIRECTED) begin
                    $fclose(fd);
                    $display("PASS directed_cases.mem count=%0d", NUM_DIRECTED);
                    $display("PASS: tb_signed_argmax5");
                    $finish;
                end
            end
        end
    end

endmodule
