// tb_max4_int8.sv
// Signed INT8 max-of-four — catches accidental unsigned compares.
// Cycle-based (no # delays).

`timescale 1ns / 1ps

module tb_max4_int8 (
    input logic clk
);
    logic signed [7:0] value_a;
    logic signed [7:0] value_b;
    logic signed [7:0] value_c;
    logic signed [7:0] value_d;
    logic signed [7:0] maximum;

    integer local_c;
    integer phase;
    integer errors;
    integer case_i;

    integer cases_a [0:13];
    integer cases_b [0:13];
    integer cases_c [0:13];
    integer cases_d [0:13];
    integer cases_e [0:13];

    max4_int8 dut (
        .value_a (value_a),
        .value_b (value_b),
        .value_c (value_c),
        .value_d (value_d),
        .maximum (maximum)
    );

    initial begin
        local_c = 0;
        phase = 0;
        errors = 0;
        case_i = 0;
        value_a = 0; value_b = 0; value_c = 0; value_d = 0;

        // a,b,c,d,expected
        cases_a[0] = 12;   cases_b[0] = 4;    cases_c[0] = 7;    cases_d[0] = 19;   cases_e[0] = 19;
        cases_a[1] = -10;  cases_b[1] = -4;   cases_c[1] = -7;   cases_d[1] = -20;  cases_e[1] = -4;
        cases_a[2] = -128; cases_b[2] = -127; cases_c[2] = -1;   cases_d[2] = 0;    cases_e[2] = 0;
        cases_a[3] = 127;  cases_b[3] = 126;  cases_c[3] = -128; cases_d[3] = -1;   cases_e[3] = 127;
        cases_a[4] = 5;    cases_b[4] = 5;    cases_c[4] = 5;    cases_d[4] = 5;    cases_e[4] = 5;
        cases_a[5] = 0;    cases_b[5] = -1;   cases_c[5] = -2;   cases_d[5] = -3;   cases_e[5] = 0;
        cases_a[6] = -128; cases_b[6] = 0;    cases_c[6] = 127;  cases_d[6] = -1;   cases_e[6] = 127;
        cases_a[7] = 12;   cases_b[7] = -4;   cases_c[7] = 7;    cases_d[7] = 19;   cases_e[7] = 19;
        cases_a[8] = -1;   cases_b[8] = -20;  cases_c[8] = -5;   cases_d[8] = -3;   cases_e[8] = -1;
        cases_a[9] = -128; cases_b[9] = -128; cases_c[9] = -128; cases_d[9] = -128; cases_e[9] = -128;
        cases_a[10] = 127; cases_b[10] = 127; cases_c[10] = 127; cases_d[10] = 127; cases_e[10] = 127;
        cases_a[11] = -128;cases_b[11] = 127; cases_c[11] = -128;cases_d[11] = 127; cases_e[11] = 127;
        // Unsigned traps
        cases_a[12] = 1;   cases_b[12] = -1;  cases_c[12] = -2;  cases_d[12] = -3;  cases_e[12] = 1;
        cases_a[13] = -5;  cases_b[13] = 2;   cases_c[13] = -100;cases_d[13] = -50; cases_e[13] = 2;
    end

    always @(posedge clk) begin
        local_c = local_c + 1;
        if (phase == 0) begin
            value_a = cases_a[0][7:0];
            value_b = cases_b[0][7:0];
            value_c = cases_c[0][7:0];
            value_d = cases_d[0][7:0];
            case_i = 0;
            phase = 1;
        end else if (phase == 1) begin
            if (maximum !== cases_e[case_i][7:0]) begin
                $error("max4 FAIL case=%0d a=%0d b=%0d c=%0d d=%0d got=%0d exp=%0d",
                       case_i,
                       $signed(value_a), $signed(value_b),
                       $signed(value_c), $signed(value_d),
                       $signed(maximum), cases_e[case_i]);
                errors = errors + 1;
            end
            if (case_i == 13) begin
                if (errors != 0)
                    $fatal(1, "tb_max4_int8 FAILED errors=%0d", errors);
                $display("PASS: tb_max4_int8 (signed comparisons verified)");
                phase = 2;
            end else begin
                case_i = case_i + 1;
                value_a = cases_a[case_i][7:0];
                value_b = cases_b[case_i][7:0];
                value_c = cases_c[case_i][7:0];
                value_d = cases_d[case_i][7:0];
            end
        end else begin
            $finish;
        end
    end
endmodule
