// tb_conv2_address_generator.sv
// Cycle-based address checks vs Python equations (trained 16->32).

`timescale 1ns / 1ps

module tb_conv2_address_generator (
    input logic clk
);
    logic [4:0] output_channel;
    logic [3:0] output_row;
    logic [3:0] output_column;
    logic [3:0] input_channel;
    logic [1:0] kernel_row;
    logic [1:0] kernel_column;

    logic signed [5:0] input_row;
    logic signed [5:0] input_column;
    logic              padding;
    logic [11:0]       pool1_read_address;
    logic [12:0]       conv2_weight_address;
    logic [4:0]        conv2_bias_address;
    logic [12:0]       conv2_output_address;

    integer local_c, phase, errors;
    integer case_i, mac_i;
    integer oc, orow, ocol, ic, kr, kc;
    integer exp_row, exp_col, exp_pad, exp_act, exp_wgt, exp_bias, exp_out;

    // Cases: oc, row, col
    integer cases [0:6][0:2];

    conv2_address_generator dut (
        .output_channel       (output_channel),
        .output_row           (output_row),
        .output_column        (output_column),
        .input_channel        (input_channel),
        .kernel_row           (kernel_row),
        .kernel_column        (kernel_column),
        .input_row            (input_row),
        .input_column         (input_column),
        .padding              (padding),
        .pool1_read_address   (pool1_read_address),
        .conv2_weight_address (conv2_weight_address),
        .conv2_bias_address   (conv2_bias_address),
        .conv2_output_address (conv2_output_address)
    );

    initial begin
        local_c = 0; phase = 0; errors = 0;
        case_i = 0; mac_i = 0;
        output_channel = 0; output_row = 0; output_column = 0;
        input_channel = 0; kernel_row = 0; kernel_column = 0;

        cases[0][0]=0;  cases[0][1]=0;  cases[0][2]=0;
        cases[1][0]=0;  cases[1][1]=7;  cases[1][2]=7;
        cases[2][0]=5;  cases[2][1]=7;  cases[2][2]=10;
        cases[3][0]=15; cases[3][1]=15; cases[3][2]=15;
        cases[4][0]=31; cases[4][1]=15; cases[4][2]=15;
        cases[5][0]=0;  cases[5][1]=0;  cases[5][2]=15;
        cases[6][0]=0;  cases[6][1]=15; cases[6][2]=0;
    end

    always @(posedge clk) begin
        local_c = local_c + 1;
        if (phase == 0) begin
            oc = cases[0][0]; orow = cases[0][1]; ocol = cases[0][2];
            output_channel = oc[4:0];
            output_row = orow[3:0];
            output_column = ocol[3:0];
            input_channel = 4'd0;
            kernel_row = 2'd0;
            kernel_column = 2'd0;
            case_i = 0; mac_i = 0;
            phase = 1;
        end else if (phase == 1) begin
            oc = cases[case_i][0];
            orow = cases[case_i][1];
            ocol = cases[case_i][2];
            ic = mac_i / 9;
            kr = (mac_i % 9) / 3;
            kc = mac_i % 3;

            exp_row = orow + kr - 1;
            exp_col = ocol + kc - 1;
            exp_pad = (exp_row < 0 || exp_col < 0 || exp_row >= 16 || exp_col >= 16);
            exp_wgt = oc * 144 + ic * 9 + kr * 3 + kc;
            exp_bias = oc;
            exp_out = oc * 256 + orow * 16 + ocol;
            if (exp_pad)
                exp_act = 0;
            else
                exp_act = ic * 256 + exp_row * 16 + exp_col;

            if (input_row !== exp_row[5:0] || input_column !== exp_col[5:0]) begin
                $error("coord mismatch case=%0d mac=%0d", case_i, mac_i);
                errors = errors + 1;
            end
            if (padding !== exp_pad[0]) begin
                $error("pad mismatch case=%0d mac=%0d got=%0d exp=%0d",
                       case_i, mac_i, padding, exp_pad);
                errors = errors + 1;
            end
            if (!exp_pad && pool1_read_address !== exp_act[11:0]) begin
                $error("act addr mismatch case=%0d mac=%0d got=%0d exp=%0d",
                       case_i, mac_i, pool1_read_address, exp_act);
                errors = errors + 1;
            end
            if (conv2_weight_address !== exp_wgt[12:0]) begin
                $error("wgt addr mismatch case=%0d mac=%0d got=%0d exp=%0d",
                       case_i, mac_i, conv2_weight_address, exp_wgt);
                errors = errors + 1;
            end
            if (conv2_bias_address !== exp_bias[4:0]) begin
                $error("bias addr mismatch");
                errors = errors + 1;
            end
            if (conv2_output_address !== exp_out[12:0]) begin
                $error("out addr mismatch");
                errors = errors + 1;
            end

            if (mac_i == 143) begin
                if (case_i == 6) begin
                    if (errors != 0)
                        $fatal(1, "tb_conv2_address_generator FAILED errors=%0d", errors);
                    $display("PASS: tb_conv2_address_generator");
                    phase = 2;
                end else begin
                    case_i = case_i + 1;
                    mac_i = 0;
                    oc = cases[case_i][0];
                    orow = cases[case_i][1];
                    ocol = cases[case_i][2];
                    output_channel = oc[4:0];
                    output_row = orow[3:0];
                    output_column = ocol[3:0];
                    input_channel = 4'd0;
                    kernel_row = 2'd0;
                    kernel_column = 2'd0;
                end
            end else begin
                mac_i = mac_i + 1;
                ic = mac_i / 9;
                kr = (mac_i % 9) / 3;
                kc = mac_i % 3;
                input_channel = ic[3:0];
                kernel_row = kr[1:0];
                kernel_column = kc[1:0];
            end
        end else begin
            $finish;
        end
    end
endmodule
