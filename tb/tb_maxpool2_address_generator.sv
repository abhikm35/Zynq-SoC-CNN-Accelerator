// tb_maxpool2_address_generator.sv
// Cycle-based address checks for MaxPool2 (trained 32-channel model).

`timescale 1ns / 1ps

module tb_maxpool2_address_generator (
    input logic clk
);
    logic [4:0] channel;
    logic [2:0] pool_row;
    logic [2:0] pool_column;
    logic [1:0] window_index;

    logic [3:0]  input_row;
    logic [3:0]  input_column;
    logic [12:0] conv2_read_address;
    logic [10:0] pool2_write_address;

    integer local_c, phase, errors, case_i, wi;
    integer ch, pr, pc, exp_row, exp_col, exp_conv, exp_pool, base_r, base_c;
    integer cases [0:5][0:2];

    maxpool2_address_generator dut (
        .channel             (channel),
        .pool_row            (pool_row),
        .pool_column         (pool_column),
        .window_index        (window_index),
        .input_row           (input_row),
        .input_column        (input_column),
        .conv2_read_address  (conv2_read_address),
        .pool2_write_address (pool2_write_address)
    );

    initial begin
        local_c=0; phase=0; errors=0; case_i=0; wi=0;
        channel=0; pool_row=0; pool_column=0; window_index=0;
        cases[0][0]=0;  cases[0][1]=0; cases[0][2]=0;
        cases[1][0]=0;  cases[1][1]=7; cases[1][2]=7;
        cases[2][0]=3;  cases[2][1]=4; cases[2][2]=5;
        cases[3][0]=8;  cases[3][1]=2; cases[3][2]=6;
        cases[4][0]=15; cases[4][1]=7; cases[4][2]=7;
        cases[5][0]=31; cases[5][1]=7; cases[5][2]=7;
    end

    always @(posedge clk) begin
        local_c = local_c + 1;
        if (phase == 0) begin
            ch = cases[0][0]; pr = cases[0][1]; pc = cases[0][2];
            channel = ch[4:0]; pool_row = pr[2:0]; pool_column = pc[2:0];
            window_index = 2'd0; case_i = 0; wi = 0;
            phase = 1;
        end else if (phase == 1) begin
            ch = cases[case_i][0]; pr = cases[case_i][1]; pc = cases[case_i][2];
            base_r = pr * 2; base_c = pc * 2;
            exp_pool = ch * 64 + pr * 8 + pc;
            case (wi)
                0: begin exp_row = base_r;     exp_col = base_c;     end
                1: begin exp_row = base_r;     exp_col = base_c + 1; end
                2: begin exp_row = base_r + 1; exp_col = base_c;     end
                default: begin exp_row = base_r + 1; exp_col = base_c + 1; end
            endcase
            exp_conv = ch * 256 + exp_row * 16 + exp_col;

            if (input_row !== exp_row[3:0]) begin
                $error("row mismatch"); errors = errors + 1;
            end
            if (input_column !== exp_col[3:0]) begin
                $error("col mismatch"); errors = errors + 1;
            end
            if (conv2_read_address !== exp_conv[12:0]) begin
                $error("conv addr mismatch ch=%0d pr=%0d pc=%0d wi=%0d got=%0d exp=%0d",
                       ch, pr, pc, wi, conv2_read_address, exp_conv);
                errors = errors + 1;
            end
            if (pool2_write_address !== exp_pool[10:0]) begin
                $error("pool addr mismatch"); errors = errors + 1;
            end

            if (wi == 3) begin
                if (case_i == 5) begin
                    if (errors != 0)
                        $fatal(1, "tb_maxpool2_address_generator FAILED errors=%0d", errors);
                    $display("PASS: tb_maxpool2_address_generator");
                    phase = 2;
                end else begin
                    case_i = case_i + 1;
                    wi = 0;
                    ch = cases[case_i][0]; pr = cases[case_i][1]; pc = cases[case_i][2];
                    channel = ch[4:0]; pool_row = pr[2:0]; pool_column = pc[2:0];
                    window_index = 2'd0;
                end
            end else begin
                wi = wi + 1;
                window_index = wi[1:0];
            end
        end else $finish;
    end
endmodule
