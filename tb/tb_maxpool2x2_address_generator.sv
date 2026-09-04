// tb_maxpool2x2_address_generator.sv
// Unit tests for MaxPool1 address equations (16-channel trained model).
// Cycle-based (no # delays — Verilator ignores STMTDLY).

`timescale 1ns / 1ps

module tb_maxpool2x2_address_generator (
    input logic clk
);
    logic [3:0] channel;
    logic [3:0] pool_row;
    logic [3:0] pool_column;
    logic [1:0] window_index;

    logic [4:0]  input_row;
    logic [4:0]  input_column;
    logic [13:0] conv1_read_address;
    logic [11:0] pool1_write_address;

    integer local_c;
    integer phase;
    integer errors;
    integer case_i;
    integer wi;
    integer exp_row, exp_col, exp_conv, exp_pool;
    integer base_r, base_c;
    integer ch, pr, pc;

    // Cases: {ch, pr, pc}
    integer cases [0:5][0:2];

    maxpool2x2_address_generator dut (
        .channel             (channel),
        .pool_row            (pool_row),
        .pool_column         (pool_column),
        .window_index        (window_index),
        .input_row           (input_row),
        .input_column        (input_column),
        .conv1_read_address  (conv1_read_address),
        .pool1_write_address (pool1_write_address)
    );

    initial begin
        local_c = 0;
        phase = 0;
        errors = 0;
        case_i = 0;
        wi = 0;
        channel = 0;
        pool_row = 0;
        pool_column = 0;
        window_index = 0;

        cases[0][0] = 0;  cases[0][1] = 0;  cases[0][2] = 0;
        cases[1][0] = 0;  cases[1][1] = 15; cases[1][2] = 15;
        cases[2][0] = 3;  cases[2][1] = 7;  cases[2][2] = 9;
        cases[3][0] = 7;  cases[3][1] = 15; cases[3][2] = 15;
        cases[4][0] = 15; cases[4][1] = 15; cases[4][2] = 15;
        cases[5][0] = 3;  cases[5][1] = 5;  cases[5][2] = 7; // brief example
    end

    always @(posedge clk) begin
        local_c = local_c + 1;

        if (phase == 0) begin
            // Drive first stimulus
            ch = cases[0][0]; pr = cases[0][1]; pc = cases[0][2];
            channel = ch[3:0];
            pool_row = pr[3:0];
            pool_column = pc[3:0];
            window_index = 2'd0;
            case_i = 0;
            wi = 0;
            phase = 1;
        end else if (phase == 1) begin
            // Check current, then advance stimulus for next cycle
            ch = cases[case_i][0];
            pr = cases[case_i][1];
            pc = cases[case_i][2];
            base_r = pr * 2;
            base_c = pc * 2;
            exp_pool = ch * 256 + pr * 16 + pc;
            case (wi)
                0: begin exp_row = base_r;     exp_col = base_c;     end
                1: begin exp_row = base_r;     exp_col = base_c + 1; end
                2: begin exp_row = base_r + 1; exp_col = base_c;     end
                default: begin exp_row = base_r + 1; exp_col = base_c + 1; end
            endcase
            exp_conv = ch * 1024 + exp_row * 32 + exp_col;

            if (input_row !== exp_row[4:0]) begin
                $error("row mismatch ch=%0d pr=%0d pc=%0d wi=%0d got=%0d exp=%0d",
                       ch, pr, pc, wi, input_row, exp_row);
                errors = errors + 1;
            end
            if (input_column !== exp_col[4:0]) begin
                $error("col mismatch ch=%0d pr=%0d pc=%0d wi=%0d got=%0d exp=%0d",
                       ch, pr, pc, wi, input_column, exp_col);
                errors = errors + 1;
            end
            if (conv1_read_address !== exp_conv[13:0]) begin
                $error("conv addr mismatch ch=%0d pr=%0d pc=%0d wi=%0d got=%0d exp=%0d",
                       ch, pr, pc, wi, conv1_read_address, exp_conv);
                errors = errors + 1;
            end
            if (pool1_write_address !== exp_pool[11:0]) begin
                $error("pool addr mismatch ch=%0d pr=%0d pc=%0d got=%0d exp=%0d",
                       ch, pr, pc, pool1_write_address, exp_pool);
                errors = errors + 1;
            end

            // Explicit A/B/C/D coords for ch=3,pr=5,pc=7
            if (case_i == 5) begin
                if (wi == 0 && (input_row !== 5'd10 || input_column !== 5'd14))
                    $fatal(1, "example A coords failed");
                if (wi == 1 && (input_row !== 5'd10 || input_column !== 5'd15))
                    $fatal(1, "example B coords failed");
                if (wi == 2 && (input_row !== 5'd11 || input_column !== 5'd14))
                    $fatal(1, "example C coords failed");
                if (wi == 3 && (input_row !== 5'd11 || input_column !== 5'd15))
                    $fatal(1, "example D coords failed");
                if (pool1_write_address !== 12'(3 * 256 + 5 * 16 + 7))
                    $fatal(1, "example pool addr failed");
            end

            if (wi == 3) begin
                if (case_i == 5) begin
                    if (errors != 0)
                        $fatal(1, "tb_maxpool2x2_address_generator FAILED errors=%0d",
                               errors);
                    $display("PASS: tb_maxpool2x2_address_generator");
                    phase = 2;
                end else begin
                    case_i = case_i + 1;
                    wi = 0;
                    ch = cases[case_i][0];
                    pr = cases[case_i][1];
                    pc = cases[case_i][2];
                    channel = ch[3:0];
                    pool_row = pr[3:0];
                    pool_column = pc[3:0];
                    window_index = 2'd0;
                end
            end else begin
                wi = wi + 1;
                window_index = wi[1:0];
            end
        end else begin
            $finish;
        end
    end
endmodule
