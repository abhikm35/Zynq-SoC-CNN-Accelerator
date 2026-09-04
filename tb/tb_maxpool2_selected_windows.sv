// tb_maxpool2_selected_windows.sv
// Drive MaxPool2 through selected windows vs Python traces.

`timescale 1ns / 1ps

module tb_maxpool2_selected_windows (
    input logic clk
);
    logic rst, start, busy, pool2_done;
    logic [11:0] output_count;
    logic [13:0] read_count;
    logic [4:0] current_channel;
    logic [2:0] current_pool_row, current_pool_column;
    logic [1:0] current_window_index;

    logic conv2_read_enable;
    logic [12:0] conv2_read_address;
    logic signed [7:0] conv2_read_data;

    logic pool2_write_enable;
    logic [10:0] pool2_write_address;
    logic signed [7:0] pool2_write_data;

    logic signed [7:0] value_a, value_b, value_c, value_d, maximum_value;

    logic pool2_read_enable;
    logic [10:0] pool2_read_address;
    logic signed [7:0] pool2_read_data;

    logic signed [31:0] t_addrs [0:3];
    logic signed [7:0]  t_vals  [0:3];
    logic signed [7:0]  t_exp   [0:0];
    logic signed [31:0] t_paddr [0:0];

    integer local_c, phase, errors, reads_seen, win_reads;
    integer check_pending, target_ch, target_pr, target_pc, case_i;

    maxpool2_top #(
        .NUM_CHANNELS(32),
        .CONV2_MEM_FILE("vectors/pool2/conv2_input.mem")
    ) dut (
        .clk(clk), .rst(rst), .start(start),
        .busy(busy), .pool2_done(pool2_done),
        .output_count(output_count), .read_count(read_count),
        .current_channel(current_channel),
        .current_pool_row(current_pool_row),
        .current_pool_column(current_pool_column),
        .current_window_index(current_window_index),
        .conv2_read_enable(conv2_read_enable),
        .conv2_read_address(conv2_read_address),
        .conv2_read_data(conv2_read_data),
        .pool2_write_enable(pool2_write_enable),
        .pool2_write_address(pool2_write_address),
        .pool2_write_data(pool2_write_data),
        .value_a(value_a), .value_b(value_b),
        .value_c(value_c), .value_d(value_d),
        .maximum_value(maximum_value),
        .pool2_read_enable(pool2_read_enable),
        .pool2_read_address(pool2_read_address),
        .pool2_read_data(pool2_read_data)
    );

    task automatic load_case;
        input integer idx;
        begin
            case (idx)
                0: begin
                    target_ch=0; target_pr=0; target_pc=0;
                    $readmemh("vectors/pool2/selected_window_traces/ch0_pr0_pc0_conv2_addrs.mem", t_addrs);
                    $readmemh("vectors/pool2/selected_window_traces/ch0_pr0_pc0_values.mem", t_vals);
                    $readmemh("vectors/pool2/selected_window_traces/ch0_pr0_pc0_expected.mem", t_exp);
                    $readmemh("vectors/pool2/selected_window_traces/ch0_pr0_pc0_pool_addr.mem", t_paddr);
                end
                1: begin
                    target_ch=0; target_pr=7; target_pc=7;
                    $readmemh("vectors/pool2/selected_window_traces/ch0_pr7_pc7_conv2_addrs.mem", t_addrs);
                    $readmemh("vectors/pool2/selected_window_traces/ch0_pr7_pc7_values.mem", t_vals);
                    $readmemh("vectors/pool2/selected_window_traces/ch0_pr7_pc7_expected.mem", t_exp);
                    $readmemh("vectors/pool2/selected_window_traces/ch0_pr7_pc7_pool_addr.mem", t_paddr);
                end
                2: begin
                    target_ch=3; target_pr=4; target_pc=5;
                    $readmemh("vectors/pool2/selected_window_traces/ch3_pr4_pc5_conv2_addrs.mem", t_addrs);
                    $readmemh("vectors/pool2/selected_window_traces/ch3_pr4_pc5_values.mem", t_vals);
                    $readmemh("vectors/pool2/selected_window_traces/ch3_pr4_pc5_expected.mem", t_exp);
                    $readmemh("vectors/pool2/selected_window_traces/ch3_pr4_pc5_pool_addr.mem", t_paddr);
                end
                3: begin
                    target_ch=8; target_pr=2; target_pc=6;
                    $readmemh("vectors/pool2/selected_window_traces/ch8_pr2_pc6_conv2_addrs.mem", t_addrs);
                    $readmemh("vectors/pool2/selected_window_traces/ch8_pr2_pc6_values.mem", t_vals);
                    $readmemh("vectors/pool2/selected_window_traces/ch8_pr2_pc6_expected.mem", t_exp);
                    $readmemh("vectors/pool2/selected_window_traces/ch8_pr2_pc6_pool_addr.mem", t_paddr);
                end
                4: begin
                    target_ch=15; target_pr=7; target_pc=7;
                    $readmemh("vectors/pool2/selected_window_traces/ch15_pr7_pc7_conv2_addrs.mem", t_addrs);
                    $readmemh("vectors/pool2/selected_window_traces/ch15_pr7_pc7_values.mem", t_vals);
                    $readmemh("vectors/pool2/selected_window_traces/ch15_pr7_pc7_expected.mem", t_exp);
                    $readmemh("vectors/pool2/selected_window_traces/ch15_pr7_pc7_pool_addr.mem", t_paddr);
                end
                default: begin
                    target_ch=31; target_pr=7; target_pc=7;
                    $readmemh("vectors/pool2/selected_window_traces/ch31_pr7_pc7_conv2_addrs.mem", t_addrs);
                    $readmemh("vectors/pool2/selected_window_traces/ch31_pr7_pc7_values.mem", t_vals);
                    $readmemh("vectors/pool2/selected_window_traces/ch31_pr7_pc7_expected.mem", t_exp);
                    $readmemh("vectors/pool2/selected_window_traces/ch31_pr7_pc7_pool_addr.mem", t_paddr);
                end
            endcase
        end
    endtask

    initial begin
        local_c=0; phase=0; errors=0; rst=1; start=0;
        pool2_read_enable=0; pool2_read_address=0;
        reads_seen=0; win_reads=0; check_pending=0; case_i=0;
        load_case(0);
    end

    always @(posedge clk) begin
        local_c = local_c + 1;
        if (conv2_read_enable) reads_seen = reads_seen + 1;

        if (pool2_write_enable &&
            (current_channel == target_ch[4:0]) &&
            (current_pool_row == target_pr[2:0]) &&
            (current_pool_column == target_pc[2:0])) begin
            if (pool2_write_address !== t_paddr[0][10:0]) begin
                $error("pool addr mismatch case=%0d", case_i);
                errors = errors + 1;
            end
            if (pool2_write_data !== t_exp[0]) begin
                $error("pool data mismatch case=%0d got=%0d exp=%0d",
                       case_i, $signed(pool2_write_data), $signed(t_exp[0]));
                errors = errors + 1;
            end
            if (value_a !== t_vals[0] || value_b !== t_vals[1] ||
                value_c !== t_vals[2] || value_d !== t_vals[3]) begin
                $error("window values mismatch case=%0d", case_i);
                errors = errors + 1;
            end
            if (maximum_value !== t_exp[0]) begin
                $error("maximum mismatch case=%0d", case_i);
                errors = errors + 1;
            end
            check_pending = 1;
        end

        if (conv2_read_enable &&
            (current_channel == target_ch[4:0]) &&
            (current_pool_row == target_pr[2:0]) &&
            (current_pool_column == target_pc[2:0])) begin
            if (conv2_read_address !== t_addrs[current_window_index][12:0]) begin
                $error("conv2 addr mismatch case=%0d wi=%0d got=%0d exp=%0d",
                       case_i, current_window_index, conv2_read_address,
                       t_addrs[current_window_index]);
                errors = errors + 1;
            end
            win_reads = win_reads + 1;
        end

        case (phase)
            0: begin
                rst = 1;
                if (local_c == 4) begin rst = 0; phase = 1; end
            end
            1: begin start = 1; phase = 2; end
            2: begin
                start = 0;
                if (check_pending) begin
                    if (win_reads != 4) begin
                        $error("expected 4 reads case=%0d got=%0d", case_i, win_reads);
                        errors = errors + 1;
                    end
                    $display("Selected window case %0d PASS (ch=%0d pr=%0d pc=%0d)",
                             case_i, target_ch, target_pr, target_pc);
                    case_i = case_i + 1;
                    check_pending = 0; win_reads = 0;
                    if (case_i >= 6) phase = 3;
                    else load_case(case_i);
                end
                if (pool2_done) begin
                    if (case_i < 6) $fatal(1, "done before all windows");
                    phase = 4;
                end
            end
            3: if (pool2_done) phase = 4;
            4: begin
                if (errors != 0)
                    $fatal(1, "tb_maxpool2_selected_windows FAILED errors=%0d", errors);
                $display("PASS: tb_maxpool2_selected_windows cases=6 reads=%0d", reads_seen);
                $finish;
            end
            default: $finish;
        endcase
    end
endmodule
