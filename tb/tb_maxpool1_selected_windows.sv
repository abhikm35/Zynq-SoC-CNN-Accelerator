// tb_maxpool1_selected_windows.sv
// Drive MaxPool1 through selected windows and compare traces vs Python.

`timescale 1ns / 1ps

module tb_maxpool1_selected_windows (
    input logic clk
);
    localparam int TOTAL = 4096;

    logic rst;
    logic start;
    logic busy;
    logic pool1_done;
    logic [12:0] output_count;
    logic [3:0] current_channel;
    logic [3:0] current_pool_row;
    logic [3:0] current_pool_column;
    logic [1:0] current_window_index;

    logic conv1_read_enable;
    logic [13:0] conv1_read_address;
    logic signed [7:0] conv1_read_data;

    logic pool1_write_enable;
    logic [11:0] pool1_write_address;
    logic signed [7:0] pool1_write_data;

    logic signed [7:0] value_a, value_b, value_c, value_d, maximum_value;

    logic pool1_read_enable;
    logic [11:0] pool1_read_address;
    logic signed [7:0] pool1_read_data;

    // Trace expectations (4 entries each)
    logic signed [31:0] t_addrs [0:3];
    logic signed [7:0]  t_vals  [0:3];
    logic signed [7:0]  t_exp   [0:0];
    logic signed [31:0] t_paddr [0:0];

    integer local_c, phase, errors;
    integer reads_seen, writes_seen;
    integer win_reads;
    integer check_pending;
    integer target_ch, target_pr, target_pc;
    integer start_cycle, done_cycle;
    integer case_i;
    integer got_a, got_b, got_c, got_d;

    // Case table: name loaded via $readmemh paths
    // 0: ch0_pr0_pc0, 1: ch0_pr15_pc15, 2: ch3_pr7_pc9,
    // 3: ch5_pr12_pc4, 4: ch7_pr15_pc15, 5: ch15_pr15_pc15

    maxpool1_top #(
        .NUM_CHANNELS(16),
        .CONV1_MEM_FILE("vectors/pool1/conv1_input_for_pool.mem")
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .busy(busy),
        .pool1_done(pool1_done),
        .output_count(output_count),
        .current_channel(current_channel),
        .current_pool_row(current_pool_row),
        .current_pool_column(current_pool_column),
        .current_window_index(current_window_index),
        .conv1_read_enable(conv1_read_enable),
        .conv1_read_address(conv1_read_address),
        .conv1_read_data(conv1_read_data),
        .pool1_write_enable(pool1_write_enable),
        .pool1_write_address(pool1_write_address),
        .pool1_write_data(pool1_write_data),
        .value_a(value_a),
        .value_b(value_b),
        .value_c(value_c),
        .value_d(value_d),
        .maximum_value(maximum_value),
        .pool1_read_enable(pool1_read_enable),
        .pool1_read_address(pool1_read_address),
        .pool1_read_data(pool1_read_data)
    );

    task automatic load_case;
        input integer idx;
        begin
            case (idx)
                0: begin
                    target_ch = 0; target_pr = 0; target_pc = 0;
                    $readmemh("vectors/pool1/selected_window_traces/ch0_pr0_pc0_conv1_addrs.mem", t_addrs);
                    $readmemh("vectors/pool1/selected_window_traces/ch0_pr0_pc0_values.mem", t_vals);
                    $readmemh("vectors/pool1/selected_window_traces/ch0_pr0_pc0_expected.mem", t_exp);
                    $readmemh("vectors/pool1/selected_window_traces/ch0_pr0_pc0_pool_addr.mem", t_paddr);
                end
                1: begin
                    target_ch = 0; target_pr = 15; target_pc = 15;
                    $readmemh("vectors/pool1/selected_window_traces/ch0_pr15_pc15_conv1_addrs.mem", t_addrs);
                    $readmemh("vectors/pool1/selected_window_traces/ch0_pr15_pc15_values.mem", t_vals);
                    $readmemh("vectors/pool1/selected_window_traces/ch0_pr15_pc15_expected.mem", t_exp);
                    $readmemh("vectors/pool1/selected_window_traces/ch0_pr15_pc15_pool_addr.mem", t_paddr);
                end
                2: begin
                    target_ch = 3; target_pr = 7; target_pc = 9;
                    $readmemh("vectors/pool1/selected_window_traces/ch3_pr7_pc9_conv1_addrs.mem", t_addrs);
                    $readmemh("vectors/pool1/selected_window_traces/ch3_pr7_pc9_values.mem", t_vals);
                    $readmemh("vectors/pool1/selected_window_traces/ch3_pr7_pc9_expected.mem", t_exp);
                    $readmemh("vectors/pool1/selected_window_traces/ch3_pr7_pc9_pool_addr.mem", t_paddr);
                end
                3: begin
                    target_ch = 5; target_pr = 12; target_pc = 4;
                    $readmemh("vectors/pool1/selected_window_traces/ch5_pr12_pc4_conv1_addrs.mem", t_addrs);
                    $readmemh("vectors/pool1/selected_window_traces/ch5_pr12_pc4_values.mem", t_vals);
                    $readmemh("vectors/pool1/selected_window_traces/ch5_pr12_pc4_expected.mem", t_exp);
                    $readmemh("vectors/pool1/selected_window_traces/ch5_pr12_pc4_pool_addr.mem", t_paddr);
                end
                4: begin
                    target_ch = 7; target_pr = 15; target_pc = 15;
                    $readmemh("vectors/pool1/selected_window_traces/ch7_pr15_pc15_conv1_addrs.mem", t_addrs);
                    $readmemh("vectors/pool1/selected_window_traces/ch7_pr15_pc15_values.mem", t_vals);
                    $readmemh("vectors/pool1/selected_window_traces/ch7_pr15_pc15_expected.mem", t_exp);
                    $readmemh("vectors/pool1/selected_window_traces/ch7_pr15_pc15_pool_addr.mem", t_paddr);
                end
                default: begin
                    target_ch = 15; target_pr = 15; target_pc = 15;
                    $readmemh("vectors/pool1/selected_window_traces/ch15_pr15_pc15_conv1_addrs.mem", t_addrs);
                    $readmemh("vectors/pool1/selected_window_traces/ch15_pr15_pc15_values.mem", t_vals);
                    $readmemh("vectors/pool1/selected_window_traces/ch15_pr15_pc15_expected.mem", t_exp);
                    $readmemh("vectors/pool1/selected_window_traces/ch15_pr15_pc15_pool_addr.mem", t_paddr);
                end
            endcase
        end
    endtask

    initial begin
        local_c = 0; phase = 0; errors = 0;
        rst = 1; start = 0;
        pool1_read_enable = 0; pool1_read_address = 0;
        reads_seen = 0; writes_seen = 0; win_reads = 0;
        check_pending = 0; case_i = 0;
        start_cycle = 0; done_cycle = 0;
        load_case(0);
    end

    always @(posedge clk) begin
        local_c = local_c + 1;

        if (conv1_read_enable)
            reads_seen = reads_seen + 1;

        // When writing the target window, verify values/addresses
        if (pool1_write_enable &&
            (current_channel == target_ch[3:0]) &&
            (current_pool_row == target_pr[3:0]) &&
            (current_pool_column == target_pc[3:0])) begin
            writes_seen = writes_seen + 1;
            if (pool1_write_address !== t_paddr[0][11:0]) begin
                $error("pool addr mismatch case=%0d got=%0d exp=%0d",
                       case_i, pool1_write_address, t_paddr[0]);
                errors = errors + 1;
            end
            if (pool1_write_data !== t_exp[0]) begin
                $error("pool data mismatch case=%0d got=%0d exp=%0d",
                       case_i, $signed(pool1_write_data), $signed(t_exp[0]));
                errors = errors + 1;
            end
            if (value_a !== t_vals[0] || value_b !== t_vals[1] ||
                value_c !== t_vals[2] || value_d !== t_vals[3]) begin
                $error("window values mismatch case=%0d got=%0d/%0d/%0d/%0d exp=%0d/%0d/%0d/%0d",
                       case_i,
                       $signed(value_a), $signed(value_b),
                       $signed(value_c), $signed(value_d),
                       $signed(t_vals[0]), $signed(t_vals[1]),
                       $signed(t_vals[2]), $signed(t_vals[3]));
                errors = errors + 1;
            end
            if (maximum_value !== t_exp[0]) begin
                $error("maximum mismatch case=%0d got=%0d exp=%0d",
                       case_i, $signed(maximum_value), $signed(t_exp[0]));
                errors = errors + 1;
            end
            check_pending = 1;
        end

        // Track four addresses for the target window while reading
        if (conv1_read_enable &&
            (current_channel == target_ch[3:0]) &&
            (current_pool_row == target_pr[3:0]) &&
            (current_pool_column == target_pc[3:0])) begin
            if (conv1_read_address !== t_addrs[current_window_index][13:0]) begin
                $error("conv1 addr mismatch case=%0d wi=%0d got=%0d exp=%0d",
                       case_i, current_window_index, conv1_read_address,
                       t_addrs[current_window_index]);
                errors = errors + 1;
            end
            win_reads = win_reads + 1;
        end

        case (phase)
            0: begin
                rst = 1;
                if (local_c == 4) begin
                    rst = 0;
                    phase = 1;
                end
            end
            1: begin
                start = 1;
                start_cycle = local_c;
                phase = 2;
            end
            2: begin
                start = 0;
                if (check_pending) begin
                    if (win_reads != 4) begin
                        $error("expected 4 reads for window case=%0d got=%0d",
                               case_i, win_reads);
                        errors = errors + 1;
                    end
                    $display("Selected window case %0d PASS (ch=%0d pr=%0d pc=%0d)",
                             case_i, target_ch, target_pr, target_pc);
                    case_i = case_i + 1;
                    check_pending = 0;
                    win_reads = 0;
                    if (case_i >= 6) begin
                        phase = 3; // wait for done, then finish early checks
                    end else begin
                        load_case(case_i);
                    end
                end
                if (pool1_done) begin
                    done_cycle = local_c;
                    if (case_i < 6)
                        $fatal(1, "done before all selected windows checked");
                    phase = 4;
                end
            end
            3: begin
                if (pool1_done) begin
                    done_cycle = local_c;
                    phase = 4;
                end
            end
            4: begin
                if (errors != 0)
                    $fatal(1, "tb_maxpool1_selected_windows FAILED errors=%0d", errors);
                $display("PASS: tb_maxpool1_selected_windows cases=6 reads=%0d",
                         reads_seen);
                $finish;
            end
            default: $finish;
        endcase
    end
endmodule
