// tb_maxpool1_full.sv
// Complete MaxPool1 (16 x 16 x 16 = 4096) bit-exact vs Python.
// Input: verified Conv1/ReLU1 tensor (16 x 32 x 32).
// No requantization, no extra ReLU — signed INT8 max only.

`timescale 1ns / 1ps

module tb_maxpool1_full (
    input logic clk
);
    localparam int NUM_CH = 16;
    localparam int TOTAL  = NUM_CH * 256; // 4096

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

    logic signed [7:0] expected [0:TOTAL-1];
    logic              written  [0:TOTAL-1];

    integer local_c, phase, i;
    integer reads_seen, writes_seen;
    integer start_cycle, done_cycle, first_write_cycle, last_write_cycle;
    integer mismatches;
    integer checksum, min_v, max_v, zeros;
    integer read_idx;
    integer ch, pr, pc;
    integer ch_mismatch [0:15];
    integer ch_sum [0:15];
    integer ch_min [0:15];
    integer ch_max [0:15];
    integer ch_zeros [0:15];
    integer prev_write_addr;
    integer exp_ch, exp_pr, exp_pc;
    integer total_cycles, avg_cycles;
    integer saw_first_write;
    integer saw_addr_255;
    integer saw_addr_256;
    integer saw_addr_4095;
    integer reads_this_window;
    integer windows_completed;

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

    integer pending_compare;
    integer compare_addr;

    initial begin
        local_c = 0; phase = 0;
        rst = 1; start = 0;
        pool1_read_enable = 0; pool1_read_address = 0;
        reads_seen = 0; writes_seen = 0;
        start_cycle = 0; done_cycle = 0;
        first_write_cycle = 0; last_write_cycle = 0;
        mismatches = 0;
        checksum = 0; min_v = 127; max_v = -128; zeros = 0;
        read_idx = 0; prev_write_addr = -1;
        saw_first_write = 0; saw_addr_255 = 0; saw_addr_256 = 0; saw_addr_4095 = 0;
        reads_this_window = 0; windows_completed = 0;
        pending_compare = 0; compare_addr = 0;
        for (i = 0; i < TOTAL; i = i + 1) written[i] = 1'b0;
        for (i = 0; i < 16; i = i + 1) begin
            ch_mismatch[i] = 0;
            ch_sum[i] = 0;
            ch_min[i] = 127;
            ch_max[i] = -128;
            ch_zeros[i] = 0;
        end
        $readmemh("vectors/pool1/pool1_expected.mem", expected);
    end

    // Issue Pool1 read addresses on negedge (same convention as Conv1 full TB).
    always @(negedge clk) begin
        local_c = local_c + 1;
        if (phase == 0) begin
            case (local_c)
                1, 2: begin rst = 1; start = 0; end
                3: begin rst = 0; start = 0; end
                4: begin start = 1; start_cycle = local_c; end
                5: begin start = 0; phase = 1; end
                default: ;
            endcase
        end else if (phase == 2) begin
            if (read_idx < TOTAL) begin
                pool1_read_enable = 1;
                pool1_read_address = read_idx[11:0];
            end else begin
                pool1_read_enable = 0;
            end
        end
    end

    always @(posedge clk) begin
        if (conv1_read_enable) begin
            reads_seen = reads_seen + 1;
            reads_this_window = reads_this_window + 1;
            if ({1'b0, conv1_read_address} >= 15'd16384)
                $fatal(1, "conv1 read address OOB %0d", conv1_read_address);
        end

        if (pool1_write_enable) begin
            writes_seen = writes_seen + 1;
            windows_completed = windows_completed + 1;
            if (reads_this_window != 4)
                $fatal(1, "write without 4 reads (got %0d) at addr %0d",
                       reads_this_window, pool1_write_address);
            reads_this_window = 0;

            if ({1'b0, pool1_write_address} >= 13'd4096)
                $fatal(1, "pool1 write address OOB %0d", pool1_write_address);
            if (written[pool1_write_address])
                $fatal(1, "duplicate write to pool1 addr %0d", pool1_write_address);
            written[pool1_write_address] = 1'b1;

            if (!saw_first_write) begin
                saw_first_write = 1;
                first_write_cycle = local_c;
                if (pool1_write_address !== 12'd0)
                    $fatal(1, "first write addr != 0 (got %0d)", pool1_write_address);
                if (current_channel !== 4'd0 || current_pool_row !== 4'd0 ||
                    current_pool_column !== 4'd0)
                    $fatal(1, "first write coords wrong");
            end
            last_write_cycle = local_c;

            if (pool1_write_address == 12'd255) saw_addr_255 = 1;
            if (pool1_write_address == 12'd256) saw_addr_256 = 1;
            if (pool1_write_address == 12'd4095) saw_addr_4095 = 1;

            if (prev_write_addr >= 0) begin
                if (pool1_write_address !== prev_write_addr[11:0] + 12'd1)
                    $fatal(1, "non-sequential write: prev=%0d cur=%0d",
                           prev_write_addr, pool1_write_address);
            end
            prev_write_addr = pool1_write_address;

            // Live compare vs expected (authoritative bit-exact check)
            if (pool1_write_data !== expected[pool1_write_address]) begin
                ch = pool1_write_address / 256;
                pr = (pool1_write_address % 256) / 16;
                pc = pool1_write_address % 16;
                $display("MISMATCH addr=%0d ch=%0d row=%0d col=%0d exp=%0d(%02h) got=%0d(%02h)",
                         pool1_write_address, ch, pr, pc,
                         $signed(expected[pool1_write_address]),
                         expected[pool1_write_address],
                         $signed(pool1_write_data),
                         pool1_write_data);
                mismatches = mismatches + 1;
                ch_mismatch[ch] = ch_mismatch[ch] + 1;
            end else begin
                ch = pool1_write_address / 256;
                ch_sum[ch] = ch_sum[ch] + $signed(pool1_write_data);
                if ($signed(pool1_write_data) < ch_min[ch])
                    ch_min[ch] = $signed(pool1_write_data);
                if ($signed(pool1_write_data) > ch_max[ch])
                    ch_max[ch] = $signed(pool1_write_data);
                if (pool1_write_data === 8'sd0)
                    ch_zeros[ch] = ch_zeros[ch] + 1;
                checksum = checksum + $signed(pool1_write_data);
                if ($signed(pool1_write_data) < min_v)
                    min_v = $signed(pool1_write_data);
                if ($signed(pool1_write_data) > max_v)
                    max_v = $signed(pool1_write_data);
                if (pool1_write_data === 8'sd0)
                    zeros = zeros + 1;
            end
        end

        if (phase == 1) begin
            if (pool1_done) begin
                done_cycle = local_c;
                if (output_count !== 13'd4096)
                    $fatal(1, "done with output_count=%0d", output_count);
                if (writes_seen != TOTAL)
                    $fatal(1, "writes_seen=%0d expected %0d", writes_seen, TOTAL);
                if (reads_seen != TOTAL * 4)
                    $fatal(1, "reads_seen=%0d expected %0d", reads_seen, TOTAL * 4);
                if (!saw_addr_255 || !saw_addr_256 || !saw_addr_4095)
                    $fatal(1, "missing landmark addresses 255/256/4095");
                if (current_channel !== 4'd15 || current_pool_row !== 4'd15 ||
                    current_pool_column !== 4'd15)
                    $fatal(1, "done coords wrong ch=%0d r=%0d c=%0d",
                           current_channel, current_pool_row, current_pool_column);
                for (i = 0; i < TOTAL; i = i + 1) begin
                    if (!written[i])
                        $fatal(1, "pool1 address %0d never written", i);
                end
                read_idx = 0;
                pending_compare = 0;
                phase = 2;
            end
        end else if (phase == 2) begin
            // Capture data one cycle after address was issued on negedge.
            if (pending_compare) begin
                i = compare_addr;
                if (pool1_read_data !== expected[i]) begin
                    ch = i / 256;
                    pr = (i % 256) / 16;
                    pc = i % 16;
                    $display("RAM MISMATCH addr=%0d ch=%0d row=%0d col=%0d exp=%0d(%02h) got=%0d(%02h)",
                             i, ch, pr, pc,
                             $signed(expected[i]), expected[i],
                             $signed(pool1_read_data), pool1_read_data);
                    mismatches = mismatches + 1;
                    ch_mismatch[ch] = ch_mismatch[ch] + 1;
                end
                pending_compare = 0;
                if (read_idx >= TOTAL)
                    phase = 3;
            end
            if (phase == 2 && pool1_read_enable && read_idx < TOTAL) begin
                compare_addr = read_idx;
                pending_compare = 1;
                read_idx = read_idx + 1;
            end
        end else if (phase == 3) begin
            total_cycles = done_cycle - start_cycle;
            avg_cycles = total_cycles / TOTAL;

            $display("");
            $display("==== MaxPool1 full-layer summary ====");
            $display("Expected output count : %0d", TOTAL);
            $display("Conv1 reads           : %0d", reads_seen);
            $display("Completed windows     : %0d", windows_completed);
            $display("Pool1 RAM writes      : %0d", writes_seen);
            $display("Total mismatches      : %0d", mismatches);
            $display("Total cycles          : %0d", total_cycles);
            $display("Avg cycles / output   : %0d", avg_cycles);
            $display("Output checksum       : %0d", checksum);
            $display("Minimum output        : %0d", min_v);
            $display("Maximum output        : %0d", max_v);
            $display("Zero-output count     : %0d", zeros);
            $display("");
            $display("Per-channel summary:");
            $display("  ch  outs  mism  checksum  min  max  zeros");
            for (i = 0; i < 16; i = i + 1) begin
                $display("  %2d  %4d  %4d  %8d  %3d  %3d  %5d",
                         i, 256, ch_mismatch[i], ch_sum[i],
                         ch_min[i], ch_max[i], ch_zeros[i]);
            end

            if (mismatches != 0)
                $fatal(1, "tb_maxpool1_full FAILED mismatches=%0d", mismatches);
            $display("PASS: tb_maxpool1_full (4096 outputs bit-exact)");
            $finish;
        end
    end
endmodule
