// tb_maxpool2_full.sv
// Complete MaxPool2 (32 x 8 x 8 = 2048) bit-exact vs Python.
// Input: verified Conv2/ReLU2 tensor (32 x 16 x 16).

`timescale 1ns / 1ps

module tb_maxpool2_full (
    input logic clk
);
    localparam int NUM_CH = 32;
    localparam int TOTAL  = NUM_CH * 64; // 2048

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

    logic signed [7:0] expected [0:TOTAL-1];
    logic              written  [0:TOTAL-1];

    integer local_c, phase, i;
    integer reads_seen, writes_seen, windows_completed, reads_this_window;
    integer start_cycle, done_cycle;
    integer mismatches, checksum, min_v, max_v, zeros;
    integer read_idx, prev_write_addr, ch, pr, pc;
    integer ch_mismatch [0:31];
    integer ch_sum [0:31];
    integer ch_min [0:31];
    integer ch_max [0:31];
    integer ch_zeros [0:31];
    integer pending_compare, compare_addr;
    integer total_cycles, avg_cycles;
    integer saw_first, saw_63, saw_64, saw_2047;

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

    initial begin
        local_c=0; phase=0; rst=1; start=0;
        pool2_read_enable=0; pool2_read_address=0;
        reads_seen=0; writes_seen=0; windows_completed=0; reads_this_window=0;
        start_cycle=0; done_cycle=0;
        mismatches=0; checksum=0; min_v=127; max_v=-128; zeros=0;
        read_idx=0; prev_write_addr=-1;
        pending_compare=0; compare_addr=0;
        saw_first=0; saw_63=0; saw_64=0; saw_2047=0;
        for (i=0; i<TOTAL; i=i+1) written[i]=1'b0;
        for (i=0; i<32; i=i+1) begin
            ch_mismatch[i]=0; ch_sum[i]=0;
            ch_min[i]=127; ch_max[i]=-128; ch_zeros[i]=0;
        end
        $readmemh("vectors/pool2/pool2_expected.mem", expected);
    end

    always @(negedge clk) begin
        local_c = local_c + 1;
        if (phase == 0) begin
            case (local_c)
                1,2: begin rst=1; start=0; end
                3: begin rst=0; end
                4: begin start=1; start_cycle=local_c; end
                5: begin start=0; phase=1; end
                default: ;
            endcase
        end else if (phase == 2) begin
            if (read_idx < TOTAL) begin
                pool2_read_enable = 1;
                pool2_read_address = read_idx[10:0];
            end else pool2_read_enable = 0;
        end
    end

    always @(posedge clk) begin
        if (conv2_read_enable) begin
            reads_seen = reads_seen + 1;
            reads_this_window = reads_this_window + 1;
            if ({1'b0, conv2_read_address} >= 14'd8192)
                $fatal(1, "conv2 read OOB");
        end

        if (pool2_write_enable) begin
            writes_seen = writes_seen + 1;
            windows_completed = windows_completed + 1;
            if (reads_this_window != 4)
                $fatal(1, "write without 4 reads got=%0d", reads_this_window);
            reads_this_window = 0;

            if ({1'b0, pool2_write_address} >= 12'd2048)
                $fatal(1, "pool2 write OOB");
            if (written[pool2_write_address])
                $fatal(1, "duplicate write %0d", pool2_write_address);
            written[pool2_write_address] = 1'b1;

            if (!saw_first) begin
                saw_first = 1;
                if (pool2_write_address !== 11'd0)
                    $fatal(1, "first write not 0");
                if (current_channel !== 5'd0 || current_pool_row !== 3'd0 ||
                    current_pool_column !== 3'd0)
                    $fatal(1, "first coords wrong");
            end
            if (pool2_write_address == 11'd63) saw_63 = 1;
            if (pool2_write_address == 11'd64) saw_64 = 1;
            if (pool2_write_address == 11'd2047) saw_2047 = 1;

            if (prev_write_addr >= 0) begin
                if (pool2_write_address !== prev_write_addr[10:0] + 11'd1)
                    $fatal(1, "non-sequential write");
            end
            prev_write_addr = pool2_write_address;

            if (pool2_write_data !== expected[pool2_write_address]) begin
                ch = pool2_write_address / 64;
                pr = (pool2_write_address % 64) / 8;
                pc = pool2_write_address % 8;
                $display("MISMATCH addr=%0d ch=%0d row=%0d col=%0d exp=%0d(%02h) got=%0d(%02h)",
                         pool2_write_address, ch, pr, pc,
                         $signed(expected[pool2_write_address]),
                         expected[pool2_write_address],
                         $signed(pool2_write_data), pool2_write_data);
                mismatches = mismatches + 1;
                ch_mismatch[ch] = ch_mismatch[ch] + 1;
            end else begin
                ch = pool2_write_address / 64;
                ch_sum[ch] = ch_sum[ch] + $signed(pool2_write_data);
                if ($signed(pool2_write_data) < ch_min[ch])
                    ch_min[ch] = $signed(pool2_write_data);
                if ($signed(pool2_write_data) > ch_max[ch])
                    ch_max[ch] = $signed(pool2_write_data);
                if (pool2_write_data === 8'sd0) ch_zeros[ch] = ch_zeros[ch] + 1;
                checksum = checksum + $signed(pool2_write_data);
                if ($signed(pool2_write_data) < min_v)
                    min_v = $signed(pool2_write_data);
                if ($signed(pool2_write_data) > max_v)
                    max_v = $signed(pool2_write_data);
                if (pool2_write_data === 8'sd0) zeros = zeros + 1;
            end
        end

        if (phase == 1) begin
            if (pool2_done) begin
                done_cycle = local_c;
                if (output_count !== 12'd2048)
                    $fatal(1, "count=%0d", output_count);
                if (read_count !== 14'd8192)
                    $fatal(1, "read_count=%0d", read_count);
                if (writes_seen != TOTAL)
                    $fatal(1, "writes=%0d", writes_seen);
                if (reads_seen != TOTAL * 4)
                    $fatal(1, "reads=%0d", reads_seen);
                if (!saw_63 || !saw_64 || !saw_2047)
                    $fatal(1, "missing landmark addresses");
                if (current_channel !== 5'd31 || current_pool_row !== 3'd7 ||
                    current_pool_column !== 3'd7)
                    $fatal(1, "done coords wrong");
                for (i=0; i<TOTAL; i=i+1)
                    if (!written[i]) $fatal(1, "unwritten %0d", i);
                read_idx = 0; pending_compare = 0; phase = 2;
            end
        end else if (phase == 2) begin
            if (pending_compare) begin
                i = compare_addr;
                if (pool2_read_data !== expected[i]) begin
                    mismatches = mismatches + 1;
                    $display("RAM MISMATCH addr=%0d", i);
                end
                pending_compare = 0;
                if (read_idx >= TOTAL) phase = 3;
            end
            if (phase == 2 && pool2_read_enable && read_idx < TOTAL) begin
                compare_addr = read_idx;
                pending_compare = 1;
                read_idx = read_idx + 1;
            end
        end else if (phase == 3) begin
            total_cycles = done_cycle - start_cycle;
            avg_cycles = total_cycles / TOTAL;

            $display("");
            $display("==== MaxPool2 full-layer summary ====");
            $display("Expected output count : %0d", TOTAL);
            $display("Conv2 reads           : %0d", reads_seen);
            $display("Completed windows     : %0d", windows_completed);
            $display("Pool2 RAM writes      : %0d", writes_seen);
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
            for (i=0; i<32; i=i+1) begin
                $display("  %2d  %4d  %4d  %8d  %3d  %3d  %5d",
                         i, 64, ch_mismatch[i], ch_sum[i],
                         ch_min[i], ch_max[i], ch_zeros[i]);
            end

            if (mismatches != 0)
                $fatal(1, "tb_maxpool2_full FAILED mismatches=%0d", mismatches);
            $display("PASS: tb_maxpool2_full (2048 outputs bit-exact)");
            $finish;
        end
    end
endmodule
