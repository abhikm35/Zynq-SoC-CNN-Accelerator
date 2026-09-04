// tb_gap_selected_channels.sv
// Compare selected GAP channels against Python traces.

`timescale 1ns / 1ps

module tb_gap_selected_channels (
    input logic clk
);
    localparam int NUM_CH = 32;
    localparam int NSEL = 6;

    logic rst, start, busy, gap_done;
    logic [11:0] read_count;
    logic [5:0] output_count;
    logic [4:0] current_channel;
    logic [5:0] current_element;
    logic signed [31:0] running_sum, final_sum;
    logic signed [7:0] averaged_int8, gap_int8;

    logic pool2_read_enable;
    logic [10:0] pool2_read_address;
    logic signed [7:0] pool2_read_data;

    logic gap_write_enable;
    logic [4:0] gap_write_address;
    logic signed [7:0] gap_write_data;

    logic gap_read_enable;
    logic [4:0] gap_read_address;
    logic signed [7:0] gap_read_data;

    logic signed [7:0] expected_gap [0:NUM_CH-1];
    logic signed [7:0] expected_raw [0:NUM_CH-1];
    integer expected_sum [0:NUM_CH-1];

    integer local_c, phase, i, sidx;
    integer target_ch;
    integer reads_seen, writes_seen;
    integer mismatches;
    integer pending_read_meta;
    integer pending_addr, pending_elem, pending_ch;
    integer accum_count;
    integer exp_running;
    integer got_sum, got_gap, got_raw;

    // Expected sums from exported summary (filled at init from mem + recomputation via pool2)
    logic signed [7:0] pool2_mem [0:2047];

    global_average_pool_top #(
        .NUM_CHANNELS(32),
        .POOL2_MEM_FILE("vectors/gap/pool2_input.mem")
    ) dut (
        .clk(clk), .rst(rst), .start(start),
        .busy(busy), .gap_done(gap_done),
        .read_count(read_count), .output_count(output_count),
        .current_channel(current_channel),
        .current_element(current_element),
        .running_sum(running_sum),
        .final_sum(final_sum),
        .averaged_int8(averaged_int8),
        .gap_int8(gap_int8),
        .pool2_read_enable(pool2_read_enable),
        .pool2_read_address(pool2_read_address),
        .pool2_read_data(pool2_read_data),
        .gap_write_enable(gap_write_enable),
        .gap_write_address(gap_write_address),
        .gap_write_data(gap_write_data),
        .gap_read_enable(gap_read_enable),
        .gap_read_address(gap_read_address),
        .gap_read_data(gap_read_data)
    );

    initial begin
        local_c=0; phase=0; rst=1; start=0;
        gap_read_enable=0; gap_read_address=0;
        reads_seen=0; writes_seen=0; mismatches=0;
        pending_read_meta=0; pending_addr=0; pending_elem=0; pending_ch=0;
        accum_count=0; exp_running=0; sidx=0; target_ch=0;
        $readmemh("vectors/gap/gap_expected.mem", expected_gap);
        $readmemh("vectors/gap/gap_raw_avg_expected.mem", expected_raw);
        $readmemh("vectors/gap/pool2_input.mem", pool2_mem);
        for (i=0; i<NUM_CH; i=i+1) begin
            expected_sum[i] = 0;
            for (sidx=0; sidx<64; sidx=sidx+1)
                expected_sum[i] = expected_sum[i] + $signed(pool2_mem[i*64 + sidx]);
        end
        sidx = 0;
    end

    always @(negedge clk) begin
        local_c = local_c + 1;
        if (phase == 0) begin
            case (local_c)
                1,2: begin rst=1; start=0; end
                3: begin rst=0; end
                4: begin start=1; end
                5: begin start=0; phase=1; end
                default: ;
            endcase
        end
    end

    always @(posedge clk) begin
        if (pool2_read_enable) begin
            reads_seen = reads_seen + 1;
            pending_read_meta = 1;
            pending_addr = pool2_read_address;
            pending_elem = current_element;
            pending_ch = current_channel;
            if ({1'b0, pool2_read_address} >= 12'd2048)
                $fatal(1, "pool2 addr OOB");
            if (pool2_read_address !== 11'(current_channel * 64 + current_element))
                $fatal(1, "addr mismatch ch=%0d elem=%0d addr=%0d",
                       current_channel, current_element, pool2_read_address);
        end

        // After WAIT, ACCUMULATE uses returned data; check on next cycle after ISSUE+WAIT
        // Track when running_sum updates by observing accumulate via read_count growth
        // and comparing at write time for selected channels.

        if (gap_write_enable) begin
            writes_seen = writes_seen + 1;
            got_sum = final_sum;
            got_raw = $signed(averaged_int8);
            got_gap = $signed(gap_write_data);
            if (gap_write_address !== current_channel)
                $fatal(1, "gap write addr != channel");

            // Always check every channel; report selected ones explicitly
            if (got_sum !== expected_sum[gap_write_address]) begin
                $display("FAIL ch=%0d sum got=%0d exp=%0d",
                         gap_write_address, got_sum, expected_sum[gap_write_address]);
                mismatches = mismatches + 1;
                $fatal(1, "sum mismatch");
            end
            if (got_raw !== $signed(expected_raw[gap_write_address])) begin
                $display("FAIL ch=%0d raw_avg got=%0d exp=%0d",
                         gap_write_address, got_raw, $signed(expected_raw[gap_write_address]));
                mismatches = mismatches + 1;
                $fatal(1, "raw avg mismatch");
            end
            if (got_gap !== $signed(expected_gap[gap_write_address])) begin
                $display("FAIL ch=%0d gap got=%0d exp=%0d",
                         gap_write_address, got_gap, $signed(expected_gap[gap_write_address]));
                mismatches = mismatches + 1;
                $fatal(1, "gap mismatch");
            end

            if (gap_write_address == 0 || gap_write_address == 1 ||
                gap_write_address == 5 || gap_write_address == 10 ||
                gap_write_address == 15 || gap_write_address == 31) begin
                $display("PASS selected ch=%0d sum=%0d raw=%0d gap=%0d",
                         gap_write_address, got_sum, got_raw, got_gap);
            end
        end

        if (gap_done) begin
            if (reads_seen !== 2048)
                $fatal(1, "reads_seen=%0d", reads_seen);
            if (writes_seen !== 32)
                $fatal(1, "writes_seen=%0d", writes_seen);
            if (mismatches !== 0)
                $fatal(1, "mismatches=%0d", mismatches);
            $display("PASS: tb_gap_selected_channels cases=%0d reads=%0d", NSEL, reads_seen);
            $finish;
        end

        if (local_c > 500000)
            $fatal(1, "timeout");
    end

endmodule
