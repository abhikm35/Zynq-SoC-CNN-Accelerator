// tb_global_average_pool_full.sv
// Complete GAP (32 channels) bit-exact vs Python post-requant outputs.

`timescale 1ns / 1ps

module tb_global_average_pool_full (
    input logic clk
);
    localparam int NUM_CH = 32;
    localparam int TOTAL_READS = 2048;

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

    logic signed [7:0] expected [0:NUM_CH-1];
    logic              written  [0:NUM_CH-1];
    integer channel_sum [0:NUM_CH-1];
    integer channel_gap [0:NUM_CH-1];

    integer local_c, phase, i;
    integer reads_seen, writes_seen;
    integer start_cycle, done_cycle;
    integer mismatches, checksum;
    integer read_idx, pending_compare, compare_addr;
    integer total_cycles, avg_cycles;
    integer got, expv;

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
        reads_seen=0; writes_seen=0;
        start_cycle=0; done_cycle=0;
        mismatches=0; checksum=0;
        read_idx=0; pending_compare=0; compare_addr=0;
        for (i=0; i<NUM_CH; i=i+1) begin
            written[i]=1'b0;
            channel_sum[i]=0;
            channel_gap[i]=0;
        end
        $readmemh("vectors/gap/gap_expected.mem", expected);
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
            if (read_idx < NUM_CH) begin
                gap_read_enable = 1;
                gap_read_address = read_idx[4:0];
            end else gap_read_enable = 0;
        end
    end

    always @(posedge clk) begin
        if (pool2_read_enable) begin
            reads_seen = reads_seen + 1;
            if ({1'b0, pool2_read_address} >= 12'd2048)
                $fatal(1, "pool2 read OOB");
        end

        if (gap_write_enable) begin
            writes_seen = writes_seen + 1;
            if ({1'b0, gap_write_address} >= 6'd32)
                $fatal(1, "gap write OOB");
            if (written[gap_write_address])
                $fatal(1, "duplicate write addr=%0d", gap_write_address);
            written[gap_write_address] = 1'b1;
            channel_sum[gap_write_address] = final_sum;
            channel_gap[gap_write_address] = $signed(gap_write_data);
            if (gap_write_data !== expected[gap_write_address]) begin
                $display("MISMATCH addr=%0d ch=%0d exp=%0d(%02h) got=%0d(%02h)",
                         gap_write_address, gap_write_address,
                         $signed(expected[gap_write_address]),
                         expected[gap_write_address],
                         $signed(gap_write_data), gap_write_data);
                mismatches = mismatches + 1;
            end
        end

        if (phase == 1 && gap_done) begin
            done_cycle = local_c;
            total_cycles = done_cycle - start_cycle;
            avg_cycles = total_cycles / NUM_CH;
            if (reads_seen !== TOTAL_READS)
                $fatal(1, "reads=%0d expected %0d", reads_seen, TOTAL_READS);
            if (writes_seen !== NUM_CH)
                $fatal(1, "writes=%0d", writes_seen);
            if (read_count !== TOTAL_READS)
                $fatal(1, "dut read_count=%0d", read_count);
            if (output_count !== NUM_CH)
                $fatal(1, "dut output_count=%0d", output_count);
            for (i=0; i<NUM_CH; i=i+1) begin
                if (!written[i])
                    $fatal(1, "missing write addr=%0d", i);
            end
            phase = 2;
            read_idx = 0;
            pending_compare = 0;
        end

        if (phase == 2) begin
            if (pending_compare) begin
                got = $signed(gap_read_data);
                expv = $signed(expected[compare_addr]);
                checksum = checksum + got;
                if (got !== expv) begin
                    $display("READBACK FAIL addr=%0d exp=%0d got=%0d",
                             compare_addr, expv, got);
                    mismatches = mismatches + 1;
                end
                pending_compare = 0;
                read_idx = read_idx + 1;
            end
            if (gap_read_enable) begin
                pending_compare = 1;
                compare_addr = gap_read_address;
            end
            if (!pending_compare && read_idx >= NUM_CH) begin
                checksum = 0;
                for (i=0; i<NUM_CH; i=i+1)
                    checksum = checksum + channel_gap[i];
                $display("==== GAP full-stage summary ====");
                $display("Expected Pool2 reads  : %0d", TOTAL_READS);
                $display("Actual Pool2 reads    : %0d", reads_seen);
                $display("Expected GAP outputs  : %0d", NUM_CH);
                $display("Actual GAP writes     : %0d", writes_seen);
                $display("Total mismatches      : %0d", mismatches);
                $display("Total cycles          : %0d", total_cycles);
                $display("Avg cycles / channel  : %0d", avg_cycles);
                $display("Output checksum       : %0d", checksum);
                $display("");
                $display("  ch   sum   gap");
                for (i=0; i<NUM_CH; i=i+1)
                    $display("  %2d %5d %5d", i, channel_sum[i], channel_gap[i]);
                if (mismatches !== 0)
                    $fatal(1, "GAP mismatches=%0d", mismatches);
                $display("PASS: tb_global_average_pool_full (%0d outputs bit-exact)", NUM_CH);
                $finish;
            end
        end

        if (local_c > 500000)
            $fatal(1, "timeout");
    end

endmodule
