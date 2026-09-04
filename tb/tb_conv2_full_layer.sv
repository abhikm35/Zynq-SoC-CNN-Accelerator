// tb_conv2_full_layer.sv
// Complete Conv2 (32 x 16 x 16 = 8192) bit-exact vs Python.
// Trained model: 16->32 (prompt sketches with 8->16 are obsolete).

`timescale 1ns / 1ps

module tb_conv2_full_layer (
    input logic clk
);
    localparam int NUM_OUT = 32;
    localparam int TOTAL = NUM_OUT * 256; // 8192
    localparam int MACS_PER = 144;

    logic rst, start, busy, conv2_done;
    logic [13:0] output_count;
    logic [4:0] current_output_channel;
    logic [3:0] current_output_row, current_output_column;

    logic out_read_enable;
    logic [12:0] out_read_address;
    logic signed [7:0] out_read_data;

    logic output_write_enable;
    logic [12:0] output_write_address;
    logic signed [7:0] output_write_data;

    logic engine_start, engine_busy, engine_done;
    logic [7:0] engine_mac_count;
    logic signed [31:0] engine_final_accumulator;
    logic signed [7:0] engine_requantized_output, engine_relu_output;
    logic [4:0] engine_output_channel, engine_bias_address;
    logic engine_operand_valid, engine_mac_enable, engine_load_bias;
    logic [3:0] engine_ic;
    logic [1:0] engine_kr, engine_kc;
    logic [11:0] engine_act_addr;
    logic [12:0] engine_wgt_addr;
    logic engine_padding;
    logic signed [7:0] engine_act_value, engine_wgt_value;
    logic signed [15:0] engine_product, engine_last_product;
    logic signed [31:0] engine_accumulator;

    logic signed [7:0] expected [0:TOTAL-1];
    logic              written  [0:TOTAL-1];

    integer local_c, phase, i;
    integer starts_seen, dones_seen, writes_seen, macs_seen;
    integer start_cycle, done_cycle, first_write_cycle;
    integer mismatches, checksum, min_v, max_v, zeros, sats;
    integer read_idx, prev_write_addr;
    integer ch, row, col;
    integer ch_mismatch [0:31];
    integer ch_sum [0:31];
    integer ch_min [0:31];
    integer ch_max [0:31];
    integer ch_zeros [0:31];
    integer pending_compare, compare_addr;
    integer total_cycles, avg_cycles, cyc_per_ch;
    integer saw_first, saw_255, saw_256, saw_8191;

    conv2_layer_top dut (
        .clk(clk), .rst(rst), .start(start),
        .busy(busy), .conv2_done(conv2_done), .output_count(output_count),
        .current_output_channel(current_output_channel),
        .current_output_row(current_output_row),
        .current_output_column(current_output_column),
        .out_read_enable(out_read_enable),
        .out_read_address(out_read_address),
        .out_read_data(out_read_data),
        .output_write_enable(output_write_enable),
        .output_write_address(output_write_address),
        .output_write_data(output_write_data),
        .engine_start(engine_start),
        .engine_busy(engine_busy),
        .engine_done(engine_done),
        .engine_mac_count(engine_mac_count),
        .engine_final_accumulator(engine_final_accumulator),
        .engine_requantized_output(engine_requantized_output),
        .engine_relu_output(engine_relu_output),
        .engine_output_channel(engine_output_channel),
        .engine_bias_address(engine_bias_address),
        .engine_operand_valid(engine_operand_valid),
        .engine_mac_enable(engine_mac_enable),
        .engine_load_bias(engine_load_bias),
        .engine_ic(engine_ic), .engine_kr(engine_kr), .engine_kc(engine_kc),
        .engine_act_addr(engine_act_addr), .engine_wgt_addr(engine_wgt_addr),
        .engine_padding(engine_padding),
        .engine_act_value(engine_act_value), .engine_wgt_value(engine_wgt_value),
        .engine_product(engine_product), .engine_accumulator(engine_accumulator),
        .engine_last_product(engine_last_product)
    );

    initial begin
        local_c=0; phase=0; rst=1; start=0;
        out_read_enable=0; out_read_address=0;
        starts_seen=0; dones_seen=0; writes_seen=0; macs_seen=0;
        start_cycle=0; done_cycle=0; first_write_cycle=0;
        mismatches=0; checksum=0; min_v=127; max_v=-128; zeros=0; sats=0;
        read_idx=0; prev_write_addr=-1;
        pending_compare=0; compare_addr=0;
        saw_first=0; saw_255=0; saw_256=0; saw_8191=0;
        for (i=0; i<TOTAL; i=i+1) written[i]=1'b0;
        for (i=0; i<32; i=i+1) begin
            ch_mismatch[i]=0; ch_sum[i]=0;
            ch_min[i]=127; ch_max[i]=-128; ch_zeros[i]=0;
        end
        $readmemh("vectors/conv2/conv2_expected.mem", expected);
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
                out_read_enable = 1;
                out_read_address = read_idx[12:0];
            end else out_read_enable = 0;
        end
    end

    always @(posedge clk) begin
        if (engine_start) starts_seen = starts_seen + 1;
        if (engine_done) begin
            dones_seen = dones_seen + 1;
            if (engine_mac_count !== 8'd144)
                $fatal(1, "engine mac_count=%0d", engine_mac_count);
            macs_seen = macs_seen + MACS_PER;
        end

        if (engine_mac_enable && !engine_load_bias)
            if (engine_wgt_addr >= 13'd4608)
                $fatal(1, "wgt addr OOB");

        if (output_write_enable) begin
            writes_seen = writes_seen + 1;
            if ({1'b0, output_write_address} >= 14'd8192)
                $fatal(1, "write OOB");
            if (written[output_write_address])
                $fatal(1, "duplicate write %0d", output_write_address);
            written[output_write_address] = 1'b1;

            if (!saw_first) begin
                saw_first = 1;
                first_write_cycle = local_c;
                if (output_write_address !== 13'd0)
                    $fatal(1, "first write not addr 0");
                if (current_output_channel !== 5'd0 ||
                    current_output_row !== 4'd0 ||
                    current_output_column !== 4'd0)
                    $fatal(1, "first coords wrong");
            end
            if (output_write_address == 13'd255) saw_255 = 1;
            if (output_write_address == 13'd256) saw_256 = 1;
            if (output_write_address == 13'd8191) saw_8191 = 1;

            if (prev_write_addr >= 0) begin
                if (output_write_address !== prev_write_addr[12:0] + 13'd1)
                    $fatal(1, "non-sequential write %0d -> %0d",
                           prev_write_addr, output_write_address);
            end
            prev_write_addr = output_write_address;

            if (output_write_data !== expected[output_write_address]) begin
                ch = output_write_address / 256;
                row = (output_write_address % 256) / 16;
                col = output_write_address % 16;
                $display("MISMATCH addr=%0d ch=%0d r=%0d c=%0d exp=%0d(%02h) got=%0d(%02h)",
                         output_write_address, ch, row, col,
                         $signed(expected[output_write_address]),
                         expected[output_write_address],
                         $signed(output_write_data), output_write_data);
                mismatches = mismatches + 1;
                ch_mismatch[ch] = ch_mismatch[ch] + 1;
            end else begin
                ch = output_write_address / 256;
                ch_sum[ch] = ch_sum[ch] + $signed(output_write_data);
                if ($signed(output_write_data) < ch_min[ch])
                    ch_min[ch] = $signed(output_write_data);
                if ($signed(output_write_data) > ch_max[ch])
                    ch_max[ch] = $signed(output_write_data);
                if (output_write_data === 8'sd0) ch_zeros[ch] = ch_zeros[ch] + 1;
                checksum = checksum + $signed(output_write_data);
                if ($signed(output_write_data) < min_v)
                    min_v = $signed(output_write_data);
                if ($signed(output_write_data) > max_v)
                    max_v = $signed(output_write_data);
                if (output_write_data === 8'sd0) zeros = zeros + 1;
                if (output_write_data === 8'sd127 || output_write_data === -8'sd128)
                    sats = sats + 1;
            end
        end

        if (phase == 1) begin
            if (conv2_done) begin
                done_cycle = local_c;
                if (output_count !== 14'd8192)
                    $fatal(1, "count=%0d", output_count);
                if (writes_seen != TOTAL)
                    $fatal(1, "writes=%0d", writes_seen);
                if (starts_seen != TOTAL || dones_seen != TOTAL)
                    $fatal(1, "starts=%0d dones=%0d", starts_seen, dones_seen);
                if (macs_seen != TOTAL * MACS_PER)
                    $fatal(1, "macs=%0d", macs_seen);
                if (!saw_255 || !saw_256 || !saw_8191)
                    $fatal(1, "missing landmark addresses");
                if (current_output_channel !== 5'd31 ||
                    current_output_row !== 4'd15 ||
                    current_output_column !== 4'd15)
                    $fatal(1, "done coords wrong");
                for (i=0; i<TOTAL; i=i+1)
                    if (!written[i]) $fatal(1, "unwritten %0d", i);
                read_idx = 0;
                pending_compare = 0;
                phase = 2;
            end
        end else if (phase == 2) begin
            if (pending_compare) begin
                i = compare_addr;
                if (out_read_data !== expected[i]) begin
                    mismatches = mismatches + 1;
                    $display("RAM MISMATCH addr=%0d", i);
                end
                pending_compare = 0;
                if (read_idx >= TOTAL) phase = 3;
            end
            if (phase == 2 && out_read_enable && read_idx < TOTAL) begin
                compare_addr = read_idx;
                pending_compare = 1;
                read_idx = read_idx + 1;
            end
        end else if (phase == 3) begin
            total_cycles = done_cycle - start_cycle;
            avg_cycles = total_cycles / TOTAL;
            cyc_per_ch = total_cycles / NUM_OUT;

            $display("");
            $display("==== Conv2 full-layer summary ====");
            $display("Expected output count : %0d", TOTAL);
            $display("Starts / dones        : %0d / %0d", starts_seen, dones_seen);
            $display("MAC operations        : %0d", macs_seen);
            $display("RAM writes            : %0d", writes_seen);
            $display("Total mismatches      : %0d", mismatches);
            $display("Total cycles          : %0d", total_cycles);
            $display("Avg cycles / output   : %0d", avg_cycles);
            $display("Cycles / channel      : %0d", cyc_per_ch);
            $display("Output checksum       : %0d", checksum);
            $display("Minimum output        : %0d", min_v);
            $display("Maximum output        : %0d", max_v);
            $display("Zero-output count     : %0d", zeros);
            $display("Saturated-output count: %0d", sats);
            $display("");
            $display("Per-channel summary:");
            $display("  ch  outs  mism  checksum  min  max  zeros");
            for (i=0; i<32; i=i+1) begin
                $display("  %2d  %4d  %4d  %8d  %3d  %3d  %5d",
                         i, 256, ch_mismatch[i], ch_sum[i],
                         ch_min[i], ch_max[i], ch_zeros[i]);
            end

            if (mismatches != 0)
                $fatal(1, "tb_conv2_full_layer FAILED mismatches=%0d", mismatches);
            $display("PASS: tb_conv2_full_layer (8192 outputs bit-exact)");
            $finish;
        end
    end
endmodule
