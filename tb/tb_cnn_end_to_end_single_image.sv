// tb_cnn_end_to_end_single_image.sv
// Full CNN: load sample_000, run inference, check all intermediates + class.
// Also covers reset-between-runs and repeated inference.

`timescale 1ns / 1ps

module tb_cnn_end_to_end_single_image (
    input logic clk
);
    localparam int IN_N    = 3072;
    localparam int CONV1_N = 16384;
    localparam int POOL1_N = 4096;
    localparam int CONV2_N = 8192;
    localparam int POOL2_N = 2048;
    localparam int GAP_N   = 32;
    localparam int LOGIT_N = 5;

    logic rst, start, busy, done;
    logic [2:0] predicted_class;
    logic signed [31:0] maximum_logit;
    logic signed [31:0] logit_0, logit_1, logit_2, logit_3, logit_4;
    logic [63:0] cycle_count;
    logic [3:0] state_id;
    logic [2:0] stage_id;

    logic dbg_conv1_done, dbg_pool1_done, dbg_conv2_done, dbg_pool2_done;
    logic dbg_gap_done, dbg_fc_done, dbg_argmax_done;
    logic [14:0] dbg_conv1_output_count;
    logic [12:0] dbg_pool1_output_count;
    logic [13:0] dbg_conv2_output_count;
    logic [11:0] dbg_pool2_output_count;
    logic [5:0]  dbg_gap_output_count;
    logic [7:0]  dbg_fc_mac_count;
    logic [63:0] dbg_conv1_cycles, dbg_pool1_cycles, dbg_conv2_cycles;
    logic [63:0] dbg_pool2_cycles, dbg_gap_cycles, dbg_fc_cycles, dbg_argmax_cycles;

    logic input_we;
    logic [11:0] input_waddr;
    logic signed [7:0] input_wdata;

    logic conv1_re, pool1_re, conv2_re, pool2_re, gap_re, logit_re;
    logic [13:0] conv1_raddr;
    logic [11:0] pool1_raddr;
    logic [12:0] conv2_raddr;
    logic [10:0] pool2_raddr;
    logic [4:0]  gap_raddr;
    logic [2:0]  logit_raddr;
    logic signed [7:0] conv1_rdata, pool1_rdata, conv2_rdata, pool2_rdata, gap_rdata;
    logic signed [31:0] logit_rdata;

    logic signed [7:0]  exp_input  [0:IN_N-1];
    logic signed [7:0]  exp_conv1  [0:CONV1_N-1];
    logic signed [7:0]  exp_pool1  [0:POOL1_N-1];
    logic signed [7:0]  exp_conv2  [0:CONV2_N-1];
    logic signed [7:0]  exp_pool2  [0:POOL2_N-1];
    logic signed [7:0]  exp_gap    [0:GAP_N-1];
    logic signed [31:0] exp_logits [0:LOGIT_N-1];

    integer local_c, phase, i, rd_i, rd_wait;
    integer start_cycle, done_cycle;
    integer exp_class, exp_max;
    integer seen_c1, seen_p1, seen_c2, seen_p2, seen_gap, seen_fc, seen_am;
    integer pass_num;
    integer mismatches;
    integer mid_reset_done;

    cnn_accelerator_top dut (
        .clk(clk), .rst(rst), .start(start),
        .busy(busy), .done(done),
        .predicted_class(predicted_class),
        .maximum_logit(maximum_logit),
        .logit_0(logit_0), .logit_1(logit_1), .logit_2(logit_2),
        .logit_3(logit_3), .logit_4(logit_4),
        .cycle_count(cycle_count),
        .state_id(state_id), .stage_id(stage_id),
        .dbg_conv1_done(dbg_conv1_done),
        .dbg_pool1_done(dbg_pool1_done),
        .dbg_conv2_done(dbg_conv2_done),
        .dbg_pool2_done(dbg_pool2_done),
        .dbg_gap_done(dbg_gap_done),
        .dbg_fc_done(dbg_fc_done),
        .dbg_argmax_done(dbg_argmax_done),
        .dbg_conv1_output_count(dbg_conv1_output_count),
        .dbg_pool1_output_count(dbg_pool1_output_count),
        .dbg_conv2_output_count(dbg_conv2_output_count),
        .dbg_pool2_output_count(dbg_pool2_output_count),
        .dbg_gap_output_count(dbg_gap_output_count),
        .dbg_fc_mac_count(dbg_fc_mac_count),
        .dbg_conv1_cycles(dbg_conv1_cycles),
        .dbg_pool1_cycles(dbg_pool1_cycles),
        .dbg_conv2_cycles(dbg_conv2_cycles),
        .dbg_pool2_cycles(dbg_pool2_cycles),
        .dbg_gap_cycles(dbg_gap_cycles),
        .dbg_fc_cycles(dbg_fc_cycles),
        .dbg_argmax_cycles(dbg_argmax_cycles),
        .input_write_enable(input_we),
        .input_write_address(input_waddr),
        .input_write_data(input_wdata),
        .conv1_read_enable(conv1_re),
        .conv1_read_address(conv1_raddr),
        .conv1_read_data(conv1_rdata),
        .pool1_read_enable(pool1_re),
        .pool1_read_address(pool1_raddr),
        .pool1_read_data(pool1_rdata),
        .conv2_read_enable(conv2_re),
        .conv2_read_address(conv2_raddr),
        .conv2_read_data(conv2_rdata),
        .pool2_read_enable(pool2_re),
        .pool2_read_address(pool2_raddr),
        .pool2_read_data(pool2_rdata),
        .gap_read_enable(gap_re),
        .gap_read_address(gap_raddr),
        .gap_read_data(gap_rdata),
        .logit_read_enable(logit_re),
        .logit_read_address(logit_raddr),
        .logit_read_data(logit_rdata)
    );

    task automatic clear_hosts;
        begin
            start = 0;
            input_we = 0; input_waddr = 0; input_wdata = 0;
            conv1_re = 0; pool1_re = 0; conv2_re = 0;
            pool2_re = 0; gap_re = 0; logit_re = 0;
            conv1_raddr = 0; pool1_raddr = 0; conv2_raddr = 0;
            pool2_raddr = 0; gap_raddr = 0; logit_raddr = 0;
        end
    endtask

    initial begin
        local_c = 0; phase = 0; i = 0; rd_i = 0; rd_wait = 0;
        start_cycle = 0; done_cycle = 0;
        seen_c1 = 0; seen_p1 = 0; seen_c2 = 0; seen_p2 = 0;
        seen_gap = 0; seen_fc = 0; seen_am = 0;
        pass_num = 0; mismatches = 0; mid_reset_done = 0;
        rst = 1;
        clear_hosts();
        $readmemh("vectors/end_to_end/input_image.mem", exp_input);
        $readmemh("vectors/end_to_end/conv1_expected.mem", exp_conv1);
        $readmemh("vectors/end_to_end/pool1_expected.mem", exp_pool1);
        $readmemh("vectors/end_to_end/conv2_expected.mem", exp_conv2);
        $readmemh("vectors/end_to_end/pool2_expected.mem", exp_pool2);
        $readmemh("vectors/end_to_end/gap_expected.mem", exp_gap);
        $readmemh("vectors/end_to_end/fc_logits_expected.mem", exp_logits);
        exp_max = exp_logits[0];
        exp_class = 0;
        for (i = 1; i < LOGIT_N; i = i + 1) begin
            if ($signed(exp_logits[i]) > exp_max) begin
                exp_max = $signed(exp_logits[i]);
                exp_class = i;
            end
        end
    end

    // phases:
    // 0 reset+load input
    // 1 pulse start
    // 2 wait done (track stage dones)
    // 3 compare tensors (rd_wait handshake)
    // 4 check class/logits/counts
    // 5 second inference start (repeat)
    // 6 wait done2
    // 7 compare finals only
    // 8 mid-reset test: start then reset then reload+run
    // 9 wait done3 + finals
    // 10 PASS

    always @(negedge clk) begin
        local_c = local_c + 1;
        if (phase == 0) begin
            if (local_c <= 3) begin
                rst = 1; clear_hosts();
            end else if (local_c == 4) begin
                rst = 0;
                i = 0;
            end else if (local_c >= 5) begin
                if (i < IN_N) begin
                    input_we = 1;
                    input_waddr = i[11:0];
                    input_wdata = exp_input[i];
                    i = i + 1;
                end else begin
                    input_we = 0;
                    phase = 1;
                    local_c = 0;
                end
            end
        end else if (phase == 1) begin
            if (local_c == 1) begin
                start = 1;
                start_cycle = local_c;
                seen_c1 = 0; seen_p1 = 0; seen_c2 = 0; seen_p2 = 0;
                seen_gap = 0; seen_fc = 0; seen_am = 0;
            end else begin
                start = 0;
                phase = 2;
            end
        end else if (phase == 2) begin
            start = 0;
            // optional mid-run start ignored
            if (local_c == 10)
                start = 1;
            else
                start = 0;
        end else if (phase == 3) begin
            // drive read requests
            conv1_re = 0; pool1_re = 0; conv2_re = 0;
            pool2_re = 0; gap_re = 0; logit_re = 0;
            if (rd_wait == 0) begin
                if (rd_i < CONV1_N) begin
                    conv1_re = 1; conv1_raddr = rd_i[13:0];
                end else if (rd_i < CONV1_N + POOL1_N) begin
                    pool1_re = 1; pool1_raddr = (rd_i - CONV1_N);
                end else if (rd_i < CONV1_N + POOL1_N + CONV2_N) begin
                    conv2_re = 1; conv2_raddr = (rd_i - CONV1_N - POOL1_N);
                end else if (rd_i < CONV1_N + POOL1_N + CONV2_N + POOL2_N) begin
                    pool2_re = 1; pool2_raddr = (rd_i - CONV1_N - POOL1_N - CONV2_N);
                end else if (rd_i < CONV1_N + POOL1_N + CONV2_N + POOL2_N + GAP_N) begin
                    gap_re = 1; gap_raddr = (rd_i - CONV1_N - POOL1_N - CONV2_N - POOL2_N);
                end else if (rd_i < CONV1_N + POOL1_N + CONV2_N + POOL2_N + GAP_N + LOGIT_N) begin
                    logit_re = 1;
                    logit_raddr = (rd_i - CONV1_N - POOL1_N - CONV2_N - POOL2_N - GAP_N);
                end
                rd_wait = 1;
            end else begin
                rd_wait = 2; // capture on posedge
            end
        end else if (phase == 5) begin
            if (local_c == 1) start = 1;
            else begin start = 0; phase = 6; end
        end else if (phase == 8) begin
            if (local_c == 1) start = 1;
            else if (local_c == 2) start = 0;
            else if (local_c == 50) begin
                rst = 1; clear_hosts();
            end else if (local_c == 53) begin
                rst = 0; i = 0;
            end else if (local_c >= 54 && rst == 0) begin
                if (i < IN_N) begin
                    input_we = 1;
                    input_waddr = i[11:0];
                    input_wdata = exp_input[i];
                    i = i + 1;
                end else begin
                    input_we = 0;
                    start = 1;
                    phase = 9;
                    local_c = 0;
                end
            end
        end else if (phase == 9) begin
            if (local_c == 1) start = 0;
        end
    end

    always @(posedge clk) begin
        if (phase == 2) begin
            if (dbg_conv1_done) seen_c1 = seen_c1 + 1;
            if (dbg_pool1_done) seen_p1 = seen_p1 + 1;
            if (dbg_conv2_done) seen_c2 = seen_c2 + 1;
            if (dbg_pool2_done) seen_p2 = seen_p2 + 1;
            if (dbg_gap_done)   seen_gap = seen_gap + 1;
            if (dbg_fc_done)    seen_fc = seen_fc + 1;
            if (dbg_argmax_done) seen_am = seen_am + 1;
            if (done) begin
                done_cycle = local_c;
                if (seen_c1 !== 1) $fatal(1, "conv1_done count=%0d", seen_c1);
                if (seen_p1 !== 1) $fatal(1, "pool1_done count=%0d", seen_p1);
                if (seen_c2 !== 1) $fatal(1, "conv2_done count=%0d", seen_c2);
                if (seen_p2 !== 1) $fatal(1, "pool2_done count=%0d", seen_p2);
                if (seen_gap !== 1) $fatal(1, "gap_done count=%0d", seen_gap);
                if (seen_fc !== 1) $fatal(1, "fc_done count=%0d", seen_fc);
                if (seen_am !== 1) $fatal(1, "argmax_done count=%0d", seen_am);
                if (dbg_conv1_output_count !== 15'd16384)
                    $fatal(1, "conv1 count %0d", dbg_conv1_output_count);
                if (dbg_pool1_output_count !== 13'd4096)
                    $fatal(1, "pool1 count %0d", dbg_pool1_output_count);
                if (dbg_conv2_output_count !== 14'd8192)
                    $fatal(1, "conv2 count %0d", dbg_conv2_output_count);
                if (dbg_pool2_output_count !== 12'd2048)
                    $fatal(1, "pool2 count %0d", dbg_pool2_output_count);
                if (dbg_gap_output_count !== 6'd32)
                    $fatal(1, "gap count %0d", dbg_gap_output_count);
                if (dbg_fc_mac_count !== 8'd160)
                    $fatal(1, "fc macs %0d", dbg_fc_mac_count);
                phase = 3;
                rd_i = 0;
                rd_wait = 0;
                mismatches = 0;
            end
        end

        if (phase == 3 && rd_wait == 2) begin
            if (rd_i < CONV1_N) begin
                if (conv1_rdata !== exp_conv1[rd_i]) begin
                    $display("FAIL Conv1 addr=%0d exp=%0d got=%0d",
                             rd_i, exp_conv1[rd_i], conv1_rdata);
                    mismatches = mismatches + 1;
                    if (mismatches > 8) $fatal(1, "too many Conv1 mismatches");
                end
            end else if (rd_i < CONV1_N + POOL1_N) begin
                if (pool1_rdata !== exp_pool1[rd_i - CONV1_N]) begin
                    $display("FAIL Pool1 addr=%0d", rd_i - CONV1_N);
                    $fatal(1, "pool1 mismatch");
                end
            end else if (rd_i < CONV1_N + POOL1_N + CONV2_N) begin
                if (conv2_rdata !== exp_conv2[rd_i - CONV1_N - POOL1_N]) begin
                    $display("FAIL Conv2 addr=%0d", rd_i - CONV1_N - POOL1_N);
                    $fatal(1, "conv2 mismatch");
                end
            end else if (rd_i < CONV1_N + POOL1_N + CONV2_N + POOL2_N) begin
                if (pool2_rdata !== exp_pool2[rd_i - CONV1_N - POOL1_N - CONV2_N]) begin
                    $display("FAIL Pool2 addr=%0d",
                             rd_i - CONV1_N - POOL1_N - CONV2_N);
                    $fatal(1, "pool2 mismatch");
                end
            end else if (rd_i < CONV1_N + POOL1_N + CONV2_N + POOL2_N + GAP_N) begin
                if (gap_rdata !== exp_gap[rd_i - CONV1_N - POOL1_N - CONV2_N - POOL2_N]) begin
                    $display("FAIL GAP ch=%0d",
                             rd_i - CONV1_N - POOL1_N - CONV2_N - POOL2_N);
                    $fatal(1, "gap mismatch");
                end
            end else begin
                if (logit_rdata !== exp_logits[rd_i - CONV1_N - POOL1_N - CONV2_N - POOL2_N - GAP_N]) begin
                    $display("FAIL logit[%0d]",
                             rd_i - CONV1_N - POOL1_N - CONV2_N - POOL2_N - GAP_N);
                    $fatal(1, "logit mismatch");
                end
            end
            rd_i = rd_i + 1;
            rd_wait = 0;
            if (rd_i >= CONV1_N + POOL1_N + CONV2_N + POOL2_N + GAP_N + LOGIT_N) begin
                phase = 4;
            end
        end

        if (phase == 4) begin
            if (mismatches !== 0) $fatal(1, "tensor mismatches=%0d", mismatches);
            if (predicted_class !== exp_class[2:0])
                $fatal(1, "class got=%0d exp=%0d", predicted_class, exp_class);
            if ($signed(maximum_logit) !== exp_max)
                $fatal(1, "max logit got=%0d exp=%0d", $signed(maximum_logit), exp_max);
            if (logit_0 !== exp_logits[0]) $fatal(1, "logit0");
            if (logit_1 !== exp_logits[1]) $fatal(1, "logit1");
            if (logit_2 !== exp_logits[2]) $fatal(1, "logit2");
            if (logit_3 !== exp_logits[3]) $fatal(1, "logit3");
            if (logit_4 !== exp_logits[4]) $fatal(1, "logit4");
            $display("==== E2E single-image pass 1 ====");
            $display("predicted_class=%0d maximum_logit=%0d", predicted_class, $signed(maximum_logit));
            $display("cycle_count=%0d", cycle_count);
            $display("stage cycles: c1=%0d p1=%0d c2=%0d p2=%0d gap=%0d fc=%0d am=%0d",
                     dbg_conv1_cycles, dbg_pool1_cycles, dbg_conv2_cycles,
                     dbg_pool2_cycles, dbg_gap_cycles, dbg_fc_cycles, dbg_argmax_cycles);
            pass_num = 1;
            phase = 5;
            local_c = 0;
        end

        if (phase == 6 && done) begin
            if (predicted_class !== exp_class[2:0]) $fatal(1, "repeat class");
            if ($signed(maximum_logit) !== exp_max) $fatal(1, "repeat max");
            $display("==== E2E repeated inference PASS (cycles=%0d) ====", cycle_count);
            phase = 8;
            local_c = 0;
            mid_reset_done = 0;
        end

        if (phase == 9 && done) begin
            if (predicted_class !== exp_class[2:0]) $fatal(1, "post-reset class");
            if ($signed(maximum_logit) !== exp_max) $fatal(1, "post-reset max");
            $display("==== E2E reset-during-inference recovery PASS ====");
            $display("PASS: tb_cnn_end_to_end_single_image");
            $finish;
        end

        if (local_c > 50000000)
            $fatal(1, "timeout phase=%0d", phase);
    end

endmodule
