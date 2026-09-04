// tb_cnn_shared_compute_end_to_end.sv
// Shared-compute activation RAM end-to-end regression.
//
// Because RAM A/B are reused, intermediate tensors are verified by capturing
// every activation write during each stage (bit-exact vs Python vectors).
// After done, RAM A (Pool2) and RAM B (Conv2) are also read back while idle,
// along with GAP and logit storage.
//
// Also covers: ownership assertions (in DUT), write counts, repeated
// inference, mid-reset recovery, and multi-image finals.

`timescale 1ns / 1ps

module tb_cnn_shared_compute_end_to_end (
    input logic clk
);
    localparam int IN_N    = 3072;
    localparam int CONV1_N = 16384;
    localparam int POOL1_N = 4096;
    localparam int CONV2_N = 8192;
    localparam int POOL2_N = 2048;
    localparam int GAP_N   = 32;
    localparam int LOGIT_N = 5;
    localparam int NUM_IMG = 5;

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

    logic own_conv1, own_pool1, own_conv2, own_pool2, own_gap;
    logic dbg_act_a_we, dbg_act_b_we;
    logic [13:0] dbg_act_a_waddr, dbg_act_b_waddr;
    logic signed [7:0] dbg_act_a_wdata, dbg_act_b_wdata;
    logic dbg_act_a_re, dbg_act_b_re;
    logic [13:0] dbg_act_a_raddr, dbg_act_b_raddr;

    logic input_we;
    logic [11:0] input_waddr;
    logic signed [7:0] input_wdata;

    logic act_a_re, act_b_re, gap_re, logit_re;
    logic [13:0] act_a_raddr, act_b_raddr;
    logic [4:0]  gap_raddr;
    logic [2:0]  logit_raddr;
    logic signed [7:0] act_a_rdata, act_b_rdata, gap_rdata;
    logic signed [31:0] logit_rdata;

    logic signed [7:0]  exp_input  [0:IN_N-1];
    logic signed [7:0]  exp_conv1  [0:CONV1_N-1];
    logic signed [7:0]  exp_pool1  [0:POOL1_N-1];
    logic signed [7:0]  exp_conv2  [0:CONV2_N-1];
    logic signed [7:0]  exp_pool2  [0:POOL2_N-1];
    logic signed [7:0]  exp_gap    [0:GAP_N-1];
    logic signed [31:0] exp_logits [0:LOGIT_N-1];

    // Write shadows + seen flags for duplicate detection
    logic signed [7:0]  shadow_conv1 [0:CONV1_N-1];
    logic signed [7:0]  shadow_pool1 [0:POOL1_N-1];
    logic signed [7:0]  shadow_conv2 [0:CONV2_N-1];
    logic signed [7:0]  shadow_pool2 [0:POOL2_N-1];
    logic               seen_conv1   [0:CONV1_N-1];
    logic               seen_pool1   [0:POOL1_N-1];
    logic               seen_conv2   [0:CONV2_N-1];
    logic               seen_pool2   [0:POOL2_N-1];

    integer local_c, phase, i, rd_i, rd_wait, j, img;
    integer exp_class, exp_max;
    integer seen_c1, seen_p1, seen_c2, seen_p2, seen_gap, seen_fc, seen_am;
    integer wr_c1, wr_p1, wr_c2, wr_p2, wr_in;
    integer mm_c1, mm_p1, mm_c2, mm_p2, mm_gap, mm_fc;
    integer check_intermediates;
    integer pass_num;

    cnn_accelerator_shared_compute_top dut (
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
        .own_conv1(own_conv1),
        .own_pool1(own_pool1),
        .own_conv2(own_conv2),
        .own_pool2(own_pool2),
        .own_gap(own_gap),
        .dbg_act_a_we(dbg_act_a_we),
        .dbg_act_a_waddr(dbg_act_a_waddr),
        .dbg_act_a_wdata(dbg_act_a_wdata),
        .dbg_act_b_we(dbg_act_b_we),
        .dbg_act_b_waddr(dbg_act_b_waddr),
        .dbg_act_b_wdata(dbg_act_b_wdata),
        .dbg_act_a_re(dbg_act_a_re),
        .dbg_act_a_raddr(dbg_act_a_raddr),
        .dbg_act_b_re(dbg_act_b_re),
        .dbg_act_b_raddr(dbg_act_b_raddr),
        .input_write_enable(input_we),
        .input_write_address(input_waddr),
        .input_write_data(input_wdata),
        .act_a_read_enable(act_a_re),
        .act_a_read_address(act_a_raddr),
        .act_a_read_data(act_a_rdata),
        .act_b_read_enable(act_b_re),
        .act_b_read_address(act_b_raddr),
        .act_b_read_data(act_b_rdata),
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
            act_a_re = 0; act_b_re = 0; gap_re = 0; logit_re = 0;
            act_a_raddr = 0; act_b_raddr = 0; gap_raddr = 0; logit_raddr = 0;
        end
    endtask

    task automatic clear_shadows;
        begin
            for (j = 0; j < CONV1_N; j = j + 1) begin
                shadow_conv1[j] = 8'sd0;
                seen_conv1[j] = 1'b0;
            end
            for (j = 0; j < POOL1_N; j = j + 1) begin
                shadow_pool1[j] = 8'sd0;
                seen_pool1[j] = 1'b0;
            end
            for (j = 0; j < CONV2_N; j = j + 1) begin
                shadow_conv2[j] = 8'sd0;
                seen_conv2[j] = 1'b0;
            end
            for (j = 0; j < POOL2_N; j = j + 1) begin
                shadow_pool2[j] = 8'sd0;
                seen_pool2[j] = 1'b0;
            end
            wr_c1 = 0; wr_p1 = 0; wr_c2 = 0; wr_p2 = 0;
            mm_c1 = 0; mm_p1 = 0; mm_c2 = 0; mm_p2 = 0; mm_gap = 0; mm_fc = 0;
        end
    endtask

    task automatic load_primary;
        begin
            $readmemh("vectors/end_to_end/input_image.mem", exp_input);
            $readmemh("vectors/end_to_end/conv1_expected.mem", exp_conv1);
            $readmemh("vectors/end_to_end/pool1_expected.mem", exp_pool1);
            $readmemh("vectors/end_to_end/conv2_expected.mem", exp_conv2);
            $readmemh("vectors/end_to_end/pool2_expected.mem", exp_pool2);
            $readmemh("vectors/end_to_end/gap_expected.mem", exp_gap);
            $readmemh("vectors/end_to_end/fc_logits_expected.mem", exp_logits);
            exp_max = exp_logits[0];
            exp_class = 0;
            for (j = 1; j < LOGIT_N; j = j + 1) begin
                if ($signed(exp_logits[j]) > exp_max) begin
                    exp_max = $signed(exp_logits[j]);
                    exp_class = j;
                end
            end
        end
    endtask

    task automatic load_sample;
        input integer sample_sel;
        begin
            case (sample_sel)
                0: begin
                    $readmemh("vectors/end_to_end/multi_image/sample_000/input_image.mem", exp_input);
                    $readmemh("vectors/end_to_end/multi_image/sample_000/conv1_expected.mem", exp_conv1);
                    $readmemh("vectors/end_to_end/multi_image/sample_000/pool1_expected.mem", exp_pool1);
                    $readmemh("vectors/end_to_end/multi_image/sample_000/conv2_expected.mem", exp_conv2);
                    $readmemh("vectors/end_to_end/multi_image/sample_000/pool2_expected.mem", exp_pool2);
                    $readmemh("vectors/end_to_end/multi_image/sample_000/gap_expected.mem", exp_gap);
                    $readmemh("vectors/end_to_end/multi_image/sample_000/fc_logits_expected.mem", exp_logits);
                end
                1: begin
                    $readmemh("vectors/end_to_end/multi_image/sample_001/input_image.mem", exp_input);
                    $readmemh("vectors/end_to_end/multi_image/sample_001/conv1_expected.mem", exp_conv1);
                    $readmemh("vectors/end_to_end/multi_image/sample_001/pool1_expected.mem", exp_pool1);
                    $readmemh("vectors/end_to_end/multi_image/sample_001/conv2_expected.mem", exp_conv2);
                    $readmemh("vectors/end_to_end/multi_image/sample_001/pool2_expected.mem", exp_pool2);
                    $readmemh("vectors/end_to_end/multi_image/sample_001/gap_expected.mem", exp_gap);
                    $readmemh("vectors/end_to_end/multi_image/sample_001/fc_logits_expected.mem", exp_logits);
                end
                2: begin
                    $readmemh("vectors/end_to_end/multi_image/sample_008/input_image.mem", exp_input);
                    $readmemh("vectors/end_to_end/multi_image/sample_008/conv1_expected.mem", exp_conv1);
                    $readmemh("vectors/end_to_end/multi_image/sample_008/pool1_expected.mem", exp_pool1);
                    $readmemh("vectors/end_to_end/multi_image/sample_008/conv2_expected.mem", exp_conv2);
                    $readmemh("vectors/end_to_end/multi_image/sample_008/pool2_expected.mem", exp_pool2);
                    $readmemh("vectors/end_to_end/multi_image/sample_008/gap_expected.mem", exp_gap);
                    $readmemh("vectors/end_to_end/multi_image/sample_008/fc_logits_expected.mem", exp_logits);
                end
                3: begin
                    $readmemh("vectors/end_to_end/multi_image/sample_002/input_image.mem", exp_input);
                    $readmemh("vectors/end_to_end/multi_image/sample_002/conv1_expected.mem", exp_conv1);
                    $readmemh("vectors/end_to_end/multi_image/sample_002/pool1_expected.mem", exp_pool1);
                    $readmemh("vectors/end_to_end/multi_image/sample_002/conv2_expected.mem", exp_conv2);
                    $readmemh("vectors/end_to_end/multi_image/sample_002/pool2_expected.mem", exp_pool2);
                    $readmemh("vectors/end_to_end/multi_image/sample_002/gap_expected.mem", exp_gap);
                    $readmemh("vectors/end_to_end/multi_image/sample_002/fc_logits_expected.mem", exp_logits);
                end
                default: begin
                    $readmemh("vectors/end_to_end/multi_image/sample_005/input_image.mem", exp_input);
                    $readmemh("vectors/end_to_end/multi_image/sample_005/conv1_expected.mem", exp_conv1);
                    $readmemh("vectors/end_to_end/multi_image/sample_005/pool1_expected.mem", exp_pool1);
                    $readmemh("vectors/end_to_end/multi_image/sample_005/conv2_expected.mem", exp_conv2);
                    $readmemh("vectors/end_to_end/multi_image/sample_005/pool2_expected.mem", exp_pool2);
                    $readmemh("vectors/end_to_end/multi_image/sample_005/gap_expected.mem", exp_gap);
                    $readmemh("vectors/end_to_end/multi_image/sample_005/fc_logits_expected.mem", exp_logits);
                end
            endcase
            exp_max = exp_logits[0];
            exp_class = 0;
            for (j = 1; j < LOGIT_N; j = j + 1) begin
                if ($signed(exp_logits[j]) > exp_max) begin
                    exp_max = $signed(exp_logits[j]);
                    exp_class = j;
                end
            end
        end
    endtask

    task automatic compare_shadows;
        begin
            for (j = 0; j < CONV1_N; j = j + 1) begin
                if (!seen_conv1[j])
                    $fatal(1, "Conv1 addr %0d never written", j);
                if (shadow_conv1[j] !== exp_conv1[j]) begin
                    $display("FAIL Conv1 addr=%0d exp=%0d got=%0d",
                             j, exp_conv1[j], shadow_conv1[j]);
                    mm_c1 = mm_c1 + 1;
                    if (mm_c1 > 8) $fatal(1, "too many Conv1 mismatches");
                end
            end
            for (j = 0; j < POOL1_N; j = j + 1) begin
                if (!seen_pool1[j])
                    $fatal(1, "Pool1 addr %0d never written", j);
                if (shadow_pool1[j] !== exp_pool1[j]) begin
                    mm_p1 = mm_p1 + 1;
                    $fatal(1, "Pool1 addr=%0d exp=%0d got=%0d",
                           j, exp_pool1[j], shadow_pool1[j]);
                end
            end
            for (j = 0; j < CONV2_N; j = j + 1) begin
                if (!seen_conv2[j])
                    $fatal(1, "Conv2 addr %0d never written", j);
                if (shadow_conv2[j] !== exp_conv2[j]) begin
                    mm_c2 = mm_c2 + 1;
                    $fatal(1, "Conv2 addr=%0d", j);
                end
            end
            for (j = 0; j < POOL2_N; j = j + 1) begin
                if (!seen_pool2[j])
                    $fatal(1, "Pool2 addr %0d never written", j);
                if (shadow_pool2[j] !== exp_pool2[j]) begin
                    mm_p2 = mm_p2 + 1;
                    $fatal(1, "Pool2 addr=%0d", j);
                end
            end
            if (wr_c1 !== CONV1_N) $fatal(1, "Conv1 write count %0d", wr_c1);
            if (wr_p1 !== POOL1_N) $fatal(1, "Pool1 write count %0d", wr_p1);
            if (wr_c2 !== CONV2_N) $fatal(1, "Conv2 write count %0d", wr_c2);
            if (wr_p2 !== POOL2_N) $fatal(1, "Pool2 write count %0d", wr_p2);
            if (mm_c1 !== 0) $fatal(1, "Conv1 mismatches=%0d", mm_c1);
            if (mm_p1 !== 0) $fatal(1, "Pool1 mismatches=%0d", mm_p1);
            if (mm_c2 !== 0) $fatal(1, "Conv2 mismatches=%0d", mm_c2);
            if (mm_p2 !== 0) $fatal(1, "Pool2 mismatches=%0d", mm_p2);
            $display("Shared-compute intermediates OK: c1=%0d p1=%0d c2=%0d p2=%0d writes matched",
                     wr_c1, wr_p1, wr_c2, wr_p2);
        end
    endtask

    initial begin
        local_c = 0; phase = 0; i = 0; rd_i = 0; rd_wait = 0; j = 0; img = 0;
        seen_c1 = 0; seen_p1 = 0; seen_c2 = 0; seen_p2 = 0;
        seen_gap = 0; seen_fc = 0; seen_am = 0;
        wr_in = 0; pass_num = 0; check_intermediates = 1;
        rst = 1;
        clear_hosts();
        clear_shadows();
        load_primary();
    end

    // phases:
    // 0 reset + load RAM A with input
    // 1 pulse start
    // 2 wait done (capture writes live)
    // 3 idle readback: Pool2 in A, Conv2 in B, GAP, logits
    // 4 check finals / report
    // 5 repeated inference (same image, no reload needed for finals-only)
    // 6 wait done2 + finals
    // 7 mid-reset: start, reset, reload, run
    // 8 wait done3
    // 9 multi-image loop
    // 10 PASS

    always @(negedge clk) begin
        local_c = local_c + 1;
        if (phase == 0) begin
            if (local_c <= 3) begin
                rst = 1; clear_hosts();
            end else if (local_c == 4) begin
                rst = 0;
                i = 0;
                wr_in = 0;
                clear_shadows();
            end else if (local_c >= 5) begin
                if (i < IN_N) begin
                    input_we = 1;
                    input_waddr = i[11:0];
                    input_wdata = exp_input[i];
                    i = i + 1;
                    wr_in = wr_in + 1;
                end else begin
                    input_we = 0;
                    if (wr_in !== IN_N) $fatal(1, "input preload count %0d", wr_in);
                    phase = 1;
                    local_c = 0;
                end
            end
        end else if (phase == 1) begin
            if (local_c == 1) begin
                start = 1;
                seen_c1 = 0; seen_p1 = 0; seen_c2 = 0; seen_p2 = 0;
                seen_gap = 0; seen_fc = 0; seen_am = 0;
            end else begin
                start = 0;
                phase = 2;
            end
        end else if (phase == 2) begin
            start = 0;
            if (local_c == 10)
                start = 1; // must be ignored while busy
            else
                start = 0;
        end else if (phase == 3) begin
            act_a_re = 0; act_b_re = 0; gap_re = 0; logit_re = 0;
            if (rd_wait == 0) begin
                // After done: A has Pool2[0:2047], B has Conv2[0:8191]
                if (rd_i < POOL2_N) begin
                    act_a_re = 1; act_a_raddr = rd_i[13:0];
                end else if (rd_i < POOL2_N + CONV2_N) begin
                    act_b_re = 1; act_b_raddr = (rd_i - POOL2_N);
                end else if (rd_i < POOL2_N + CONV2_N + GAP_N) begin
                    gap_re = 1; gap_raddr = (rd_i - POOL2_N - CONV2_N);
                end else if (rd_i < POOL2_N + CONV2_N + GAP_N + LOGIT_N) begin
                    logit_re = 1;
                    logit_raddr = (rd_i - POOL2_N - CONV2_N - GAP_N);
                end
                rd_wait = 1;
            end else begin
                rd_wait = 2;
            end
        end else if (phase == 5) begin
            if (local_c == 1) start = 1;
            else begin start = 0; phase = 6; end
        end else if (phase == 7) begin
            if (local_c == 1) start = 1;
            else if (local_c == 2) start = 0;
            else if (local_c == 50) begin
                rst = 1; clear_hosts();
            end else if (local_c == 53) begin
                rst = 0; i = 0; clear_shadows();
            end else if (local_c >= 54 && rst == 0) begin
                if (i < IN_N) begin
                    input_we = 1;
                    input_waddr = i[11:0];
                    input_wdata = exp_input[i];
                    i = i + 1;
                end else begin
                    input_we = 0;
                    start = 1;
                    check_intermediates = 0; // finals only after mid-reset
                    phase = 8;
                    local_c = 0;
                    seen_c1 = 0; seen_p1 = 0; seen_c2 = 0; seen_p2 = 0;
                    seen_gap = 0; seen_fc = 0; seen_am = 0;
                end
            end
        end else if (phase == 8) begin
            if (local_c == 1) start = 0;
        end else if (phase == 9) begin
            // multi-image: load + write + start handled below via sub-phases
            // reuse phase 0 style via img_phase encoded in local state
            ;
        end
    end

    // Live write capture (ping-pong intermediates)
    always @(posedge clk) begin
        if (!rst && (phase == 2 || phase == 6 || phase == 8 || phase == 12)) begin
            if (own_conv1 && dbg_act_b_we) begin
                if (seen_conv1[dbg_act_b_waddr])
                    $fatal(1, "duplicate Conv1 write addr=%0d", dbg_act_b_waddr);
                if (dbg_act_b_waddr >= CONV1_N)
                    $fatal(1, "Conv1 write OOB %0d", dbg_act_b_waddr);
                shadow_conv1[dbg_act_b_waddr] = dbg_act_b_wdata;
                seen_conv1[dbg_act_b_waddr] = 1'b1;
                wr_c1 = wr_c1 + 1;
            end
            if (own_pool1 && dbg_act_a_we) begin
                if (seen_pool1[dbg_act_a_waddr[11:0]])
                    $fatal(1, "duplicate Pool1 write");
                if (dbg_act_a_waddr >= POOL1_N)
                    $fatal(1, "Pool1 write OOB");
                shadow_pool1[dbg_act_a_waddr] = dbg_act_a_wdata;
                seen_pool1[dbg_act_a_waddr] = 1'b1;
                wr_p1 = wr_p1 + 1;
            end
            if (own_conv2 && dbg_act_b_we) begin
                if (seen_conv2[dbg_act_b_waddr[12:0]])
                    $fatal(1, "duplicate Conv2 write");
                if (dbg_act_b_waddr >= CONV2_N)
                    $fatal(1, "Conv2 write OOB");
                shadow_conv2[dbg_act_b_waddr] = dbg_act_b_wdata;
                seen_conv2[dbg_act_b_waddr] = 1'b1;
                wr_c2 = wr_c2 + 1;
            end
            if (own_pool2 && dbg_act_a_we) begin
                if (seen_pool2[dbg_act_a_waddr[10:0]])
                    $fatal(1, "duplicate Pool2 write");
                if (dbg_act_a_waddr >= POOL2_N)
                    $fatal(1, "Pool2 write OOB");
                shadow_pool2[dbg_act_a_waddr] = dbg_act_a_wdata;
                seen_pool2[dbg_act_a_waddr] = 1'b1;
                wr_p2 = wr_p2 + 1;
            end

            // Ownership: only the owning stage may drive the matching RAM
            if (own_conv1 && dbg_act_a_we)
                $fatal(1, "Conv1 must not write RAM A");
            if (own_pool1 && dbg_act_b_we)
                $fatal(1, "Pool1 must not write RAM B");
            if (own_conv2 && dbg_act_a_we)
                $fatal(1, "Conv2 must not write RAM A");
            if (own_pool2 && dbg_act_b_we)
                $fatal(1, "Pool2 must not write RAM B");
            if (own_gap && (dbg_act_a_we || dbg_act_b_we))
                $fatal(1, "GAP must not write activation RAMs");
            if (own_conv1 && dbg_act_b_re)
                $fatal(1, "Conv1 must not read RAM B");
            if (own_pool1 && dbg_act_a_re)
                $fatal(1, "Pool1 must not read RAM A");
            if (own_conv2 && dbg_act_b_re)
                $fatal(1, "Conv2 must not read RAM B");
            if (own_pool2 && dbg_act_a_re)
                $fatal(1, "Pool2 must not read RAM A");
            if (own_gap && dbg_act_b_re)
                $fatal(1, "GAP must not read RAM B");
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
                if (check_intermediates)
                    compare_shadows();
                phase = 3;
                rd_i = 0;
                rd_wait = 0;
            end
        end

        if (phase == 3 && rd_wait == 2) begin
            if (rd_i < POOL2_N) begin
                if (act_a_rdata !== exp_pool2[rd_i])
                    $fatal(1, "idle RAM A pool2[%0d]", rd_i);
            end else if (rd_i < POOL2_N + CONV2_N) begin
                if (act_b_rdata !== exp_conv2[rd_i - POOL2_N])
                    $fatal(1, "idle RAM B conv2[%0d]", rd_i - POOL2_N);
            end else if (rd_i < POOL2_N + CONV2_N + GAP_N) begin
                if (gap_rdata !== exp_gap[rd_i - POOL2_N - CONV2_N]) begin
                    mm_gap = mm_gap + 1;
                    $fatal(1, "gap[%0d]", rd_i - POOL2_N - CONV2_N);
                end
            end else begin
                if (logit_rdata !== exp_logits[rd_i - POOL2_N - CONV2_N - GAP_N]) begin
                    mm_fc = mm_fc + 1;
                    $fatal(1, "logit[%0d]", rd_i - POOL2_N - CONV2_N - GAP_N);
                end
            end
            rd_i = rd_i + 1;
            rd_wait = 0;
            if (rd_i >= POOL2_N + CONV2_N + GAP_N + LOGIT_N)
                phase = 4;
        end

        if (phase == 4) begin
            if (predicted_class !== exp_class[2:0])
                $fatal(1, "class got=%0d exp=%0d", predicted_class, exp_class);
            if ($signed(maximum_logit) !== exp_max)
                $fatal(1, "max logit got=%0d exp=%0d", $signed(maximum_logit), exp_max);
            if (logit_0 !== exp_logits[0]) $fatal(1, "logit0");
            if (logit_1 !== exp_logits[1]) $fatal(1, "logit1");
            if (logit_2 !== exp_logits[2]) $fatal(1, "logit2");
            if (logit_3 !== exp_logits[3]) $fatal(1, "logit3");
            if (logit_4 !== exp_logits[4]) $fatal(1, "logit4");
            $display("==== Shared-compute single-image pass 1 ====");
            $display("predicted_class=%0d maximum_logit=%0d",
                     predicted_class, $signed(maximum_logit));
            $display("mismatch counts: c1=%0d p1=%0d c2=%0d p2=%0d gap=%0d fc=%0d",
                     mm_c1, mm_p1, mm_c2, mm_p2, mm_gap, mm_fc);
            $display("cycle_count=%0d", cycle_count);
            $display("shared-compute layer cycles: Conv1=%0d Pool1=%0d Conv2=%0d Pool2=%0d",
                     dbg_conv1_cycles, dbg_pool1_cycles,
                     dbg_conv2_cycles, dbg_pool2_cycles);
            $display("NOTE: RAM B Conv1 overwritten by Conv2 (intentional reuse)");
            $display("NOTE: RAM A input/Pool1 overwritten by later stages (intentional)");
            pass_num = 1;
            // Pool2 overwrote RAM A[0:2047]; reload input 0..3071 before repeat
            phase = 20;
            local_c = 0;
            i = 0;
        end

        // phase 20: reload input for repeat
        // handled in negedge below via dual use — do on posedge simply:
        if (phase == 20) begin
            // drive happens on negedge; here just wait
        end

        if (phase == 6 && done) begin
            if (predicted_class !== exp_class[2:0]) $fatal(1, "repeat class");
            if ($signed(maximum_logit) !== exp_max) $fatal(1, "repeat max");
            $display("==== Shared-compute repeated inference PASS (cycles=%0d) ====",
                     cycle_count);
            phase = 7;
            local_c = 0;
        end

        if (phase == 8 && done) begin
            if (predicted_class !== exp_class[2:0]) $fatal(1, "post-reset class");
            if ($signed(maximum_logit) !== exp_max) $fatal(1, "post-reset max");
            $display("==== Shared-compute reset-during-inference recovery PASS ====");
            img = 0;
            load_sample(0);
            check_intermediates = 1; // full intermediates for multi image 0
            phase = 10; // multi-image load
            local_c = 0;
            i = 0;
        end

        // Multi-image wait done
        if (phase == 12 && done) begin
            if (check_intermediates)
                compare_shadows();
            if (predicted_class !== exp_class[2:0])
                $fatal(1, "multi img%0d class got=%0d exp=%0d",
                       img, predicted_class, exp_class);
            if ($signed(maximum_logit) !== exp_max)
                $fatal(1, "multi img%0d max", img);
            if (logit_0 !== exp_logits[0]) $fatal(1, "multi logit0");
            if (logit_1 !== exp_logits[1]) $fatal(1, "multi logit1");
            if (logit_2 !== exp_logits[2]) $fatal(1, "multi logit2");
            if (logit_3 !== exp_logits[3]) $fatal(1, "multi logit3");
            if (logit_4 !== exp_logits[4]) $fatal(1, "multi logit4");
            $display("PASS shared-compute image %0d class=%0d cycles=%0d",
                     img, predicted_class, cycle_count);
            img = img + 1;
            if (img >= NUM_IMG) begin
                $display("PASS: tb_cnn_shared_compute_end_to_end");
                $display("Conv1 mismatches=%0d Pool1=%0d Conv2=%0d Pool2=%0d GAP=%0d FC=%0d",
                         mm_c1, mm_p1, mm_c2, mm_p2, mm_gap, mm_fc);
                $finish;
            end else begin
                load_sample(img);
                // Full intermediates for all images if runtime OK
                check_intermediates = 1;
                phase = 10;
                local_c = 0;
                i = 0;
                clear_shadows();
            end
        end

        if (local_c > 100000000)
            $fatal(1, "timeout phase=%0d", phase);
    end

    // Extended negedge for reload/multi-image
    always @(negedge clk) begin
        if (phase == 20) begin
            if (local_c == 1) begin
                i = 0;
            end
            if (i < IN_N) begin
                input_we = 1;
                input_waddr = i[11:0];
                input_wdata = exp_input[i];
                i = i + 1;
            end else begin
                input_we = 0;
                clear_shadows();
                check_intermediates = 0; // finals-only on repeat
                phase = 5;
                local_c = 0;
            end
        end else if (phase == 10) begin
            if (local_c <= 2) begin
                clear_hosts();
                clear_shadows();
            end else if (i < IN_N) begin
                input_we = 1;
                input_waddr = i[11:0];
                input_wdata = exp_input[i];
                i = i + 1;
            end else begin
                input_we = 0;
                phase = 11;
                local_c = 0;
            end
        end else if (phase == 11) begin
            if (local_c == 1) begin
                start = 1;
                seen_c1 = 0; seen_p1 = 0; seen_c2 = 0; seen_p2 = 0;
                seen_gap = 0; seen_fc = 0; seen_am = 0;
            end else begin
                start = 0;
                phase = 12;
            end
        end
    end

endmodule
