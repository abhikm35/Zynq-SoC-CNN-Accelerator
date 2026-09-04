// tb_cnn_end_to_end_multi_image.sv
// One image per class. Reloads expected mems per image.

`timescale 1ns / 1ps

module tb_cnn_end_to_end_multi_image (
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
    integer exp_class, exp_max;

    integer local_c, phase, i, img, rd_i, rd_wait, j;

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

    initial begin
        local_c=0; phase=0; i=0; img=0; rd_i=0; rd_wait=0; j=0;
        rst=1; start=0; input_we=0;
        conv1_re=0; pool1_re=0; conv2_re=0; pool2_re=0; gap_re=0; logit_re=0;
        load_sample(0);
    end

    always @(negedge clk) begin
        local_c = local_c + 1;
        if (phase == 0) begin
            if (local_c <= 3) begin
                rst = 1; start = 0; input_we = 0;
            end else if (local_c == 4) begin
                rst = 0; i = 0;
            end else begin
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
            if (local_c == 1) start = 1;
            else begin start = 0; phase = 2; end
        end else if (phase == 3) begin
            conv1_re=0; pool1_re=0; conv2_re=0; pool2_re=0; gap_re=0; logit_re=0;
            if (rd_wait == 0) begin
                if (rd_i < CONV1_N) begin
                    conv1_re=1; conv1_raddr=rd_i[13:0];
                end else if (rd_i < CONV1_N+POOL1_N) begin
                    pool1_re=1; pool1_raddr=(rd_i-CONV1_N);
                end else if (rd_i < CONV1_N+POOL1_N+CONV2_N) begin
                    conv2_re=1; conv2_raddr=(rd_i-CONV1_N-POOL1_N);
                end else if (rd_i < CONV1_N+POOL1_N+CONV2_N+POOL2_N) begin
                    pool2_re=1; pool2_raddr=(rd_i-CONV1_N-POOL1_N-CONV2_N);
                end else if (rd_i < CONV1_N+POOL1_N+CONV2_N+POOL2_N+GAP_N) begin
                    gap_re=1; gap_raddr=(rd_i-CONV1_N-POOL1_N-CONV2_N-POOL2_N);
                end else begin
                    logit_re=1;
                    logit_raddr=(rd_i-CONV1_N-POOL1_N-CONV2_N-POOL2_N-GAP_N);
                end
                rd_wait = 1;
            end else rd_wait = 2;
        end
    end

    always @(posedge clk) begin
        if (phase == 2 && done) begin
            phase = 3; rd_i = 0; rd_wait = 0;
        end
        if (phase == 3 && rd_wait == 2) begin
            if (rd_i < CONV1_N) begin
                if (conv1_rdata !== exp_conv1[rd_i])
                    $fatal(1, "img%0d conv1[%0d] exp=%0d got=%0d", img, rd_i, exp_conv1[rd_i], conv1_rdata);
            end else if (rd_i < CONV1_N+POOL1_N) begin
                if (pool1_rdata !== exp_pool1[rd_i-CONV1_N])
                    $fatal(1, "img%0d pool1", img);
            end else if (rd_i < CONV1_N+POOL1_N+CONV2_N) begin
                if (conv2_rdata !== exp_conv2[rd_i-CONV1_N-POOL1_N])
                    $fatal(1, "img%0d conv2", img);
            end else if (rd_i < CONV1_N+POOL1_N+CONV2_N+POOL2_N) begin
                if (pool2_rdata !== exp_pool2[rd_i-CONV1_N-POOL1_N-CONV2_N])
                    $fatal(1, "img%0d pool2", img);
            end else if (rd_i < CONV1_N+POOL1_N+CONV2_N+POOL2_N+GAP_N) begin
                if (gap_rdata !== exp_gap[rd_i-CONV1_N-POOL1_N-CONV2_N-POOL2_N])
                    $fatal(1, "img%0d gap", img);
            end else begin
                if (logit_rdata !== exp_logits[rd_i-CONV1_N-POOL1_N-CONV2_N-POOL2_N-GAP_N])
                    $fatal(1, "img%0d logit", img);
            end
            rd_i = rd_i + 1;
            rd_wait = 0;
            if (rd_i >= CONV1_N+POOL1_N+CONV2_N+POOL2_N+GAP_N+LOGIT_N) begin
                if (predicted_class !== exp_class[2:0])
                    $fatal(1, "img%0d class got=%0d exp=%0d", img, predicted_class, exp_class);
                if ($signed(maximum_logit) !== exp_max)
                    $fatal(1, "img%0d max", img);
                $display("PASS image %0d class=%0d cycles=%0d", img, predicted_class, cycle_count);
                img = img + 1;
                if (img >= NUM_IMG) begin
                    $display("PASS: tb_cnn_end_to_end_multi_image");
                    $finish;
                end else begin
                    load_sample(img);
                    phase = 0;
                    local_c = 0;
                    i = 0;
                end
            end
        end
        if (local_c > 100000000)
            $fatal(1, "timeout img=%0d phase=%0d", img, phase);
    end
endmodule
