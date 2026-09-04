// cnn_accelerator_shared_conv_top.sv
// Ping-pong activation RAMs + one shared Conv1/Conv2 compute engine.
//
// Trained sizes (obsolete 8/16 prompt sketches are not used):
//   Input 3x32x32 -> Conv1 16x32x32 -> Pool1 16x16x16
//                -> Conv2 32x16x16 -> Pool2 32x8x8 -> GAP 32 -> FC 5 -> Argmax
//
// Shared convolution engine runs Conv1 then later Conv2 (never simultaneous).
// Pool1/Pool2/GAP/FC/Argmax remain separate. Parameter ROMs stay layer-specific.
// Prior ping-pong top with separate Conv engines remains for regression.

`timescale 1ns / 1ps

module cnn_accelerator_shared_conv_top #(
    parameter int LOGIT_WIDTH = 32,
    parameter CONV1_WGT_MEM  = "vectors/conv1_memory/conv1_weights.mem",
    parameter CONV1_BIAS_MEM = "vectors/conv1_memory/conv1_biases.mem",
    parameter CONV1_MULT_MEM = "vectors/conv1_memory/conv1_multipliers.mem",
    parameter CONV1_SHIFT_MEM= "vectors/conv1_memory/conv1_shifts.mem",
    parameter CONV2_WGT_MEM  = "vectors/conv2/conv2_weights.mem",
    parameter CONV2_BIAS_MEM = "vectors/conv2/conv2_biases.mem",
    parameter CONV2_MULT_MEM = "vectors/conv2/conv2_multipliers.mem",
    parameter CONV2_SHIFT_MEM= "vectors/conv2/conv2_shifts.mem",
    parameter FC_WGT_MEM     = "vectors/fc/fc_weights.mem",
    parameter FC_BIAS_MEM    = "vectors/fc/fc_biases.mem",
    parameter FC_MULT_MEM    = "vectors/fc/fc_multipliers.mem",
    parameter FC_SHIFT_MEM   = "vectors/fc/fc_shifts.mem",
    parameter logic signed [31:0] GAP_MULTIPLIER = 32'sd1759306569,
    parameter logic [5:0]         GAP_SHIFT = 6'd29
) (
    input  logic                           clk,
    input  logic                           rst,
    input  logic                           start,

    output logic                           busy,
    output logic                           done,

    output logic [2:0]                     predicted_class,
    output logic signed [LOGIT_WIDTH-1:0]  maximum_logit,

    output logic signed [LOGIT_WIDTH-1:0]  logit_0,
    output logic signed [LOGIT_WIDTH-1:0]  logit_1,
    output logic signed [LOGIT_WIDTH-1:0]  logit_2,
    output logic signed [LOGIT_WIDTH-1:0]  logit_3,
    output logic signed [LOGIT_WIDTH-1:0]  logit_4,

    output logic [63:0]                    cycle_count,
    output logic [3:0]                     state_id,
    output logic [2:0]                     stage_id,

    output logic                           dbg_conv1_done,
    output logic                           dbg_pool1_done,
    output logic                           dbg_conv2_done,
    output logic                           dbg_pool2_done,
    output logic                           dbg_gap_done,
    output logic                           dbg_fc_done,
    output logic                           dbg_argmax_done,

    output logic [14:0]                    dbg_conv1_output_count,
    output logic [12:0]                    dbg_pool1_output_count,
    output logic [13:0]                    dbg_conv2_output_count,
    output logic [11:0]                    dbg_pool2_output_count,
    output logic [5:0]                     dbg_gap_output_count,
    output logic [7:0]                     dbg_fc_mac_count,

    output logic [63:0]                    dbg_conv1_cycles,
    output logic [63:0]                    dbg_pool1_cycles,
    output logic [63:0]                    dbg_conv2_cycles,
    output logic [63:0]                    dbg_pool2_cycles,
    output logic [63:0]                    dbg_gap_cycles,
    output logic [63:0]                    dbg_fc_cycles,
    output logic [63:0]                    dbg_argmax_cycles,

    // Ownership flags (for TB assertions)
    output logic                           own_conv1,
    output logic                           own_pool1,
    output logic                           own_conv2,
    output logic                           own_pool2,
    output logic                           own_gap,

    // Live activation write monitors (muxed RAM ports)
    output logic                           dbg_act_a_we,
    output logic [13:0]                    dbg_act_a_waddr,
    output logic signed [7:0]              dbg_act_a_wdata,
    output logic                           dbg_act_b_we,
    output logic [13:0]                    dbg_act_b_waddr,
    output logic signed [7:0]              dbg_act_b_wdata,

    // Live activation read monitors
    output logic                           dbg_act_a_re,
    output logic [13:0]                    dbg_act_a_raddr,
    output logic                           dbg_act_b_re,
    output logic [13:0]                    dbg_act_b_raddr,

    // Host / TB: load input image into RAM A when idle
    input  logic                           input_write_enable,
    input  logic [11:0]                    input_write_address,
    input  logic signed [7:0]              input_write_data,

    // Host / TB: readback when idle
    input  logic                           act_a_read_enable,
    input  logic [13:0]                    act_a_read_address,
    output logic signed [7:0]              act_a_read_data,

    input  logic                           act_b_read_enable,
    input  logic [13:0]                    act_b_read_address,
    output logic signed [7:0]              act_b_read_data,

    input  logic                           gap_read_enable,
    input  logic [4:0]                     gap_read_address,
    output logic signed [7:0]              gap_read_data,

    input  logic                           logit_read_enable,
    input  logic [2:0]                     logit_read_address,
    output logic signed [LOGIT_WIDTH-1:0]  logit_read_data
);

    // -------------------------------------------------------------------------
    // Top controller (unchanged sequencing)
    // -------------------------------------------------------------------------
    logic conv1_start, pool1_start, conv2_start, pool2_start;
    logic gap_start, fc_start, argmax_start;
    logic conv1_busy, pool1_busy, conv2_busy, pool2_busy;
    logic gap_busy, fc_busy, argmax_busy;
    logic conv1_done, pool1_done, conv2_done, pool2_done;
    logic gap_done, fc_done, argmax_done;

    cnn_top_controller u_top_ctrl (
        .clk(clk), .rst(rst), .start(start),
        .busy(busy), .done(done),
        .conv1_start(conv1_start), .conv1_busy(conv1_busy), .conv1_done(conv1_done),
        .pool1_start(pool1_start), .pool1_busy(pool1_busy), .pool1_done(pool1_done),
        .conv2_start(conv2_start), .conv2_busy(conv2_busy), .conv2_done(conv2_done),
        .pool2_start(pool2_start), .pool2_busy(pool2_busy), .pool2_done(pool2_done),
        .gap_start(gap_start),     .gap_busy(gap_busy),     .gap_done(gap_done),
        .fc_start(fc_start),       .fc_busy(fc_busy),       .fc_done(fc_done),
        .argmax_start(argmax_start),.argmax_busy(argmax_busy),.argmax_done(argmax_done),
        .state_id(state_id), .stage_id(stage_id), .cycle_count(cycle_count),
        .conv1_cycles(dbg_conv1_cycles),
        .pool1_cycles(dbg_pool1_cycles),
        .conv2_cycles(dbg_conv2_cycles),
        .pool2_cycles(dbg_pool2_cycles),
        .gap_cycles(dbg_gap_cycles),
        .fc_cycles(dbg_fc_cycles),
        .argmax_cycles(dbg_argmax_cycles)
    );

    assign dbg_conv1_done  = conv1_done;
    assign dbg_pool1_done  = pool1_done;
    assign dbg_conv2_done  = conv2_done;
    assign dbg_pool2_done  = pool2_done;
    assign dbg_gap_done    = gap_done;
    assign dbg_fc_done     = fc_done;
    assign dbg_argmax_done = argmax_done;

    // Ownership from FSM state_id (matches cnn_top_controller encoding)
    // 1/2 Conv1, 3/4 Pool1, 5/6 Conv2, 7/8 Pool2, 9/10 GAP
    assign own_conv1 = (state_id == 4'd1) || (state_id == 4'd2);
    assign own_pool1 = (state_id == 4'd3) || (state_id == 4'd4);
    assign own_conv2 = (state_id == 4'd5) || (state_id == 4'd6);
    assign own_pool2 = (state_id == 4'd7) || (state_id == 4'd8);
    assign own_gap   = (state_id == 4'd9) || (state_id == 4'd10);

    // -------------------------------------------------------------------------
    // Stage memory interface wires
    // -------------------------------------------------------------------------
    // Shared conv engine (Conv1 then Conv2)
    logic shared_conv_start;
    logic shared_conv_busy, shared_conv_done;
    logic shared_layer_is_conv2;
    logic shared_act_re;
    logic [11:0] shared_act_addr;
    logic signed [7:0] shared_act_rdata;
    logic shared_we;
    logic [13:0] shared_waddr;
    logic signed [7:0] shared_wdata;
    logic [14:0] shared_out_count;
    logic shared_dbg_layer;

    // Pool1 reads Conv1 tensor / writes Pool1 tensor
    logic pool1_re;
    logic [13:0] pool1_raddr;
    logic signed [7:0] pool1_rdata;
    logic pool1_we;
    logic [11:0] pool1_waddr;
    logic signed [7:0] pool1_wdata;

    // Pool2 reads Conv2 / writes Pool2
    logic pool2_re;
    logic [12:0] pool2_raddr;
    logic signed [7:0] pool2_rdata;
    logic pool2_we;
    logic [10:0] pool2_waddr;
    logic signed [7:0] pool2_wdata;

    // GAP reads Pool2 / writes GAP storage
    logic gap_act_re;
    logic [10:0] gap_act_raddr;
    logic signed [7:0] gap_act_rdata;
    logic gap_we;
    logic [4:0] gap_waddr;
    logic signed [7:0] gap_wdata;

    logic gap_re_hw;
    logic [4:0] gap_raddr_hw;
    logic signed [7:0] gap_rdata_w;

    logic logit_we;
    logic [2:0] logit_waddr;
    logic signed [LOGIT_WIDTH-1:0] logit_wdata;
    logic logit_re_hw;
    logic [2:0] logit_raddr_hw;
    logic signed [LOGIT_WIDTH-1:0] logit_rdata_w;

    // -------------------------------------------------------------------------
    // Ping-pong RAM A / RAM B ownership mux
    // -------------------------------------------------------------------------
    logic act_a_we;
    logic [13:0] act_a_waddr;
    logic signed [7:0] act_a_wdata;
    logic act_a_re;
    logic [13:0] act_a_raddr;
    logic signed [7:0] act_a_rdata;

    logic act_b_we;
    logic [13:0] act_b_waddr;
    logic signed [7:0] act_b_wdata;
    logic act_b_re;
    logic [13:0] act_b_raddr;
    logic signed [7:0] act_b_rdata;

    // RAM A write: input load | Pool1 | Pool2
    always_comb begin
        act_a_we    = 1'b0;
        act_a_waddr = 14'd0;
        act_a_wdata = 8'sd0;
        if (~busy && input_write_enable) begin
            act_a_we    = 1'b1;
            act_a_waddr = {2'b0, input_write_address};
            act_a_wdata = input_write_data;
        end else if (own_pool1) begin
            act_a_we    = pool1_we;
            act_a_waddr = {2'b0, pool1_waddr};
            act_a_wdata = pool1_wdata;
        end else if (own_pool2) begin
            act_a_we    = pool2_we;
            act_a_waddr = {3'b0, pool2_waddr};
            act_a_wdata = pool2_wdata;
        end
    end

    // RAM A read: shared Conv | GAP | TB idle
    always_comb begin
        act_a_re    = 1'b0;
        act_a_raddr = 14'd0;
        if (own_conv1 || own_conv2) begin
            act_a_re    = shared_act_re;
            act_a_raddr = {2'b0, shared_act_addr};
        end else if (own_gap) begin
            act_a_re    = gap_act_re;
            act_a_raddr = {3'b0, gap_act_raddr};
        end else if (~busy) begin
            act_a_re    = act_a_read_enable;
            act_a_raddr = act_a_read_address;
        end
    end

    // RAM B write: shared Conv1 or Conv2
    always_comb begin
        act_b_we    = 1'b0;
        act_b_waddr = 14'd0;
        act_b_wdata = 8'sd0;
        if (own_conv1 || own_conv2) begin
            act_b_we    = shared_we;
            act_b_waddr = shared_waddr;
            act_b_wdata = shared_wdata;
        end
    end

    // RAM B read: Pool1 | Pool2 | TB idle
    always_comb begin
        act_b_re    = 1'b0;
        act_b_raddr = 14'd0;
        if (own_pool1) begin
            act_b_re    = pool1_re;
            act_b_raddr = pool1_raddr;
        end else if (own_pool2) begin
            act_b_re    = pool2_re;
            act_b_raddr = {1'b0, pool2_raddr};
        end else if (~busy) begin
            act_b_re    = act_b_read_enable;
            act_b_raddr = act_b_read_address;
        end
    end

    // Fan-out read data to stage consumers
    assign shared_act_rdata = act_a_rdata;
    assign gap_act_rdata  = act_a_rdata;
    assign pool1_rdata    = act_b_rdata;
    assign pool2_rdata    = act_b_rdata;
    assign act_a_read_data = act_a_rdata;
    assign act_b_read_data = act_b_rdata;

    assign dbg_act_a_we    = act_a_we;
    assign dbg_act_a_waddr = act_a_waddr;
    assign dbg_act_a_wdata = act_a_wdata;
    assign dbg_act_b_we    = act_b_we;
    assign dbg_act_b_waddr = act_b_waddr;
    assign dbg_act_b_wdata = act_b_wdata;
    assign dbg_act_a_re    = act_a_re;
    assign dbg_act_a_raddr = act_a_raddr;
    assign dbg_act_b_re    = act_b_re;
    assign dbg_act_b_raddr = act_b_raddr;

    activation_ram #(.DEPTH(16384), .ADDR_WIDTH(14)) u_act_ram_a (
        .clk(clk),
        .write_enable(act_a_we),
        .write_address(act_a_waddr),
        .write_data(act_a_wdata),
        .read_enable(act_a_re),
        .read_address(act_a_raddr),
        .read_data(act_a_rdata)
    );

    activation_ram #(.DEPTH(16384), .ADDR_WIDTH(14)) u_act_ram_b (
        .clk(clk),
        .write_enable(act_b_we),
        .write_address(act_b_waddr),
        .write_data(act_b_wdata),
        .read_enable(act_b_re),
        .read_address(act_b_raddr),
        .read_data(act_b_rdata)
    );

    // GAP / logit storage (unchanged)
    logic gap_re_mux;
    logic [4:0] gap_raddr_mux;
    assign gap_re_mux    = busy ? gap_re_hw : gap_read_enable;
    assign gap_raddr_mux = busy ? gap_raddr_hw : gap_read_address;
    assign gap_read_data = gap_rdata_w;

    logic logit_re_mux;
    logic [2:0] logit_raddr_mux;
    assign logit_re_mux    = busy ? logit_re_hw : logit_read_enable;
    assign logit_raddr_mux = busy ? logit_raddr_hw : logit_read_address;
    assign logit_read_data = logit_rdata_w;

    gap_output_storage #(.DEPTH(32), .ADDR_WIDTH(5)) u_gap_store (
        .clk(clk),
        .write_enable(gap_we),
        .write_address(gap_waddr),
        .write_data(gap_wdata),
        .read_enable(gap_re_mux),
        .read_address(gap_raddr_mux),
        .read_data(gap_rdata_w)
    );

    logit_storage #(.DEPTH(5), .ADDR_WIDTH(3)) u_logit_store (
        .clk(clk),
        .write_enable(logit_we),
        .write_address(logit_waddr),
        .write_data(logit_wdata),
        .read_enable(logit_re_mux),
        .read_address(logit_raddr_mux),
        .read_data(logit_rdata_w)
    );

    // -------------------------------------------------------------------------
    // Shared Conv1/Conv2 engine
    // -------------------------------------------------------------------------
    assign shared_conv_start     = conv1_start | conv2_start;
    assign shared_layer_is_conv2 = own_conv2;
    assign conv1_busy = own_conv1 & shared_conv_busy;
    assign conv2_busy = own_conv2 & shared_conv_busy;
    assign conv1_done = own_conv1 & shared_conv_done;
    assign conv2_done = own_conv2 & shared_conv_done;

    logic [4:0] shared_cur_oc, shared_cur_row, shared_cur_col;
    logic [7:0] shared_mac_dbg;
    logic shared_eng_done_dbg;

    shared_conv_engine #(
        .CONV1_WGT_MEM(CONV1_WGT_MEM),
        .CONV1_BIAS_MEM(CONV1_BIAS_MEM),
        .CONV1_MULT_MEM(CONV1_MULT_MEM),
        .CONV1_SHIFT_MEM(CONV1_SHIFT_MEM),
        .CONV2_WGT_MEM(CONV2_WGT_MEM),
        .CONV2_BIAS_MEM(CONV2_BIAS_MEM),
        .CONV2_MULT_MEM(CONV2_MULT_MEM),
        .CONV2_SHIFT_MEM(CONV2_SHIFT_MEM)
    ) u_shared_conv (
        .clk(clk), .rst(rst),
        .start(shared_conv_start),
        .layer_is_conv2(shared_layer_is_conv2),
        .external_act_data(shared_act_rdata),
        .busy(shared_conv_busy),
        .done(shared_conv_done),
        .output_count(shared_out_count),
        .current_output_channel(shared_cur_oc),
        .current_output_row(shared_cur_row),
        .current_output_column(shared_cur_col),
        .dbg_layer_is_conv2(shared_dbg_layer),
        .act_read_enable(shared_act_re),
        .act_read_address(shared_act_addr),
        .output_write_enable(shared_we),
        .output_write_address(shared_waddr),
        .output_write_data(shared_wdata),
        .dbg_engine_mac_count(shared_mac_dbg),
        .dbg_engine_done(shared_eng_done_dbg)
    );

    // Capture output counts at layer done for debug ports
    logic [14:0] conv1_count_r;
    logic [13:0] conv2_count_r;
    always_ff @(posedge clk) begin
        if (rst) begin
            conv1_count_r <= 15'd0;
            conv2_count_r <= 14'd0;
        end else begin
            if (own_conv1 && shared_conv_done)
                conv1_count_r <= shared_out_count;
            if (own_conv2 && shared_conv_done)
                conv2_count_r <= shared_out_count[13:0];
        end
    end
    assign dbg_conv1_output_count = conv1_count_r;
    assign dbg_conv2_output_count = conv2_count_r;

    // -------------------------------------------------------------------------
    // MaxPool1
    // -------------------------------------------------------------------------
    logic signed [7:0] p1_a, p1_b, p1_c, p1_d, p1_max;
    logic [3:0] p1_ch, p1_pr, p1_pc;
    logic [1:0] p1_win;

    maxpool1_controller #(.NUM_CHANNELS(16)) u_pool1_ctrl (
        .clk(clk), .rst(rst), .start(pool1_start),
        .busy(pool1_busy), .pool1_done(pool1_done),
        .conv1_read_enable(pool1_re),
        .conv1_read_address(pool1_raddr),
        .conv1_read_data(pool1_rdata),
        .pool1_write_enable(pool1_we),
        .pool1_write_address(pool1_waddr),
        .pool1_write_data(pool1_wdata),
        .current_channel(p1_ch),
        .current_pool_row(p1_pr),
        .current_pool_column(p1_pc),
        .current_window_index(p1_win),
        .output_count(dbg_pool1_output_count),
        .value_a(p1_a), .value_b(p1_b), .value_c(p1_c), .value_d(p1_d),
        .maximum_value(p1_max)
    );

    // -------------------------------------------------------------------------
    // MaxPool2
    // -------------------------------------------------------------------------
    logic signed [7:0] p2_a, p2_b, p2_c, p2_d, p2_max;
    logic [4:0] p2_ch;
    logic [2:0] p2_pr, p2_pc;
    logic [1:0] p2_win;
    logic [13:0] p2_reads;

    maxpool2_controller #(.NUM_CHANNELS(32)) u_pool2_ctrl (
        .clk(clk), .rst(rst), .start(pool2_start),
        .busy(pool2_busy), .pool2_done(pool2_done),
        .conv2_read_enable(pool2_re),
        .conv2_read_address(pool2_raddr),
        .conv2_read_data(pool2_rdata),
        .pool2_write_enable(pool2_we),
        .pool2_write_address(pool2_waddr),
        .pool2_write_data(pool2_wdata),
        .current_channel(p2_ch),
        .current_pool_row(p2_pr),
        .current_pool_column(p2_pc),
        .current_window_index(p2_win),
        .output_count(dbg_pool2_output_count),
        .read_count(p2_reads),
        .value_a(p2_a), .value_b(p2_b), .value_c(p2_c), .value_d(p2_d),
        .maximum_value(p2_max)
    );

    // -------------------------------------------------------------------------
    // GAP
    // -------------------------------------------------------------------------
    logic [11:0] gap_reads;
    logic [4:0] gap_ch;
    logic [5:0] gap_elem;
    logic signed [31:0] gap_sum, gap_final;
    logic signed [7:0] gap_avg, gap_int8;

    global_average_pool_controller #(
        .NUM_CHANNELS(32),
        .GAP_MULTIPLIER(GAP_MULTIPLIER),
        .GAP_SHIFT(GAP_SHIFT)
    ) u_gap_ctrl (
        .clk(clk), .rst(rst), .start(gap_start),
        .busy(gap_busy), .gap_done(gap_done),
        .pool2_read_enable(gap_act_re),
        .pool2_read_address(gap_act_raddr),
        .pool2_read_data(gap_act_rdata),
        .gap_write_enable(gap_we),
        .gap_write_address(gap_waddr),
        .gap_write_data(gap_wdata),
        .current_channel(gap_ch),
        .current_element(gap_elem),
        .running_sum(gap_sum),
        .read_count(gap_reads),
        .output_count(dbg_gap_output_count),
        .final_sum(gap_final),
        .averaged_int8(gap_avg),
        .gap_int8(gap_int8)
    );

    // -------------------------------------------------------------------------
    // FC
    // -------------------------------------------------------------------------
    logic eng_fc_start, eng_fc_busy, eng_fc_done, eng_fc_logit_valid;
    logic [2:0] eng_fc_class;
    logic signed [31:0] eng_fc_logit, eng_fc_acc, eng_fc_run;
    logic [5:0] eng_fc_macs;
    logic [4:0] eng_fc_in;
    logic signed [15:0] eng_fc_prod;
    logic [2:0] fc_class_idx;
    logic [3:0] fc_class_count;

    logic fc_wgt_re, fc_bias_re, fc_mult_re, fc_shift_re;
    logic [7:0] fc_wgt_addr;
    logic [2:0] fc_bias_addr, fc_mult_addr, fc_shift_addr;
    logic signed [7:0] fc_wgt_data;
    logic signed [31:0] fc_bias_data, fc_mult_data, fc_shift_data;

    int8_sync_rom #(.DEPTH(160), .ADDR_WIDTH(8), .MEM_FILE(FC_WGT_MEM)) u_fc_wgt (
        .clk(clk), .read_enable(fc_wgt_re), .read_address(fc_wgt_addr), .read_data(fc_wgt_data)
    );
    int32_sync_rom #(.DEPTH(5), .ADDR_WIDTH(3), .MEM_FILE(FC_BIAS_MEM)) u_fc_bias (
        .clk(clk), .read_enable(fc_bias_re), .read_address(fc_bias_addr), .read_data(fc_bias_data)
    );
    int32_sync_rom #(.DEPTH(5), .ADDR_WIDTH(3), .MEM_FILE(FC_MULT_MEM)) u_fc_mult (
        .clk(clk), .read_enable(fc_mult_re), .read_address(fc_mult_addr), .read_data(fc_mult_data)
    );
    int32_sync_rom #(.DEPTH(5), .ADDR_WIDTH(3), .MEM_FILE(FC_SHIFT_MEM)) u_fc_shift (
        .clk(clk), .read_enable(fc_shift_re), .read_address(fc_shift_addr), .read_data(fc_shift_data)
    );

    fully_connected_class_engine #(.NUM_FEATURES(32)) u_fc_eng (
        .clk(clk), .rst(rst), .start(eng_fc_start),
        .class_index(eng_fc_class),
        .busy(eng_fc_busy), .done(eng_fc_done),
        .logit_valid(eng_fc_logit_valid),
        .logit_value(eng_fc_logit),
        .final_accumulator(eng_fc_acc),
        .mac_count(eng_fc_macs),
        .debug_input_index(eng_fc_in),
        .debug_product(eng_fc_prod),
        .debug_accumulator(eng_fc_run),
        .gap_read_enable(gap_re_hw),
        .gap_read_address(gap_raddr_hw),
        .gap_read_data(gap_rdata_w),
        .weight_read_enable(fc_wgt_re),
        .weight_read_address(fc_wgt_addr),
        .weight_read_data(fc_wgt_data),
        .bias_read_enable(fc_bias_re),
        .bias_read_address(fc_bias_addr),
        .bias_read_data(fc_bias_data),
        .mult_read_enable(fc_mult_re),
        .mult_read_address(fc_mult_addr),
        .mult_read_data(fc_mult_data),
        .shift_read_enable(fc_shift_re),
        .shift_read_address(fc_shift_addr),
        .shift_read_data(fc_shift_data)
    );

    fully_connected_layer_controller #(
        .NUM_CLASSES(5), .NUM_FEATURES(32)
    ) u_fc_ctrl (
        .clk(clk), .rst(rst), .start(fc_start),
        .busy(fc_busy), .fc_done(fc_done),
        .class_index(fc_class_idx),
        .class_count(fc_class_count),
        .total_mac_count(dbg_fc_mac_count),
        .engine_start(eng_fc_start),
        .engine_class_index(eng_fc_class),
        .engine_busy(eng_fc_busy),
        .engine_done(eng_fc_done),
        .engine_logit(eng_fc_logit),
        .engine_accumulator(eng_fc_acc),
        .engine_mac_count(eng_fc_macs),
        .logit_write_enable(logit_we),
        .logit_write_address(logit_waddr),
        .logit_write_data(logit_wdata)
    );

    logic signed [LOGIT_WIDTH-1:0] logit_r [0:4];
    always_ff @(posedge clk) begin
        if (rst) begin
            logit_r[0] <= '0;
            logit_r[1] <= '0;
            logit_r[2] <= '0;
            logit_r[3] <= '0;
            logit_r[4] <= '0;
        end else if (logit_we) begin
            logit_r[logit_waddr] <= logit_wdata;
        end
    end
    assign logit_0 = logit_r[0];
    assign logit_1 = logit_r[1];
    assign logit_2 = logit_r[2];
    assign logit_3 = logit_r[3];
    assign logit_4 = logit_r[4];

    // -------------------------------------------------------------------------
    // Argmax
    // -------------------------------------------------------------------------
    logic [2:0] argmax_idx;
    logic [3:0] argmax_reads;
    logic [2:0] pred_w;
    logic signed [LOGIT_WIDTH-1:0] max_w;

    signed_argmax5_controller #(.LOGIT_WIDTH(LOGIT_WIDTH)) u_argmax (
        .clk(clk), .rst(rst), .start(argmax_start),
        .busy(argmax_busy), .done(argmax_done),
        .logit_read_enable(logit_re_hw),
        .logit_read_address(logit_raddr_hw),
        .logit_read_data(logit_rdata_w),
        .predicted_class(pred_w),
        .maximum_logit(max_w),
        .current_index(argmax_idx),
        .read_count(argmax_reads)
    );

    assign predicted_class = pred_w;
    assign maximum_logit   = max_w;

    // -------------------------------------------------------------------------
    // Ownership / range assertions (simulation only)
    // -------------------------------------------------------------------------
    // synopsys translate_off
    always_ff @(posedge clk) begin
        if (!rst) begin
            // Single writer per RAM
            if (own_pool1 && own_pool2)
                $error("RAM A dual writers: pool1 and pool2");
            if (own_conv1 && own_conv2)
                $error("RAM B dual writers: conv1 and conv2");

            if (act_a_we && busy && !(own_pool1 || own_pool2))
                $error("RAM A write while not owned by pool1/pool2");
            if (act_b_we && !(own_conv1 || own_conv2))
                $error("RAM B write while not owned by conv1/conv2");
            if (busy && input_write_enable)
                $error("input write while accelerator busy");

            // Address ranges for writes during owned stages
            // Widen comparisons: 14'd16384 truncates to 0 in 14 bits.
            if (own_conv1 && shared_we && ({1'b0, shared_waddr} >= 15'd16384))
                $error("Conv1 write OOB");
            if (own_pool1 && pool1_we && ({2'b0, pool1_waddr} >= 14'd4096))
                $error("Pool1 write OOB");
            if (own_conv2 && shared_we && (shared_waddr >= 14'd8192))
                $error("Conv2 write OOB");
            if (own_pool2 && pool2_we && ({3'b0, pool2_waddr} >= 14'd2048))
                $error("Pool2 write OOB");

            // Read ranges (padded Conv reads may skip RAM; when enable is high)
            if (own_conv1 && shared_act_re && ({2'b0, shared_act_addr} >= 14'd3072))
                $error("Conv1 read OOB %0d", shared_act_addr);
            if (own_pool1 && pool1_re && ({1'b0, pool1_raddr} >= 15'd16384))
                $error("Pool1 read OOB");
            if (own_conv2 && shared_act_re && ({2'b0, shared_act_addr} >= 14'd4096))
                $error("Conv2 read OOB %0d", shared_act_addr);
            if (own_pool2 && pool2_re && ({1'b0, pool2_raddr} >= 14'd8192))
                $error("Pool2 read OOB");
            if (own_gap && gap_act_re && ({3'b0, gap_act_raddr} >= 14'd2048))
                $error("GAP read OOB");

            // GAP must not write activation RAMs
            if (own_gap && (act_a_we || act_b_we))
                $error("GAP wrote activation RAM");
        end
    end
    // synopsys translate_on

    /* verilator lint_off UNUSED */
    wire unused_dbg =
        |{p1_a, p1_b, p1_c, p1_d, p1_max, p1_ch, p1_pr, p1_pc, p1_win,
          p2_a, p2_b, p2_c, p2_d, p2_max, p2_ch, p2_pr, p2_pc, p2_win, p2_reads,
          gap_reads, gap_ch, gap_elem, gap_sum, gap_final, gap_avg, gap_int8,
          eng_fc_logit_valid, eng_fc_run, eng_fc_in, eng_fc_prod,
          fc_class_idx, fc_class_count, argmax_idx, argmax_reads,
          conv1_start, pool1_start, conv2_start, pool2_start,
          gap_start, fc_start, argmax_start, shared_dbg_layer,
          shared_cur_oc, shared_cur_row, shared_cur_col,
          shared_mac_dbg, shared_eng_done_dbg};
    /* verilator lint_on UNUSED */

endmodule
