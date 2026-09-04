// cnn_accelerator_top.sv
// End-to-end CNN accelerator: shared stage-specific activation memories.
//
// Pipeline (trained sizes; obsolete 8/16 prompt sketches are not used):
//   Input 3x32x32
//     -> Conv1/ReLU 16x32x32
//     -> MaxPool1 16x16x16
//     -> Conv2/ReLU 32x16x16
//     -> MaxPool2 32x8x8
//     -> GAP 32
//     -> FC 5 INT32 logits
//     -> Argmax predicted_class[2:0]
//
// No ping-pong in this phase. Weight / bias / quant ROMs stay in engines.
// Standalone stage tops remain unchanged for unit tests.

`timescale 1ns / 1ps

module cnn_accelerator_top #(
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

    // Host / TB: load input image when idle
    input  logic                           input_write_enable,
    input  logic [11:0]                    input_write_address,
    input  logic signed [7:0]              input_write_data,

    // Host / TB: readback when idle (after done)
    input  logic                           conv1_read_enable,
    input  logic [13:0]                    conv1_read_address,
    output logic signed [7:0]              conv1_read_data,

    input  logic                           pool1_read_enable,
    input  logic [11:0]                    pool1_read_address,
    output logic signed [7:0]              pool1_read_data,

    input  logic                           conv2_read_enable,
    input  logic [12:0]                    conv2_read_address,
    output logic signed [7:0]              conv2_read_data,

    input  logic                           pool2_read_enable,
    input  logic [10:0]                    pool2_read_address,
    output logic signed [7:0]              pool2_read_data,

    input  logic                           gap_read_enable,
    input  logic [4:0]                     gap_read_address,
    output logic signed [7:0]              gap_read_data,

    input  logic                           logit_read_enable,
    input  logic [2:0]                     logit_read_address,
    output logic signed [LOGIT_WIDTH-1:0]  logit_read_data
);

    // -------------------------------------------------------------------------
    // Top controller
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

    // -------------------------------------------------------------------------
    // Shared activation memories
    // -------------------------------------------------------------------------
    logic input_re_hw;
    logic [11:0] input_raddr_hw;
    logic signed [7:0] input_rdata;

    logic conv1_we;
    logic [13:0] conv1_waddr;
    logic signed [7:0] conv1_wdata;
    logic conv1_re_hw;
    logic [13:0] conv1_raddr_hw;
    logic signed [7:0] conv1_rdata_w;

    logic pool1_we;
    logic [11:0] pool1_waddr;
    logic signed [7:0] pool1_wdata;
    logic pool1_re_hw;
    logic [11:0] pool1_raddr_hw;
    logic signed [7:0] pool1_rdata_w;

    logic conv2_we;
    logic [12:0] conv2_waddr;
    logic signed [7:0] conv2_wdata;
    logic conv2_re_hw;
    logic [12:0] conv2_raddr_hw;
    logic signed [7:0] conv2_rdata_w;

    logic pool2_we;
    logic [10:0] pool2_waddr;
    logic signed [7:0] pool2_wdata;
    logic pool2_re_hw;
    logic [10:0] pool2_raddr_hw;
    logic signed [7:0] pool2_rdata_w;

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

    // Ownership mux: hardware while busy, TB when idle
    logic input_we_mux;
    logic input_re_mux;
    logic [11:0] input_raddr_mux;
    assign input_we_mux   = (~busy) & input_write_enable;
    assign input_re_mux   = busy ? input_re_hw : 1'b0;
    assign input_raddr_mux= input_raddr_hw;

    logic conv1_re_mux;
    logic [13:0] conv1_raddr_mux;
    assign conv1_re_mux    = busy ? conv1_re_hw : conv1_read_enable;
    assign conv1_raddr_mux = busy ? conv1_raddr_hw : conv1_read_address;
    assign conv1_read_data = conv1_rdata_w;

    logic pool1_re_mux;
    logic [11:0] pool1_raddr_mux;
    assign pool1_re_mux    = busy ? pool1_re_hw : pool1_read_enable;
    assign pool1_raddr_mux = busy ? pool1_raddr_hw : pool1_read_address;
    assign pool1_read_data = pool1_rdata_w;

    logic conv2_re_mux;
    logic [12:0] conv2_raddr_mux;
    assign conv2_re_mux    = busy ? conv2_re_hw : conv2_read_enable;
    assign conv2_raddr_mux = busy ? conv2_raddr_hw : conv2_read_address;
    assign conv2_read_data = conv2_rdata_w;

    logic pool2_re_mux;
    logic [10:0] pool2_raddr_mux;
    assign pool2_re_mux    = busy ? pool2_re_hw : pool2_read_enable;
    assign pool2_raddr_mux = busy ? pool2_raddr_hw : pool2_read_address;
    assign pool2_read_data = pool2_rdata_w;

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

    int8_sync_ram #(.DEPTH(3072), .ADDR_WIDTH(12)) u_input_ram (
        .clk(clk),
        .write_enable(input_we_mux),
        .write_address(input_write_address),
        .write_data(input_write_data),
        .read_enable(input_re_mux),
        .read_address(input_raddr_mux),
        .read_data(input_rdata)
    );

    int8_sync_ram #(.DEPTH(16384), .ADDR_WIDTH(14)) u_conv1_ram (
        .clk(clk),
        .write_enable(conv1_we),
        .write_address(conv1_waddr),
        .write_data(conv1_wdata),
        .read_enable(conv1_re_mux),
        .read_address(conv1_raddr_mux),
        .read_data(conv1_rdata_w)
    );

    int8_sync_ram #(.DEPTH(4096), .ADDR_WIDTH(12)) u_pool1_ram (
        .clk(clk),
        .write_enable(pool1_we),
        .write_address(pool1_waddr),
        .write_data(pool1_wdata),
        .read_enable(pool1_re_mux),
        .read_address(pool1_raddr_mux),
        .read_data(pool1_rdata_w)
    );

    int8_sync_ram #(.DEPTH(8192), .ADDR_WIDTH(13)) u_conv2_ram (
        .clk(clk),
        .write_enable(conv2_we),
        .write_address(conv2_waddr),
        .write_data(conv2_wdata),
        .read_enable(conv2_re_mux),
        .read_address(conv2_raddr_mux),
        .read_data(conv2_rdata_w)
    );

    int8_sync_ram #(.DEPTH(2048), .ADDR_WIDTH(11)) u_pool2_ram (
        .clk(clk),
        .write_enable(pool2_we),
        .write_address(pool2_waddr),
        .write_data(pool2_wdata),
        .read_enable(pool2_re_mux),
        .read_address(pool2_raddr_mux),
        .read_data(pool2_rdata_w)
    );

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
    // Conv1: controller + EXTERNAL_ACT engine
    // -------------------------------------------------------------------------
    logic eng1_start, eng1_busy, eng1_done;
    logic [3:0] eng1_oc;
    logic [4:0] eng1_row, eng1_col;
    logic signed [7:0] eng1_relu;
    logic [4:0] eng1_macs;
    logic eng1_act_re;
    logic [11:0] eng1_act_addr;
    logic signed [31:0] eng1_final_acc;
    logic signed [7:0] eng1_rq;
    logic [1:0] eng1_ic, eng1_kr, eng1_kc;
    logic [8:0] eng1_wgt_addr;
    logic [3:0] eng1_bias_addr;
    logic eng1_pad, eng1_op_valid, eng1_mac_en, eng1_load_bias;
    logic signed [7:0] eng1_act_v, eng1_wgt_v;
    logic signed [15:0] eng1_prod, eng1_last_prod;
    logic signed [31:0] eng1_acc;
    logic eng1_wgt_re;
    logic [3:0] c1_cur_oc;
    logic [4:0] c1_cur_row, c1_cur_col;
    logic [4:0] c2_cur_oc;
    logic [3:0] c2_cur_row, c2_cur_col;

    assign input_re_hw    = eng1_act_re;
    assign input_raddr_hw = eng1_act_addr;

    conv1_layer_controller #(.NUM_OUT_CHANNELS(16)) u_conv1_ctrl (
        .clk(clk), .rst(rst), .start(conv1_start),
        .engine_start(eng1_start),
        .engine_output_channel(eng1_oc),
        .engine_output_row(eng1_row),
        .engine_output_column(eng1_col),
        .engine_busy(eng1_busy),
        .engine_done(eng1_done),
        .engine_relu_output(eng1_relu),
        .engine_mac_count(eng1_macs),
        .busy(conv1_busy),
        .conv1_done(conv1_done),
        .output_count(dbg_conv1_output_count),
        .current_output_channel(c1_cur_oc),
        .current_output_row(c1_cur_row),
        .current_output_column(c1_cur_col),
        .output_write_enable(conv1_we),
        .output_write_address(conv1_waddr),
        .output_write_data(conv1_wdata)
    );

    conv1_memory_single_output #(
        .EXTERNAL_ACT(1),
        .WGT_MEM_FILE(CONV1_WGT_MEM),
        .BIAS_MEM_FILE(CONV1_BIAS_MEM),
        .MULT_MEM_FILE(CONV1_MULT_MEM),
        .SHIFT_MEM_FILE(CONV1_SHIFT_MEM)
    ) u_conv1_eng (
        .clk(clk), .rst(rst), .start(eng1_start),
        .output_channel(eng1_oc),
        .output_row(eng1_row),
        .output_column(eng1_col),
        .external_act_data(input_rdata),
        .busy(eng1_busy), .done(eng1_done),
        .final_accumulator(eng1_final_acc),
        .requantized_output(eng1_rq),
        .relu_output(eng1_relu),
        .mac_count(eng1_macs),
        .current_input_channel(eng1_ic),
        .current_kernel_row(eng1_kr),
        .current_kernel_column(eng1_kc),
        .current_activation_address(eng1_act_addr),
        .current_weight_address(eng1_wgt_addr),
        .current_bias_address(eng1_bias_addr),
        .current_padding(eng1_pad),
        .current_activation_value(eng1_act_v),
        .current_weight_value(eng1_wgt_v),
        .current_product(eng1_prod),
        .current_accumulator(eng1_acc),
        .operand_valid(eng1_op_valid),
        .mac_enable(eng1_mac_en),
        .load_bias(eng1_load_bias),
        .act_read_enable(eng1_act_re),
        .wgt_read_enable(eng1_wgt_re),
        .last_product(eng1_last_prod)
    );

    // -------------------------------------------------------------------------
    // MaxPool1
    // -------------------------------------------------------------------------
    logic signed [7:0] p1_a, p1_b, p1_c, p1_d, p1_max;
    logic [3:0] p1_ch, p1_pr, p1_pc;
    logic [1:0] p1_win;

    maxpool1_controller #(.NUM_CHANNELS(16)) u_pool1_ctrl (
        .clk(clk), .rst(rst), .start(pool1_start),
        .busy(pool1_busy), .pool1_done(pool1_done),
        .conv1_read_enable(conv1_re_hw),
        .conv1_read_address(conv1_raddr_hw),
        .conv1_read_data(conv1_rdata_w),
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
    // Conv2: EXTERNAL_ACT from pool1 RAM
    // -------------------------------------------------------------------------
    logic eng2_start, eng2_busy, eng2_done;
    logic [4:0] eng2_oc;
    logic [3:0] eng2_row, eng2_col;
    logic signed [7:0] eng2_relu;
    logic [7:0] eng2_macs;
    logic eng2_act_re;
    logic [11:0] eng2_act_addr;
    logic signed [31:0] eng2_final_acc;
    logic signed [7:0] eng2_rq;
    logic [3:0] eng2_ic;
    logic [1:0] eng2_kr, eng2_kc;
    logic [12:0] eng2_wgt_addr, eng2_out_addr;
    logic [4:0] eng2_bias_addr;
    logic eng2_pad, eng2_op_valid, eng2_mac_en, eng2_load_bias;
    logic signed [7:0] eng2_act_v, eng2_wgt_v;
    logic signed [15:0] eng2_prod, eng2_last_prod;
    logic signed [31:0] eng2_acc;
    logic eng2_wgt_re;

    assign pool1_re_hw    = eng2_act_re;
    assign pool1_raddr_hw = eng2_act_addr;

    conv2_layer_controller #(.NUM_OUT_CHANNELS(32)) u_conv2_ctrl (
        .clk(clk), .rst(rst), .start(conv2_start),
        .engine_start(eng2_start),
        .engine_output_channel(eng2_oc),
        .engine_output_row(eng2_row),
        .engine_output_column(eng2_col),
        .engine_busy(eng2_busy),
        .engine_done(eng2_done),
        .engine_relu_output(eng2_relu),
        .engine_mac_count(eng2_macs),
        .busy(conv2_busy),
        .conv2_done(conv2_done),
        .output_count(dbg_conv2_output_count),
        .current_output_channel(c2_cur_oc),
        .current_output_row(c2_cur_row),
        .current_output_column(c2_cur_col),
        .output_write_enable(conv2_we),
        .output_write_address(conv2_waddr),
        .output_write_data(conv2_wdata)
    );

    conv2_memory_single_output #(
        .EXTERNAL_ACT(1),
        .WGT_MEM_FILE(CONV2_WGT_MEM),
        .BIAS_MEM_FILE(CONV2_BIAS_MEM),
        .MULT_MEM_FILE(CONV2_MULT_MEM),
        .SHIFT_MEM_FILE(CONV2_SHIFT_MEM)
    ) u_conv2_eng (
        .clk(clk), .rst(rst), .start(eng2_start),
        .output_channel(eng2_oc),
        .output_row(eng2_row),
        .output_column(eng2_col),
        .external_act_data(pool1_rdata_w),
        .busy(eng2_busy), .done(eng2_done),
        .final_accumulator(eng2_final_acc),
        .requantized_output(eng2_rq),
        .relu_output(eng2_relu),
        .mac_count(eng2_macs),
        .current_input_channel(eng2_ic),
        .current_kernel_row(eng2_kr),
        .current_kernel_column(eng2_kc),
        .current_activation_address(eng2_act_addr),
        .current_weight_address(eng2_wgt_addr),
        .current_bias_address(eng2_bias_addr),
        .current_output_address(eng2_out_addr),
        .current_padding(eng2_pad),
        .current_activation_value(eng2_act_v),
        .current_weight_value(eng2_wgt_v),
        .current_product(eng2_prod),
        .current_accumulator(eng2_acc),
        .operand_valid(eng2_op_valid),
        .mac_enable(eng2_mac_en),
        .load_bias(eng2_load_bias),
        .act_read_enable(eng2_act_re),
        .wgt_read_enable(eng2_wgt_re),
        .last_product(eng2_last_prod)
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
        .conv2_read_enable(conv2_re_hw),
        .conv2_read_address(conv2_raddr_hw),
        .conv2_read_data(conv2_rdata_w),
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
        .pool2_read_enable(pool2_re_hw),
        .pool2_read_address(pool2_raddr_hw),
        .pool2_read_data(pool2_rdata_w),
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

    // Shadow registers for parallel logit outputs
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

    // Silence unused debug nets
    /* verilator lint_off UNUSED */
    wire unused_dbg =
        |{eng1_final_acc, eng1_rq, eng1_ic, eng1_kr, eng1_kc, eng1_wgt_addr,
          eng1_bias_addr, eng1_pad, eng1_op_valid, eng1_mac_en, eng1_load_bias,
          eng1_act_v, eng1_wgt_v, eng1_prod, eng1_acc, eng1_last_prod, eng1_wgt_re,
          p1_a, p1_b, p1_c, p1_d, p1_max, p1_ch, p1_pr, p1_pc, p1_win,
          eng2_final_acc, eng2_rq, eng2_ic, eng2_kr, eng2_kc, eng2_wgt_addr,
          eng2_out_addr, eng2_bias_addr, eng2_pad, eng2_op_valid, eng2_mac_en,
          eng2_load_bias, eng2_act_v, eng2_wgt_v, eng2_prod, eng2_acc,
          eng2_last_prod, eng2_wgt_re,
          p2_a, p2_b, p2_c, p2_d, p2_max, p2_ch, p2_pr, p2_pc, p2_win, p2_reads,
          gap_reads, gap_ch, gap_elem, gap_sum, gap_final, gap_avg, gap_int8,
          eng_fc_logit_valid, eng_fc_run, eng_fc_in, eng_fc_prod,
          fc_class_idx, fc_class_count, argmax_idx, argmax_reads,
          c1_cur_oc, c1_cur_row, c1_cur_col, c2_cur_oc, c2_cur_row, c2_cur_col};
    /* verilator lint_on UNUSED */

endmodule
