// maxpool1_controller.sv
// MaxPool1 over full Conv1 tensor (trained: 16 x 32 x 32 -> 16 x 16 x 16).
//
// For each (channel, pool_row, pool_column):
//   sequentially read 4 Conv1 values (1-cycle sync ROM/RAM latency)
//   signed max-of-four
//   write one Pool1 INT8 (no requant, no ReLU)
//
// Column fastest, then row, then channel.
//
// FSM (per window value):
//   ISSUE -> WAIT -> CAPTURE   (repeat for window_index 0..3)
// then COMPARE -> WRITE -> ADVANCE -> (next | DONE)

`timescale 1ns / 1ps

module maxpool1_controller #(
    parameter int NUM_CHANNELS = 16
) (
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    start,

    output logic                    busy,
    output logic                    pool1_done,

    output logic                    conv1_read_enable,
    output logic [13:0]             conv1_read_address,
    input  logic signed [7:0]       conv1_read_data,

    output logic                    pool1_write_enable,
    output logic [11:0]             pool1_write_address,
    output logic signed [7:0]       pool1_write_data,

    output logic [3:0]              current_channel,
    output logic [3:0]              current_pool_row,
    output logic [3:0]              current_pool_column,
    output logic [1:0]              current_window_index,
    output logic [12:0]             output_count, // 0 .. 4096

    // Debug captures
    output logic signed [7:0]       value_a,
    output logic signed [7:0]       value_b,
    output logic signed [7:0]       value_c,
    output logic signed [7:0]       value_d,
    output logic signed [7:0]       maximum_value
);

    localparam logic [3:0] LAST_CH = 4'd15;
    localparam logic [12:0] TOTAL_COUNT = 13'd4096;

    localparam logic [3:0] ST_IDLE     = 4'd0;
    localparam logic [3:0] ST_ISSUE    = 4'd1;
    localparam logic [3:0] ST_WAIT     = 4'd2;
    localparam logic [3:0] ST_CAPTURE  = 4'd3;
    localparam logic [3:0] ST_COMPARE  = 4'd4;
    localparam logic [3:0] ST_WRITE    = 4'd5;
    localparam logic [3:0] ST_ADVANCE  = 4'd6;
    localparam logic [3:0] ST_DONE     = 4'd7;

    logic [3:0] state;
    logic [3:0] state_n;

    logic [3:0] ch_r;
    logic [3:0] pr_r;
    logic [3:0] pc_r;
    logic [1:0] win_r;
    logic [12:0] count_r;

    logic signed [7:0] a_r, b_r, c_r, d_r;
    logic signed [7:0] max_w;

    logic [4:0] in_row_w;
    logic [4:0] in_col_w;
    logic [13:0] conv_addr_w;
    logic [11:0] pool_addr_w;
    logic last_coord;

    // Coordinate outputs kept for waveform / assertion visibility.
    /* verilator lint_off UNUSED */
    wire [4:0] dbg_input_row = in_row_w;
    wire [4:0] dbg_input_col = in_col_w;
    /* verilator lint_on UNUSED */

    maxpool2x2_address_generator u_agen (
        .channel             (ch_r),
        .pool_row            (pr_r),
        .pool_column         (pc_r),
        .window_index        (win_r),
        .input_row           (in_row_w),
        .input_column        (in_col_w),
        .conv1_read_address  (conv_addr_w),
        .pool1_write_address (pool_addr_w)
    );

    max4_int8 u_max4 (
        .value_a (a_r),
        .value_b (b_r),
        .value_c (c_r),
        .value_d (d_r),
        .maximum (max_w)
    );

    assign current_channel        = ch_r;
    assign current_pool_row       = pr_r;
    assign current_pool_column    = pc_r;
    assign current_window_index   = win_r;
    assign output_count           = count_r;
    assign conv1_read_address     = conv_addr_w;
    assign pool1_write_address    = pool_addr_w;
    assign pool1_write_data       = max_w;
    assign value_a = a_r;
    assign value_b = b_r;
    assign value_c = c_r;
    assign value_d = d_r;
    assign maximum_value = max_w;

    assign conv1_read_enable  = (state == ST_ISSUE);
    assign pool1_write_enable = (state == ST_WRITE);
    assign last_coord =
        (ch_r == LAST_CH) && (pr_r == 4'd15) && (pc_r == 4'd15);

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= ST_IDLE;
            ch_r <= 4'd0;
            pr_r <= 4'd0;
            pc_r <= 4'd0;
            win_r <= 2'd0;
            count_r <= 13'd0;
            a_r <= 8'sd0;
            b_r <= 8'sd0;
            c_r <= 8'sd0;
            d_r <= 8'sd0;
            busy <= 1'b0;
            pool1_done <= 1'b0;
        end else begin
            state <= state_n;
            pool1_done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        ch_r <= 4'd0;
                        pr_r <= 4'd0;
                        pc_r <= 4'd0;
                        win_r <= 2'd0;
                        count_r <= 13'd0;
                    end
                end
                ST_CAPTURE: begin
                    // ROM data was updated on the posedge entering WAIT;
                    // capture here one full cycle later (NBA-safe).
                    unique case (win_r)
                        2'd0: a_r <= conv1_read_data;
                        2'd1: b_r <= conv1_read_data;
                        2'd2: c_r <= conv1_read_data;
                        default: d_r <= conv1_read_data;
                    endcase
                    if (win_r != 2'd3)
                        win_r <= win_r + 2'd1;
                end
                ST_WRITE: begin
                    count_r <= count_r + 13'd1;
                end
                ST_ADVANCE: begin
                    win_r <= 2'd0;
                    if (!last_coord) begin
                        if (pc_r == 4'd15) begin
                            pc_r <= 4'd0;
                            if (pr_r == 4'd15) begin
                                pr_r <= 4'd0;
                                ch_r <= ch_r + 4'd1;
                            end else begin
                                pr_r <= pr_r + 4'd1;
                            end
                        end else begin
                            pc_r <= pc_r + 4'd1;
                        end
                    end
                end
                ST_DONE: begin
                    pool1_done <= 1'b1;
                    busy <= 1'b0;
                end
                default: ;
            endcase
        end
    end

    always_comb begin
        state_n = state;
        case (state)
            ST_IDLE: begin
                if (start)
                    state_n = ST_ISSUE;
            end
            ST_ISSUE: begin
                state_n = ST_WAIT;
            end
            ST_WAIT: begin
                state_n = ST_CAPTURE;
            end
            ST_CAPTURE: begin
                if (win_r == 2'd3)
                    state_n = ST_COMPARE;
                else
                    state_n = ST_ISSUE;
            end
            ST_COMPARE: begin
                state_n = ST_WRITE;
            end
            ST_WRITE: begin
                state_n = ST_ADVANCE;
            end
            ST_ADVANCE: begin
                if (last_coord)
                    state_n = ST_DONE;
                else
                    state_n = ST_ISSUE;
            end
            ST_DONE: begin
                state_n = ST_IDLE;
            end
            default: state_n = ST_IDLE;
        endcase
    end

    // synopsys translate_off
    always_ff @(posedge clk) begin
        if (!rst) begin
            if (conv1_read_enable &&
                ({1'b0, conv1_read_address} >= 15'd16384))
                $error("conv1 read addr OOB");
            if (pool1_write_enable &&
                ({1'b0, pool1_write_address} >= 13'd4096))
                $error("pool1 write addr OOB");
            if (count_r > TOTAL_COUNT)
                $error("output_count exceeded");
            if (pool1_done && (count_r != TOTAL_COUNT))
                $error("pool1_done with bad count");
            if (pool1_done &&
                ((ch_r != LAST_CH) || (pr_r != 4'd15) || (pc_r != 4'd15)))
                $error("pool1_done with wrong final coordinate");
        end
    end
    // synopsys translate_on

endmodule
