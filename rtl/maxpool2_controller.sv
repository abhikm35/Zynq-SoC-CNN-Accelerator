// maxpool2_controller.sv
// MaxPool2 over full Conv2 tensor (trained: 32 x 16 x 16 -> 32 x 8 x 8).
//
// Same FSM timing as MaxPool1 (reuse max4_int8):
//   ISSUE -> WAIT -> CAPTURE   (window_index 0..3)
//   COMPARE -> WRITE -> ADVANCE -> (next | DONE)
//
// Column fastest, then row, then channel.
// No requantization, no ReLU — signed INT8 max only.

`timescale 1ns / 1ps

module maxpool2_controller #(
    parameter int NUM_CHANNELS = 32
) (
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    start,

    output logic                    busy,
    output logic                    pool2_done,

    output logic                    conv2_read_enable,
    output logic [12:0]             conv2_read_address,
    input  logic signed [7:0]       conv2_read_data,

    output logic                    pool2_write_enable,
    output logic [10:0]             pool2_write_address,
    output logic signed [7:0]       pool2_write_data,

    output logic [4:0]              current_channel,
    output logic [2:0]              current_pool_row,
    output logic [2:0]              current_pool_column,
    output logic [1:0]              current_window_index,
    output logic [11:0]             output_count, // 0 .. 2048
    output logic [13:0]             read_count,   // 0 .. 8192

    output logic signed [7:0]       value_a,
    output logic signed [7:0]       value_b,
    output logic signed [7:0]       value_c,
    output logic signed [7:0]       value_d,
    output logic signed [7:0]       maximum_value
);

    localparam logic [4:0] LAST_CH = 5'd31;
    localparam logic [11:0] TOTAL_COUNT = 12'd2048;
    localparam logic [13:0] TOTAL_READS = 14'd8192;

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

    logic [4:0] ch_r;
    logic [2:0] pr_r;
    logic [2:0] pc_r;
    logic [1:0] win_r;
    logic [11:0] count_r;
    logic [13:0] reads_r;

    logic signed [7:0] a_r, b_r, c_r, d_r;
    logic signed [7:0] max_w;

    logic [3:0] in_row_w;
    logic [3:0] in_col_w;
    logic [12:0] conv_addr_w;
    logic [10:0] pool_addr_w;
    logic last_coord;

    /* verilator lint_off UNUSED */
    wire [3:0] dbg_input_row = in_row_w;
    wire [3:0] dbg_input_col = in_col_w;
    /* verilator lint_on UNUSED */

    maxpool2_address_generator u_agen (
        .channel             (ch_r),
        .pool_row            (pr_r),
        .pool_column         (pc_r),
        .window_index        (win_r),
        .input_row           (in_row_w),
        .input_column        (in_col_w),
        .conv2_read_address  (conv_addr_w),
        .pool2_write_address (pool_addr_w)
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
    assign read_count             = reads_r;
    assign conv2_read_address     = conv_addr_w;
    assign pool2_write_address    = pool_addr_w;
    assign pool2_write_data       = max_w;
    assign value_a = a_r;
    assign value_b = b_r;
    assign value_c = c_r;
    assign value_d = d_r;
    assign maximum_value = max_w;

    assign conv2_read_enable  = (state == ST_ISSUE);
    assign pool2_write_enable = (state == ST_WRITE);
    assign last_coord =
        (ch_r == LAST_CH) && (pr_r == 3'd7) && (pc_r == 3'd7);

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= ST_IDLE;
            ch_r <= 5'd0;
            pr_r <= 3'd0;
            pc_r <= 3'd0;
            win_r <= 2'd0;
            count_r <= 12'd0;
            reads_r <= 14'd0;
            a_r <= 8'sd0;
            b_r <= 8'sd0;
            c_r <= 8'sd0;
            d_r <= 8'sd0;
            busy <= 1'b0;
            pool2_done <= 1'b0;
        end else begin
            state <= state_n;
            pool2_done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        ch_r <= 5'd0;
                        pr_r <= 3'd0;
                        pc_r <= 3'd0;
                        win_r <= 2'd0;
                        count_r <= 12'd0;
                        reads_r <= 14'd0;
                    end
                end
                ST_ISSUE: begin
                    reads_r <= reads_r + 14'd1;
                end
                ST_CAPTURE: begin
                    unique case (win_r)
                        2'd0: a_r <= conv2_read_data;
                        2'd1: b_r <= conv2_read_data;
                        2'd2: c_r <= conv2_read_data;
                        default: d_r <= conv2_read_data;
                    endcase
                    if (win_r != 2'd3)
                        win_r <= win_r + 2'd1;
                end
                ST_WRITE: begin
                    count_r <= count_r + 12'd1;
                end
                ST_ADVANCE: begin
                    win_r <= 2'd0;
                    if (!last_coord) begin
                        if (pc_r == 3'd7) begin
                            pc_r <= 3'd0;
                            if (pr_r == 3'd7) begin
                                pr_r <= 3'd0;
                                ch_r <= ch_r + 5'd1;
                            end else begin
                                pr_r <= pr_r + 3'd1;
                            end
                        end else begin
                            pc_r <= pc_r + 3'd1;
                        end
                    end
                end
                ST_DONE: begin
                    pool2_done <= 1'b1;
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
            if (conv2_read_enable &&
                ({1'b0, conv2_read_address} >= 14'd8192))
                $error("conv2 read addr OOB");
            if (pool2_write_enable &&
                ({1'b0, pool2_write_address} >= 12'd2048))
                $error("pool2 write addr OOB");
            if (count_r > TOTAL_COUNT)
                $error("output_count exceeded");
            if (reads_r > TOTAL_READS)
                $error("read_count exceeded");
            if (pool2_done && (count_r != TOTAL_COUNT))
                $error("pool2_done with bad count");
            if (pool2_done && (reads_r != TOTAL_READS))
                $error("pool2_done with bad read_count");
            if (pool2_done &&
                ((ch_r != LAST_CH) || (pr_r != 3'd7) || (pc_r != 3'd7)))
                $error("pool2_done with wrong final coordinate");
        end
    end
    // synopsys translate_on

endmodule
