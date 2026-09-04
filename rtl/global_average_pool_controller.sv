// global_average_pool_controller.sv
// GAP over full Pool2 tensor (trained: 32 x 8 x 8 -> 32 INT8 values).
//
// Per channel:
//   CLEAR_SUM -> (ISSUE -> WAIT -> ACCUMULATE) x64 -> AVERAGE -> WRITE -> ADVANCE
//
// Exact Python path (integer_inference.py):
//   avg = saturate(round_divide_int(sum, 64))          // ties away from zero
//   gap = requantize_int32(avg, gap_mult, gap_shift)    // shared per-tensor
//
// Sync Pool2 memory: ISSUE -> WAIT -> ACCUMULATE (1-cycle read latency).

`timescale 1ns / 1ps

module global_average_pool_controller #(
    parameter int NUM_CHANNELS = 32,
    parameter logic signed [31:0] GAP_MULTIPLIER = 32'sd1759306569,
    parameter logic [5:0]         GAP_SHIFT = 6'd29
) (
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    start,

    output logic                    busy,
    output logic                    gap_done,

    output logic                    pool2_read_enable,
    output logic [10:0]             pool2_read_address,
    input  logic signed [7:0]       pool2_read_data,

    output logic                    gap_write_enable,
    output logic [4:0]              gap_write_address,
    output logic signed [7:0]       gap_write_data,

    output logic [4:0]              current_channel,
    output logic [5:0]              current_element,
    output logic signed [31:0]      running_sum,
    output logic [11:0]             read_count,   // 0 .. 2048
    output logic [5:0]              output_count, // 0 .. 32

    output logic signed [31:0]      final_sum,
    output logic signed [7:0]       averaged_int8,
    output logic signed [7:0]       gap_int8
);

    localparam logic [4:0] LAST_CH = 5'(NUM_CHANNELS - 1);
    localparam logic [5:0] LAST_ELEM = 6'd63;
    localparam logic [11:0] TOTAL_READS = 12'(NUM_CHANNELS * 64);
    localparam logic [5:0] TOTAL_OUTS = 6'(NUM_CHANNELS);

    localparam logic [3:0] ST_IDLE       = 4'd0;
    localparam logic [3:0] ST_CLEAR_SUM  = 4'd1;
    localparam logic [3:0] ST_ISSUE      = 4'd2;
    localparam logic [3:0] ST_WAIT       = 4'd3;
    localparam logic [3:0] ST_ACCUMULATE = 4'd4;
    localparam logic [3:0] ST_AVERAGE    = 4'd5;
    localparam logic [3:0] ST_WRITE      = 4'd6;
    localparam logic [3:0] ST_ADVANCE    = 4'd7;
    localparam logic [3:0] ST_DONE       = 4'd8;

    logic [3:0] state;
    logic [3:0] state_n;

    logic [4:0] ch_r;
    logic [5:0] elem_r;
    logic [11:0] reads_r;
    logic [5:0] outs_r;
    logic signed [31:0] sum_r;
    logic signed [31:0] final_sum_r;
    logic signed [7:0] avg_r;
    logic signed [7:0] gap_r;

    logic signed [7:0] avg_w;
    logic signed [31:0] rounding_adjustment_w;
    logic signed [31:0] adjusted_sum_w;
    logic signed [31:0] shifted_w;
    logic signed [7:0] sat_avg_w;

    logic signed [31:0] avg_as_acc;
    logic signed [63:0] wide_product_w;
    logic signed [63:0] rounding_offset_w;
    logic signed [63:0] rounded_product_w;
    logic signed [63:0] shifted_req_w;
    logic signed [31:0] zp_adj_w;
    logic signed [7:0] gap_sat_w;

    logic last_elem;
    logic last_channel;

    /* verilator lint_off UNUSED */
    wire signed [31:0] dbg_rounding_adjustment = rounding_adjustment_w;
    wire signed [31:0] dbg_adjusted_sum = adjusted_sum_w;
    wire signed [31:0] dbg_shifted = shifted_w;
    wire signed [7:0]  dbg_sat_avg = sat_avg_w;
    wire signed [63:0] dbg_wide = wide_product_w;
    wire signed [63:0] dbg_roff = rounding_offset_w;
    wire signed [63:0] dbg_rprod = rounded_product_w;
    wire signed [63:0] dbg_shreq = shifted_req_w;
    wire signed [31:0] dbg_zpadj = zp_adj_w;
    /* verilator lint_on UNUSED */

    // Average from the live channel sum (valid once all 64 values are in sum_r).
    gap_average u_avg (
        .sum_value                (sum_r),
        .rounding_adjustment      (rounding_adjustment_w),
        .adjusted_sum             (adjusted_sum_w),
        .shifted_or_divided_value (shifted_w),
        .saturated_value          (sat_avg_w),
        .final_output             (avg_w)
    );

    assign avg_as_acc = {{24{avg_w[7]}}, avg_w};

    requantize u_req (
        .accumulator         (avg_as_acc),
        .multiplier          (GAP_MULTIPLIER),
        .shift               (GAP_SHIFT),
        .output_zero_point   (8'sd0),
        .wide_product        (wide_product_w),
        .rounding_offset     (rounding_offset_w),
        .rounded_product     (rounded_product_w),
        .shifted_value       (shifted_req_w),
        .zero_point_adjusted (zp_adj_w),
        .saturated_value     (gap_sat_w)
    );

    assign current_channel   = ch_r;
    assign current_element   = elem_r;
    assign running_sum       = sum_r;
    assign read_count        = reads_r;
    assign output_count      = outs_r;
    assign final_sum         = final_sum_r;
    assign averaged_int8     = avg_r;
    assign gap_int8          = gap_r;

    assign pool2_read_address = {ch_r, elem_r}; // channel * 64 + element_index
    assign pool2_read_enable  = (state == ST_ISSUE);
    assign gap_write_enable   = (state == ST_WRITE);
    assign gap_write_address  = ch_r;
    assign gap_write_data     = gap_r;

    assign last_elem    = (elem_r == LAST_ELEM);
    assign last_channel = (ch_r == LAST_CH);

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= ST_IDLE;
            ch_r <= 5'd0;
            elem_r <= 6'd0;
            reads_r <= 12'd0;
            outs_r <= 6'd0;
            sum_r <= 32'sd0;
            final_sum_r <= 32'sd0;
            avg_r <= 8'sd0;
            gap_r <= 8'sd0;
            busy <= 1'b0;
            gap_done <= 1'b0;
        end else begin
            state <= state_n;
            gap_done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        ch_r <= 5'd0;
                        elem_r <= 6'd0;
                        reads_r <= 12'd0;
                        outs_r <= 6'd0;
                        sum_r <= 32'sd0;
                        final_sum_r <= 32'sd0;
                        avg_r <= 8'sd0;
                        gap_r <= 8'sd0;
                    end
                end
                ST_CLEAR_SUM: begin
                    sum_r <= 32'sd0;
                    elem_r <= 6'd0;
                end
                ST_ISSUE: begin
                    reads_r <= reads_r + 12'd1;
                end
                ST_ACCUMULATE: begin
                    sum_r <= sum_r + {{24{pool2_read_data[7]}}, pool2_read_data};
                    if (!last_elem)
                        elem_r <= elem_r + 6'd1;
                end
                ST_AVERAGE: begin
                    final_sum_r <= sum_r;
                    avg_r <= avg_w;
                    gap_r <= gap_sat_w;
                end
                ST_WRITE: begin
                    outs_r <= outs_r + 6'd1;
                end
                ST_ADVANCE: begin
                    if (!last_channel)
                        ch_r <= ch_r + 5'd1;
                end
                ST_DONE: begin
                    gap_done <= 1'b1;
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
                    state_n = ST_CLEAR_SUM;
            end
            ST_CLEAR_SUM: begin
                state_n = ST_ISSUE;
            end
            ST_ISSUE: begin
                state_n = ST_WAIT;
            end
            ST_WAIT: begin
                state_n = ST_ACCUMULATE;
            end
            ST_ACCUMULATE: begin
                if (last_elem)
                    state_n = ST_AVERAGE;
                else
                    state_n = ST_ISSUE;
            end
            ST_AVERAGE: begin
                state_n = ST_WRITE;
            end
            ST_WRITE: begin
                state_n = ST_ADVANCE;
            end
            ST_ADVANCE: begin
                if (last_channel)
                    state_n = ST_DONE;
                else
                    state_n = ST_CLEAR_SUM;
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
            if (busy && start)
                $error("GAP start while busy");
            if (pool2_read_enable && ({1'b0, pool2_read_address} >= 12'd2048))
                $error("pool2 read addr OOB");
            if (gap_write_enable && ({1'b0, gap_write_address} >= 6'd32))
                $error("gap write addr OOB");
            if (reads_r > TOTAL_READS)
                $error("read_count exceeded");
            if (outs_r > TOTAL_OUTS)
                $error("output_count exceeded");
            if (gap_done && (reads_r != TOTAL_READS))
                $error("gap_done with bad read_count");
            if (gap_done && (outs_r != TOTAL_OUTS))
                $error("gap_done with bad output_count");
            if (gap_done && (ch_r != LAST_CH))
                $error("gap_done with wrong final channel");
        end
    end
    // synopsys translate_on

endmodule
