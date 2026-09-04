// cnn_top_controller.sv
// Top-level sequencer for end-to-end CNN inference:
//   Conv1 -> MaxPool1 -> Conv2 -> MaxPool2 -> GAP -> FC -> Argmax
//
// Each stage start is a 1-cycle pulse. Next stage starts only after done.
// Start-while-busy is ignored at this level (IDLE only accepts start).
//
// Trained sizes (obsolete prompt 8/16 sketches are not used):
//   3x32x32 -> 16x32x32 -> 16x16x16 -> 32x16x16 -> 32x8x8 -> 32 -> 5 -> class

`timescale 1ns / 1ps

module cnn_top_controller (
    input  logic        clk,
    input  logic        rst,
    input  logic        start,

    output logic        busy,
    output logic        done,

    output logic        conv1_start,
    input  logic        conv1_busy,
    input  logic        conv1_done,

    output logic        pool1_start,
    input  logic        pool1_busy,
    input  logic        pool1_done,

    output logic        conv2_start,
    input  logic        conv2_busy,
    input  logic        conv2_done,

    output logic        pool2_start,
    input  logic        pool2_busy,
    input  logic        pool2_done,

    output logic        gap_start,
    input  logic        gap_busy,
    input  logic        gap_done,

    output logic        fc_start,
    input  logic        fc_busy,
    input  logic        fc_done,

    output logic        argmax_start,
    input  logic        argmax_busy,
    input  logic        argmax_done,

    output logic [3:0]  state_id,
    output logic [2:0]  stage_id,
    output logic [63:0] cycle_count,

    output logic [63:0] conv1_cycles,
    output logic [63:0] pool1_cycles,
    output logic [63:0] conv2_cycles,
    output logic [63:0] pool2_cycles,
    output logic [63:0] gap_cycles,
    output logic [63:0] fc_cycles,
    output logic [63:0] argmax_cycles
);

    localparam logic [3:0] ST_IDLE         = 4'd0;
    localparam logic [3:0] ST_START_CONV1  = 4'd1;
    localparam logic [3:0] ST_WAIT_CONV1   = 4'd2;
    localparam logic [3:0] ST_START_POOL1  = 4'd3;
    localparam logic [3:0] ST_WAIT_POOL1   = 4'd4;
    localparam logic [3:0] ST_START_CONV2  = 4'd5;
    localparam logic [3:0] ST_WAIT_CONV2   = 4'd6;
    localparam logic [3:0] ST_START_POOL2  = 4'd7;
    localparam logic [3:0] ST_WAIT_POOL2   = 4'd8;
    localparam logic [3:0] ST_START_GAP    = 4'd9;
    localparam logic [3:0] ST_WAIT_GAP     = 4'd10;
    localparam logic [3:0] ST_START_FC     = 4'd11;
    localparam logic [3:0] ST_WAIT_FC      = 4'd12;
    localparam logic [3:0] ST_START_ARGMAX = 4'd13;
    localparam logic [3:0] ST_WAIT_ARGMAX  = 4'd14;
    localparam logic [3:0] ST_DONE         = 4'd15;

    logic [3:0] state;
    logic [3:0] state_n;

    logic [63:0] cycles_r;
    logic [63:0] s_conv1, s_pool1, s_conv2, s_pool2, s_gap, s_fc, s_argmax;

    assign state_id = state;
    assign cycle_count = cycles_r;
    assign conv1_cycles  = s_conv1;
    assign pool1_cycles  = s_pool1;
    assign conv2_cycles  = s_conv2;
    assign pool2_cycles  = s_pool2;
    assign gap_cycles    = s_gap;
    assign fc_cycles     = s_fc;
    assign argmax_cycles = s_argmax;

    always_comb begin
        unique case (state)
            ST_START_CONV1, ST_WAIT_CONV1: stage_id = 3'd0;
            ST_START_POOL1, ST_WAIT_POOL1: stage_id = 3'd1;
            ST_START_CONV2, ST_WAIT_CONV2: stage_id = 3'd2;
            ST_START_POOL2, ST_WAIT_POOL2: stage_id = 3'd3;
            ST_START_GAP,   ST_WAIT_GAP:   stage_id = 3'd4;
            ST_START_FC,    ST_WAIT_FC:    stage_id = 3'd5;
            ST_START_ARGMAX,ST_WAIT_ARGMAX:stage_id = 3'd6;
            default: stage_id = 3'd7;
        endcase
    end

    assign conv1_start  = (state == ST_START_CONV1);
    assign pool1_start  = (state == ST_START_POOL1);
    assign conv2_start  = (state == ST_START_CONV2);
    assign pool2_start  = (state == ST_START_POOL2);
    assign gap_start    = (state == ST_START_GAP);
    assign fc_start     = (state == ST_START_FC);
    assign argmax_start = (state == ST_START_ARGMAX);

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= ST_IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            cycles_r <= 64'd0;
            s_conv1 <= 64'd0;
            s_pool1 <= 64'd0;
            s_conv2 <= 64'd0;
            s_pool2 <= 64'd0;
            s_gap <= 64'd0;
            s_fc <= 64'd0;
            s_argmax <= 64'd0;
        end else begin
            state <= state_n;
            done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        cycles_r <= 64'd0;
                        s_conv1 <= 64'd0;
                        s_pool1 <= 64'd0;
                        s_conv2 <= 64'd0;
                        s_pool2 <= 64'd0;
                        s_gap <= 64'd0;
                        s_fc <= 64'd0;
                        s_argmax <= 64'd0;
                    end
                end
                ST_DONE: begin
                    done <= 1'b1;
                    busy <= 1'b0;
                end
                default: begin
                    if (busy)
                        cycles_r <= cycles_r + 64'd1;
                    unique case (stage_id)
                        3'd0: s_conv1  <= s_conv1  + 64'd1;
                        3'd1: s_pool1  <= s_pool1  + 64'd1;
                        3'd2: s_conv2  <= s_conv2  + 64'd1;
                        3'd3: s_pool2  <= s_pool2  + 64'd1;
                        3'd4: s_gap    <= s_gap    + 64'd1;
                        3'd5: s_fc     <= s_fc     + 64'd1;
                        3'd6: s_argmax <= s_argmax + 64'd1;
                        default: ;
                    endcase
                end
            endcase
        end
    end

    always_comb begin
        state_n = state;
        unique case (state)
            ST_IDLE: if (start) state_n = ST_START_CONV1;
            ST_START_CONV1: state_n = ST_WAIT_CONV1;
            ST_WAIT_CONV1: if (conv1_done) state_n = ST_START_POOL1;
            ST_START_POOL1: state_n = ST_WAIT_POOL1;
            ST_WAIT_POOL1: if (pool1_done) state_n = ST_START_CONV2;
            ST_START_CONV2: state_n = ST_WAIT_CONV2;
            ST_WAIT_CONV2: if (conv2_done) state_n = ST_START_POOL2;
            ST_START_POOL2: state_n = ST_WAIT_POOL2;
            ST_WAIT_POOL2: if (pool2_done) state_n = ST_START_GAP;
            ST_START_GAP: state_n = ST_WAIT_GAP;
            ST_WAIT_GAP: if (gap_done) state_n = ST_START_FC;
            ST_START_FC: state_n = ST_WAIT_FC;
            ST_WAIT_FC: if (fc_done) state_n = ST_START_ARGMAX;
            ST_START_ARGMAX: state_n = ST_WAIT_ARGMAX;
            ST_WAIT_ARGMAX: if (argmax_done) state_n = ST_DONE;
            ST_DONE: state_n = ST_IDLE;
            default: state_n = ST_IDLE;
        endcase
    end

    /* verilator lint_off UNUSED */
    wire unused_stage_busy =
        conv1_busy | pool1_busy | conv2_busy | pool2_busy |
        gap_busy | fc_busy | argmax_busy;
    /* verilator lint_on UNUSED */

endmodule
