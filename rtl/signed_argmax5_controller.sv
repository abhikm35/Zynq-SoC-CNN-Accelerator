// signed_argmax5_controller.sv
// Sequential signed argmax over five logits in sync storage (1-cycle latency).
//
//   IDLE -> ISSUE_READ -> WAIT_READ -> CAPTURE_AND_COMPARE -> ADVANCE -> DONE
//
// Tie-breaking: update only on strictly greater signed value (lowest index wins).
// No softmax / requant / ReLU.

`timescale 1ns / 1ps

module signed_argmax5_controller #(
    parameter int LOGIT_WIDTH = 32
) (
    input  logic                          clk,
    input  logic                          rst,
    input  logic                          start,

    output logic                          busy,
    output logic                          done,

    output logic                          logit_read_enable,
    output logic [2:0]                    logit_read_address,
    input  logic signed [LOGIT_WIDTH-1:0] logit_read_data,

    output logic [2:0]                    predicted_class,
    output logic signed [LOGIT_WIDTH-1:0] maximum_logit,
    output logic [2:0]                    current_index,
    output logic [3:0]                    read_count  // 0 .. 5
);

    localparam logic [2:0] LAST_IDX = 3'd4;

    localparam logic [2:0] ST_IDLE     = 3'd0;
    localparam logic [2:0] ST_ISSUE    = 3'd1;
    localparam logic [2:0] ST_WAIT     = 3'd2;
    localparam logic [2:0] ST_CAPTURE  = 3'd3;
    localparam logic [2:0] ST_ADVANCE  = 3'd4;
    localparam logic [2:0] ST_DONE     = 3'd5;

    logic [2:0] state;
    logic [2:0] state_n;

    logic [2:0] idx_r;
    logic [3:0] reads_r;
    logic [2:0] pred_r;
    logic signed [LOGIT_WIDTH-1:0] max_r;
    logic first_r;

    logic last_idx;

    assign logit_read_enable  = (state == ST_ISSUE);
    assign logit_read_address = idx_r;
    assign predicted_class    = pred_r;
    assign maximum_logit      = max_r;
    assign current_index      = idx_r;
    assign read_count         = reads_r;
    assign last_idx           = (idx_r == LAST_IDX);

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= ST_IDLE;
            idx_r <= 3'd0;
            reads_r <= 4'd0;
            pred_r <= 3'd0;
            max_r <= '0;
            first_r <= 1'b1;
            busy <= 1'b0;
            done <= 1'b0;
        end else begin
            state <= state_n;
            done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        idx_r <= 3'd0;
                        reads_r <= 4'd0;
                        first_r <= 1'b1;
                        pred_r <= 3'd0;
                        max_r <= '0;
                    end
                end
                ST_ISSUE: begin
                    reads_r <= reads_r + 4'd1;
                end
                ST_CAPTURE: begin
                    if (first_r) begin
                        max_r   <= logit_read_data;
                        pred_r  <= 3'd0;
                        first_r <= 1'b0;
                    end else if (logit_read_data > max_r) begin
                        max_r  <= logit_read_data;
                        pred_r <= idx_r;
                    end
                end
                ST_ADVANCE: begin
                    if (!last_idx)
                        idx_r <= idx_r + 3'd1;
                end
                ST_DONE: begin
                    done <= 1'b1;
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
                state_n = ST_ADVANCE;
            end
            ST_ADVANCE: begin
                if (last_idx)
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
            if (busy && start)
                $error("argmax start while busy");
            if (logit_read_enable && ({1'b0, logit_read_address} >= 4'd5))
                $error("logit read OOB");
            if (reads_r > 4'd5)
                $error("read_count exceeded");
            if (done && (reads_r != 4'd5))
                $error("done with bad read_count");
            if (done && (pred_r > 3'd4))
                $error("predicted_class OOB");
        end
    end
    // synopsys translate_on

endmodule
