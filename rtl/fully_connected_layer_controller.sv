// fully_connected_layer_controller.sv
// Drive the single-class FC engine across all 5 classes.
//
//   IDLE -> START_CLASS -> WAIT_CLASS -> WRITE_LOGIT -> ADVANCE -> (next | DONE)
//
// Exactly 5 logit writes. fc_done after class 4 is written.

`timescale 1ns / 1ps

module fully_connected_layer_controller #(
    parameter int NUM_CLASSES = 5,
    parameter int NUM_FEATURES = 32
) (
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    start,

    output logic                    busy,
    output logic                    fc_done,
    output logic [2:0]              class_index,
    output logic [3:0]              class_count,     // 0 .. 5
    output logic [7:0]              total_mac_count, // 0 .. 160

    // Engine control
    output logic                    engine_start,
    output logic [2:0]              engine_class_index,
    input  logic                    engine_busy,
    input  logic                    engine_done,
    input  logic signed [31:0]      engine_logit,
    input  logic signed [31:0]      engine_accumulator,
    input  logic [5:0]              engine_mac_count,

    // Logit storage write
    output logic                    logit_write_enable,
    output logic [2:0]              logit_write_address,
    output logic signed [31:0]      logit_write_data
);

    localparam logic [2:0] LAST_CLASS = 3'(NUM_CLASSES - 1);
    localparam logic [3:0] TOTAL_CLASSES = 4'(NUM_CLASSES);
    localparam logic [7:0] TOTAL_MACS = 8'(NUM_CLASSES * NUM_FEATURES);

    localparam logic [2:0] ST_IDLE        = 3'd0;
    localparam logic [2:0] ST_START_CLASS = 3'd1;
    localparam logic [2:0] ST_WAIT_CLASS  = 3'd2;
    localparam logic [2:0] ST_WRITE_LOGIT = 3'd3;
    localparam logic [2:0] ST_ADVANCE     = 3'd4;
    localparam logic [2:0] ST_DONE        = 3'd5;

    logic [2:0] state;
    logic [2:0] state_n;

    logic [2:0] class_r;
    logic [3:0] count_r;
    logic [7:0] macs_r;
    logic signed [31:0] logit_r;
    logic signed [31:0] acc_r;

    logic last_class;

    assign class_index         = class_r;
    assign class_count         = count_r;
    assign total_mac_count     = macs_r;
    assign engine_class_index  = class_r;
    assign engine_start        = (state == ST_START_CLASS);
    assign logit_write_enable  = (state == ST_WRITE_LOGIT);
    assign logit_write_address = class_r;
    assign logit_write_data    = logit_r;
    assign last_class          = (class_r == LAST_CLASS);

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= ST_IDLE;
            class_r <= 3'd0;
            count_r <= 4'd0;
            macs_r <= 8'd0;
            logit_r <= 32'sd0;
            acc_r <= 32'sd0;
            busy <= 1'b0;
            fc_done <= 1'b0;
        end else begin
            state <= state_n;
            fc_done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        class_r <= 3'd0;
                        count_r <= 4'd0;
                        macs_r <= 8'd0;
                    end
                end
                ST_WAIT_CLASS: begin
                    if (engine_done) begin
                        logit_r <= engine_logit;
                        acc_r   <= engine_accumulator;
                        macs_r  <= macs_r + {2'b0, engine_mac_count};
                    end
                end
                ST_WRITE_LOGIT: begin
                    count_r <= count_r + 4'd1;
                end
                ST_ADVANCE: begin
                    if (!last_class)
                        class_r <= class_r + 3'd1;
                end
                ST_DONE: begin
                    fc_done <= 1'b1;
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
                    state_n = ST_START_CLASS;
            end
            ST_START_CLASS: begin
                state_n = ST_WAIT_CLASS;
            end
            ST_WAIT_CLASS: begin
                if (engine_done)
                    state_n = ST_WRITE_LOGIT;
            end
            ST_WRITE_LOGIT: begin
                state_n = ST_ADVANCE;
            end
            ST_ADVANCE: begin
                if (last_class)
                    state_n = ST_DONE;
                else
                    state_n = ST_START_CLASS;
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
                $error("FC start while busy");
            if (engine_start && engine_busy)
                $error("engine start while busy");
            if (logit_write_enable && ({1'b0, logit_write_address} >= 4'd5))
                $error("logit write OOB");
            if (count_r > TOTAL_CLASSES)
                $error("class_count exceeded");
            if (macs_r > TOTAL_MACS)
                $error("total_mac_count exceeded");
            if (fc_done && (count_r != TOTAL_CLASSES))
                $error("fc_done with bad class_count");
            if (fc_done && (macs_r != TOTAL_MACS))
                $error("fc_done with bad total_mac_count");
            if (fc_done && (class_r != LAST_CLASS))
                $error("fc_done with wrong final class");
        end
    end
    // synopsys translate_on

endmodule
