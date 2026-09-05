// fully_connected_class_engine.sv
// Compute one classifier logit for Linear 32 -> 5.
//
// FSM (1-cycle sync memories):
//   IDLE -> ISSUE_BIAS -> WAIT_BIAS -> LOAD_BIAS
//       -> ISSUE_OP -> WAIT_MAC  (x32)
//       -> REQ_MUL -> REQ_SHIFT -> REQ_OUT -> DONE
//
// Arithmetic (Python integer_linear + requantize_int32):
//   acc = bias + Σ gap[i] * weight[class][i]   // ZP = 0
//   logit = saturate_int32(round(acc * mult / 2^shift))
// No ReLU on logits.
//
// Timing: 64-bit multiply, round/shift, and INT32 sat are separate cycles.
// Reuses int8_mac. External memories.

`timescale 1ns / 1ps

module fully_connected_class_engine #(
    parameter int NUM_FEATURES = 32
) (
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    start,
    input  logic [2:0]              class_index,  // 0 .. 4

    output logic                    busy,
    output logic                    done,
    output logic                    logit_valid,
    output logic signed [31:0]      logit_value,
    output logic signed [31:0]      final_accumulator,
    output logic [5:0]              mac_count,    // 0 .. 32

    output logic [4:0]              debug_input_index,
    output logic signed [15:0]      debug_product,
    output logic signed [31:0]      debug_accumulator,

    // GAP feature ROM / storage (1-cycle latency)
    output logic                    gap_read_enable,
    output logic [4:0]              gap_read_address,
    input  logic signed [7:0]       gap_read_data,

    // Weight ROM
    output logic                    weight_read_enable,
    output logic [7:0]              weight_read_address,
    input  logic signed [7:0]       weight_read_data,

    // Bias / multiplier / shift ROMs (INT32)
    output logic                    bias_read_enable,
    output logic [2:0]              bias_read_address,
    input  logic signed [31:0]      bias_read_data,

    output logic                    mult_read_enable,
    output logic [2:0]              mult_read_address,
    input  logic signed [31:0]      mult_read_data,

    output logic                    shift_read_enable,
    output logic [2:0]              shift_read_address,
    input  logic signed [31:0]      shift_read_data
);

    localparam logic [4:0] LAST_INPUT = 5'(NUM_FEATURES - 1);
    localparam logic [5:0] MACS_PER_CLASS = 6'(NUM_FEATURES);

    localparam logic [3:0] ST_IDLE       = 4'd0;
    localparam logic [3:0] ST_ISSUE_BIAS = 4'd1;
    localparam logic [3:0] ST_WAIT_BIAS  = 4'd2;
    localparam logic [3:0] ST_LOAD_BIAS  = 4'd3;
    localparam logic [3:0] ST_ISSUE_OP   = 4'd4;
    localparam logic [3:0] ST_WAIT_MAC   = 4'd5;
    localparam logic [3:0] ST_REQ_MUL    = 4'd6;
    localparam logic [3:0] ST_REQ_SHIFT  = 4'd7;
    localparam logic [3:0] ST_REQ_OUT    = 4'd8;
    localparam logic [3:0] ST_DONE       = 4'd9;

    logic [3:0] state;
    logic [3:0] state_n;

    logic [2:0] class_r;
    logic [4:0] input_r;
    logic [5:0] macs_r;

    logic signed [31:0] bias_r;
    logic signed [31:0] mult_r;
    logic [5:0]         shift_r;

    logic signed [31:0] final_acc_r;
    logic signed [31:0] logit_r;
    logic signed [63:0] wide_product_r;
    logic signed [63:0] shifted_value_r;

    logic load_bias_i;
    logic mac_en;
    logic signed [15:0] product_w;
    logic signed [31:0] acc_w;

    logic [4:0] gap_addr_w;
    logic [7:0] wgt_addr_w;
    logic [2:0] bias_addr_w;
    logic [2:0] logit_addr_w;

    logic signed [63:0] wide_w, roff_w, rprod_w, shifted_w;
    logic signed [31:0] zpadj_w;
    logic signed [31:0] logit_w;

    logic last_input;

    /* verilator lint_off UNUSED */
    wire [2:0] dbg_logit_addr = logit_addr_w;
    wire signed [63:0] dbg_wide = wide_w;
    wire signed [63:0] dbg_roff = roff_w;
    wire signed [63:0] dbg_rprod = rprod_w;
    wire signed [63:0] dbg_shifted = shifted_w;
    wire signed [31:0] dbg_zpadj = zpadj_w;
    /* verilator lint_on UNUSED */

    fc_address_generator u_agen (
        .class_index         (class_r),
        .input_index         (input_r),
        .gap_read_address    (gap_addr_w),
        .fc_weight_address   (wgt_addr_w),
        .fc_bias_address     (bias_addr_w),
        .logit_write_address (logit_addr_w)
    );

    int8_mac u_mac (
        .clk         (clk),
        .rst         (rst),
        .load_bias   (load_bias_i),
        .enable      (mac_en),
        .activation  (gap_read_data),
        .weight      (weight_read_data),
        .bias        (bias_r),
        .product     (product_w),
        .accumulator (acc_w)
    );

    // Pipeline: product -> round/shift -> INT32 saturate.
    always_comb begin
        wide_w = {{32{acc_w[31]}}, acc_w} * {{32{mult_r[31]}}, mult_r};
    end

    rounding_right_shift64 u_shift (
        .wide_product    (wide_product_r),
        .shift           (shift_r),
        .rounding_offset (roff_w),
        .rounded_product (rprod_w),
        .shifted_value   (shifted_w)
    );

    saturate_shifted_int32 u_sat (
        .shifted_value       (shifted_value_r),
        .output_zero_point   (8'sd0),
        .zero_point_adjusted (zpadj_w),
        .saturated_value     (logit_w)
    );

    assign gap_read_address    = gap_addr_w;
    assign weight_read_address = wgt_addr_w;
    assign bias_read_address   = class_r;
    assign mult_read_address   = class_r;
    assign shift_read_address  = class_r;

    assign gap_read_enable    = (state == ST_ISSUE_OP);
    assign weight_read_enable = (state == ST_ISSUE_OP);
    assign bias_read_enable   = (state == ST_ISSUE_BIAS);
    assign mult_read_enable   = (state == ST_ISSUE_BIAS);
    assign shift_read_enable  = (state == ST_ISSUE_BIAS);

    assign mac_count           = macs_r;
    assign debug_input_index   = input_r;
    assign debug_product       = product_w;
    assign debug_accumulator   = acc_w;
    assign final_accumulator   = final_acc_r;
    assign logit_value         = logit_r;
    assign logit_valid         = (state == ST_DONE);
    assign last_input          = (input_r == LAST_INPUT);

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= ST_IDLE;
            class_r <= 3'd0;
            input_r <= 5'd0;
            macs_r <= 6'd0;
            bias_r <= 32'sd0;
            mult_r <= 32'sd0;
            shift_r <= 6'd0;
            final_acc_r <= 32'sd0;
            logit_r <= 32'sd0;
            wide_product_r <= 64'sd0;
            shifted_value_r <= 64'sd0;
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
                        class_r <= class_index;
                        input_r <= 5'd0;
                        macs_r <= 6'd0;
                    end
                end
                ST_WAIT_BIAS: begin
                    bias_r  <= bias_read_data;
                    mult_r  <= mult_read_data;
                    shift_r <= shift_read_data[5:0];
                end
                ST_WAIT_MAC: begin
                    macs_r <= macs_r + 6'd1;
                    if (!last_input)
                        input_r <= input_r + 5'd1;
                end
                ST_REQ_MUL: begin
                    final_acc_r    <= acc_w;
                    wide_product_r <= wide_w;
                end
                ST_REQ_SHIFT: begin
                    shifted_value_r <= shifted_w;
                end
                ST_REQ_OUT: begin
                    logit_r <= logit_w;
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
        state_n     = state;
        load_bias_i = 1'b0;
        mac_en      = 1'b0;
        case (state)
            ST_IDLE: begin
                if (start)
                    state_n = ST_ISSUE_BIAS;
            end
            ST_ISSUE_BIAS: begin
                state_n = ST_WAIT_BIAS;
            end
            ST_WAIT_BIAS: begin
                state_n = ST_LOAD_BIAS;
            end
            ST_LOAD_BIAS: begin
                load_bias_i = 1'b1;
                mac_en      = 1'b1;
                state_n     = ST_ISSUE_OP;
            end
            ST_ISSUE_OP: begin
                state_n = ST_WAIT_MAC;
            end
            ST_WAIT_MAC: begin
                mac_en = 1'b1;
                if (last_input)
                    state_n = ST_REQ_MUL;
                else
                    state_n = ST_ISSUE_OP;
            end
            ST_REQ_MUL: begin
                state_n = ST_REQ_SHIFT;
            end
            ST_REQ_SHIFT: begin
                state_n = ST_REQ_OUT;
            end
            ST_REQ_OUT: begin
                state_n = ST_DONE;
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
            if (gap_read_enable && ({1'b0, gap_read_address} >= 6'd32))
                $error("GAP addr OOB");
            if (weight_read_enable && ({1'b0, weight_read_address} >= 9'd160))
                $error("weight addr OOB");
            if (mac_en && !load_bias_i && (macs_r >= MACS_PER_CLASS))
                $error("MAC count would exceed 32");
            if (done && (macs_r != MACS_PER_CLASS))
                $error("done with mac_count != 32");
        end
    end
    // synopsys translate_on

endmodule
