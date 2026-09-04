// conv_single_output.sv
// One Conv1 output pixel: exactly 27 signed INT8 MACs into INT32, then
// requantize + ReLU matching the Python integer golden model.
//
// Operand streaming (Verilator-friendly — no unpacked array ports):
//   In ST_MAC the module drives mac_sel (0..26). The testbench must present
//   activation/weight for that index in the same cycle (combinational ROM).
//
// FSM:
//   IDLE -> LOAD_BIAS -> MAC x27 -> REQUANTIZE -> DONE
//
// MAC order: input_channel (0..2), kernel_row (0..2), kernel_col (0..2)
//   mac_sel = ic*9 + kr*3 + kc

`timescale 1ns / 1ps

module conv_single_output (
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    start,
    input  logic signed [31:0]      bias,
    input  logic signed [31:0]      multiplier,
    input  logic        [5:0]       shift,
    input  logic signed [7:0]       output_zero_point,
    // Streamed operands for mac_sel (TB ROM)
    input  logic signed [7:0]       activation,
    input  logic signed [7:0]       weight,

    output logic        [4:0]       mac_sel,
    output logic signed [15:0]      product,
    output logic signed [15:0]      last_product,
    output logic signed [31:0]      accumulator,
    output logic signed [31:0]      final_accumulator,
    output logic signed [7:0]       requantized_output,
    output logic signed [7:0]       relu_output,
    output logic        [4:0]       mac_count,
    output logic                    busy,
    output logic                    done
);

    localparam logic [2:0] ST_IDLE       = 3'd0;
    localparam logic [2:0] ST_LOAD_BIAS  = 3'd1;
    localparam logic [2:0] ST_MAC        = 3'd2;
    localparam logic [2:0] ST_REQUANTIZE = 3'd3;
    localparam logic [2:0] ST_DONE       = 3'd4;

    logic [2:0] state;
    logic [2:0] state_n;

    logic load_bias;
    logic mac_enable;
    logic [4:0] mac_sel_r;
    logic [4:0] macs_done_r;

    logic signed [15:0] prod_w;
    logic signed [31:0] acc_w;

    logic signed [63:0] wide_product;
    logic signed [63:0] rounding_offset;
    logic signed [63:0] rounded_product;
    logic signed [63:0] shifted_value;
    logic signed [31:0] zero_point_adjusted;
    logic signed [7:0]  sat_value;
    logic signed [7:0]  relu_value;

    int8_mac u_mac (
        .clk         (clk),
        .rst         (rst),
        .load_bias   (load_bias),
        .enable      (mac_enable),
        .activation  (activation),
        .weight      (weight),
        .bias        (bias),
        .product     (prod_w),
        .accumulator (acc_w)
    );

    requantize u_requant (
        .accumulator         (acc_w),
        .multiplier          (multiplier),
        .shift               (shift),
        .output_zero_point   (output_zero_point),
        .wide_product        (wide_product),
        .rounding_offset     (rounding_offset),
        .rounded_product     (rounded_product),
        .shifted_value       (shifted_value),
        .zero_point_adjusted (zero_point_adjusted),
        .saturated_value     (sat_value)
    );

    relu_int8 u_relu (
        .in_value   (sat_value),
        .zero_point (8'sd0),
        .out_value  (relu_value)
    );

    assign product = prod_w;
    assign accumulator = acc_w;
    assign mac_sel = mac_sel_r;
    assign mac_count = macs_done_r;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= ST_IDLE;
            mac_sel_r <= 5'd0;
            macs_done_r <= 5'd0;
            final_accumulator <= 32'sd0;
            requantized_output <= 8'sd0;
            relu_output <= 8'sd0;
            last_product <= 16'sd0;
            done <= 1'b0;
            busy <= 1'b0;
        end else begin
            state <= state_n;
            done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    busy <= 1'b0;
                    mac_sel_r <= 5'd0;
                    macs_done_r <= 5'd0;
                    if (start)
                        busy <= 1'b1;
                end
                ST_LOAD_BIAS: begin
                    mac_sel_r <= 5'd0;
                    macs_done_r <= 5'd0;
                end
                ST_MAC: begin
                    last_product <= prod_w;
                    macs_done_r <= mac_sel_r + 5'd1;
                    if (mac_sel_r < 5'd26)
                        mac_sel_r <= mac_sel_r + 5'd1;
                end
                ST_REQUANTIZE: begin
                    final_accumulator  <= acc_w;
                    requantized_output <= sat_value;
                    relu_output        <= relu_value;
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
        load_bias = 1'b0;
        mac_enable = 1'b0;
        case (state)
            ST_IDLE: begin
                if (start)
                    state_n = ST_LOAD_BIAS;
            end
            ST_LOAD_BIAS: begin
                load_bias  = 1'b1;
                mac_enable = 1'b1;
                state_n    = ST_MAC;
            end
            ST_MAC: begin
                mac_enable = 1'b1;
                if (mac_sel_r == 5'd26)
                    state_n = ST_REQUANTIZE;
            end
            ST_REQUANTIZE: begin
                state_n = ST_DONE;
            end
            ST_DONE: begin
                state_n = ST_IDLE;
            end
            default: state_n = ST_IDLE;
        endcase
    end

endmodule
