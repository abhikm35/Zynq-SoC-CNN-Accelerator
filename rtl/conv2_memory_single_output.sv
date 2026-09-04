// conv2_memory_single_output.sv
// Memory-driven single Conv2 output pixel (trained 16->32 on 16x16 Pool1).
//
// 144 MACs = 16 input channels x 3 x 3. Same FSM timing as Conv1 engine:
//   IDLE -> ISSUE_BIAS -> WAIT_BIAS -> LOAD_BIAS
//       -> ISSUE_OP -> WAIT_MAC  (x144)
//       -> REQUANTIZE -> DONE
//
// Reuses int8_mac, requantize, relu_int8. No floating-point.

`timescale 1ns / 1ps

module conv2_memory_single_output #(
    parameter bit EXTERNAL_ACT = 0,
    parameter ACT_MEM_FILE   = "vectors/conv2/pool1_input.mem",
    parameter WGT_MEM_FILE   = "vectors/conv2/conv2_weights.mem",
    parameter BIAS_MEM_FILE  = "vectors/conv2/conv2_biases.mem",
    parameter MULT_MEM_FILE  = "vectors/conv2/conv2_multipliers.mem",
    parameter SHIFT_MEM_FILE = "vectors/conv2/conv2_shifts.mem"
) (
    input  logic                clk,
    input  logic                rst,
    input  logic                start,
    input  logic [4:0]          output_channel,  // 0 .. 31
    input  logic [3:0]          output_row,      // 0 .. 15
    input  logic [3:0]          output_column,   // 0 .. 15

    // When EXTERNAL_ACT=1, activation comes from shared Pool1 RAM (1-cycle latency).
    input  logic signed [7:0]   external_act_data,

    output logic                busy,
    output logic                done,
    output logic signed [31:0]  final_accumulator,
    output logic signed [7:0]   requantized_output,
    output logic signed [7:0]   relu_output,
    output logic [7:0]          mac_count,       // 0 .. 144

    output logic [3:0]          current_input_channel,
    output logic [1:0]          current_kernel_row,
    output logic [1:0]          current_kernel_column,
    output logic [11:0]         current_activation_address,
    output logic [12:0]         current_weight_address,
    output logic [4:0]          current_bias_address,
    output logic [12:0]         current_output_address,
    output logic                current_padding,
    output logic signed [7:0]   current_activation_value,
    output logic signed [7:0]   current_weight_value,
    output logic signed [15:0]  current_product,
    output logic signed [31:0]  current_accumulator,
    output logic                operand_valid,
    output logic                mac_enable,
    output logic                load_bias,
    output logic                act_read_enable,
    output logic                wgt_read_enable,
    output logic signed [15:0]  last_product
);

    localparam logic [3:0] ST_IDLE       = 4'd0;
    localparam logic [3:0] ST_ISSUE_BIAS = 4'd1;
    localparam logic [3:0] ST_WAIT_BIAS  = 4'd2;
    localparam logic [3:0] ST_LOAD_BIAS  = 4'd3;
    localparam logic [3:0] ST_ISSUE_OP   = 4'd4;
    localparam logic [3:0] ST_WAIT_MAC   = 4'd5;
    localparam logic [3:0] ST_REQUANTIZE = 4'd6;
    localparam logic [3:0] ST_DONE       = 4'd7;

    localparam logic [7:0] MACS_PER_OUTPUT = 8'd144;

    logic [3:0] state;
    logic [3:0] state_n;

    logic [4:0] oc_r;
    logic [3:0] or_r;
    logic [3:0] oc_col_r;

    logic [3:0] ic_r;
    logic [1:0] kr_r;
    logic [1:0] kc_r;

    logic [7:0] macs_done_r;

    logic signed [31:0] bias_r;
    logic signed [31:0] multiplier_r;
    logic [5:0]         shift_r;

    logic padding_w;
    logic signed [5:0] in_row_w;
    logic signed [5:0] in_col_w;
    logic [11:0] act_addr_w;
    logic [12:0] wgt_addr_w;
    logic [4:0]  bias_addr_w;
    logic [12:0] out_addr_w;

    logic        padding_pend;
    logic [11:0] act_addr_pend;
    logic [12:0] wgt_addr_pend;
    logic [3:0]  ic_pend;
    logic [1:0]  kr_pend;
    logic [1:0]  kc_pend;

    logic bias_re, mult_re, shift_re, act_re, wgt_re;

    logic signed [7:0]  act_rom_data;
    logic signed [7:0]  wgt_rom_data;
    logic signed [31:0] bias_rom_data;
    logic signed [31:0] mult_rom_data;
    logic signed [31:0] shift_rom_data;

    logic load_bias_i;
    logic mac_en;
    logic signed [7:0]  act_mac;
    logic signed [7:0]  wgt_mac;
    logic signed [15:0] prod_w;
    logic signed [31:0] acc_w;

    logic signed [63:0] wide_product;
    logic signed [63:0] rounding_offset;
    logic signed [63:0] rounded_product;
    logic signed [63:0] shifted_value;
    logic signed [31:0] zero_point_adjusted;
    logic signed [7:0]  sat_value;
    logic signed [7:0]  relu_value;

    logic operand_valid_c;
    logic last_mac;

    conv2_address_generator u_agen (
        .output_channel       (oc_r),
        .output_row           (or_r),
        .output_column        (oc_col_r),
        .input_channel        (ic_r),
        .kernel_row           (kr_r),
        .kernel_column        (kc_r),
        .input_row            (in_row_w),
        .input_column         (in_col_w),
        .padding              (padding_w),
        .pool1_read_address   (act_addr_w),
        .conv2_weight_address (wgt_addr_w),
        .conv2_bias_address   (bias_addr_w),
        .conv2_output_address (out_addr_w)
    );

    generate
        if (EXTERNAL_ACT) begin : g_ext_act
            assign act_rom_data = external_act_data;
        end else begin : g_int_act
            int8_sync_rom #(
                .DEPTH(4096),
                .ADDR_WIDTH(12),
                .MEM_FILE(ACT_MEM_FILE)
            ) u_act_rom (
                .clk          (clk),
                .read_enable  (act_re),
                .read_address (act_addr_w),
                .read_data    (act_rom_data)
            );
        end
    endgenerate

    int8_sync_rom #(
        .DEPTH(4608),
        .ADDR_WIDTH(13),
        .MEM_FILE(WGT_MEM_FILE)
    ) u_wgt_rom (
        .clk          (clk),
        .read_enable  (wgt_re),
        .read_address (wgt_addr_w),
        .read_data    (wgt_rom_data)
    );

    int32_sync_rom #(
        .DEPTH(32),
        .ADDR_WIDTH(5),
        .MEM_FILE(BIAS_MEM_FILE)
    ) u_bias_rom (
        .clk          (clk),
        .read_enable  (bias_re),
        .read_address (oc_r),
        .read_data    (bias_rom_data)
    );

    int32_sync_rom #(
        .DEPTH(32),
        .ADDR_WIDTH(5),
        .MEM_FILE(MULT_MEM_FILE)
    ) u_mult_rom (
        .clk          (clk),
        .read_enable  (mult_re),
        .read_address (oc_r),
        .read_data    (mult_rom_data)
    );

    int32_sync_rom #(
        .DEPTH(32),
        .ADDR_WIDTH(5),
        .MEM_FILE(SHIFT_MEM_FILE)
    ) u_shift_rom (
        .clk          (clk),
        .read_enable  (shift_re),
        .read_address (oc_r),
        .read_data    (shift_rom_data)
    );

    assign act_mac = padding_pend ? 8'sd0 : act_rom_data;
    assign wgt_mac = wgt_rom_data;

    logic signed [31:0] bias_for_mac;
    assign bias_for_mac = (state == ST_LOAD_BIAS) ? bias_rom_data : bias_r;

    int8_mac u_mac (
        .clk         (clk),
        .rst         (rst),
        .load_bias   (load_bias_i),
        .enable      (mac_en),
        .activation  (act_mac),
        .weight      (wgt_mac),
        .bias        (bias_for_mac),
        .product     (prod_w),
        .accumulator (acc_w)
    );

    requantize u_requant (
        .accumulator         (acc_w),
        .multiplier          (multiplier_r),
        .shift               (shift_r),
        .output_zero_point   (8'sd0),
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

    assign last_mac = (ic_r == 4'd15) && (kr_r == 2'd2) && (kc_r == 2'd2);
    assign operand_valid_c = (state == ST_WAIT_MAC);
    assign operand_valid = operand_valid_c;
    assign mac_enable = mac_en;
    assign load_bias = load_bias_i;
    assign act_read_enable = act_re;
    assign wgt_read_enable = wgt_re;

    assign current_input_channel      = (state == ST_WAIT_MAC) ? ic_pend : ic_r;
    assign current_kernel_row         = (state == ST_WAIT_MAC) ? kr_pend : kr_r;
    assign current_kernel_column      = (state == ST_WAIT_MAC) ? kc_pend : kc_r;
    assign current_activation_address = (state == ST_WAIT_MAC) ? act_addr_pend : act_addr_w;
    assign current_weight_address     = (state == ST_WAIT_MAC) ? wgt_addr_pend : wgt_addr_w;
    assign current_bias_address       = oc_r;
    assign current_output_address     = out_addr_w;
    assign current_padding            = (state == ST_WAIT_MAC) ? padding_pend : padding_w;
    assign current_activation_value   = act_mac;
    assign current_weight_value       = wgt_mac;
    assign current_product            = prod_w;
    assign current_accumulator        = acc_w;
    assign mac_count                  = macs_done_r;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= ST_IDLE;
            oc_r <= 5'd0;
            or_r <= 4'd0;
            oc_col_r <= 4'd0;
            ic_r <= 4'd0;
            kr_r <= 2'd0;
            kc_r <= 2'd0;
            macs_done_r <= 8'd0;
            bias_r <= 32'sd0;
            multiplier_r <= 32'sd0;
            shift_r <= 6'd0;
            padding_pend <= 1'b0;
            act_addr_pend <= 12'd0;
            wgt_addr_pend <= 13'd0;
            ic_pend <= 4'd0;
            kr_pend <= 2'd0;
            kc_pend <= 2'd0;
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
                    macs_done_r <= 8'd0;
                    ic_r <= 4'd0;
                    kr_r <= 2'd0;
                    kc_r <= 2'd0;
                    if (start) begin
                        busy <= 1'b1;
                        oc_r <= output_channel;
                        or_r <= output_row;
                        oc_col_r <= output_column;
                    end
                end
                ST_LOAD_BIAS: begin
                    bias_r       <= bias_rom_data;
                    multiplier_r <= mult_rom_data;
                    shift_r      <= shift_rom_data[5:0];
                    ic_r <= 4'd0;
                    kr_r <= 2'd0;
                    kc_r <= 2'd0;
                    macs_done_r <= 8'd0;
                end
                ST_ISSUE_OP: begin
                    padding_pend  <= padding_w;
                    act_addr_pend <= act_addr_w;
                    wgt_addr_pend <= wgt_addr_w;
                    ic_pend       <= ic_r;
                    kr_pend       <= kr_r;
                    kc_pend       <= kc_r;
                end
                ST_WAIT_MAC: begin
                    last_product <= prod_w;
                    macs_done_r  <= macs_done_r + 8'd1;
                    if (!last_mac) begin
                        if (kc_r == 2'd2) begin
                            kc_r <= 2'd0;
                            if (kr_r == 2'd2) begin
                                kr_r <= 2'd0;
                                ic_r <= ic_r + 4'd1;
                            end else begin
                                kr_r <= kr_r + 2'd1;
                            end
                        end else begin
                            kc_r <= kc_r + 2'd1;
                        end
                    end
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
        state_n     = state;
        load_bias_i = 1'b0;
        mac_en      = 1'b0;
        bias_re     = 1'b0;
        mult_re     = 1'b0;
        shift_re    = 1'b0;
        act_re      = 1'b0;
        wgt_re      = 1'b0;

        case (state)
            ST_IDLE: begin
                if (start)
                    state_n = ST_ISSUE_BIAS;
            end
            ST_ISSUE_BIAS: begin
                bias_re  = 1'b1;
                mult_re  = 1'b1;
                shift_re = 1'b1;
                state_n  = ST_WAIT_BIAS;
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
                act_re  = ~padding_w;
                wgt_re  = 1'b1;
                state_n = ST_WAIT_MAC;
            end
            ST_WAIT_MAC: begin
                mac_en = 1'b1;
                if (last_mac)
                    state_n = ST_REQUANTIZE;
                else
                    state_n = ST_ISSUE_OP;
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

    // synopsys translate_off
    always_ff @(posedge clk) begin
        if (!rst) begin
            if (mac_en && !load_bias_i) begin
                assert (operand_valid_c)
                    else $error("mac_enable without operand_valid");
                assert (macs_done_r < MACS_PER_OUTPUT)
                    else $error("mac_count would exceed 144");
            end
            if (done) begin
                assert (macs_done_r == MACS_PER_OUTPUT)
                    else $error("done with mac_count != 144");
            end
            if (act_re) begin
                assert (!padding_w)
                    else $error("activation read issued while padding");
                assert ({1'b0, act_addr_w} < 13'd4096)
                    else $error("activation address OOB");
            end
            if (wgt_re) begin
                assert ({1'b0, wgt_addr_w} < 14'd4608)
                    else $error("weight address OOB");
            end
            if (state == ST_REQUANTIZE) begin
                assert (macs_done_r == MACS_PER_OUTPUT)
                    else $error("requantize before 144 MACs");
            end
        end
    end
    // synopsys translate_on

endmodule
