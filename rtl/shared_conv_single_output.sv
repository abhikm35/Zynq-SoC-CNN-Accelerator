// shared_conv_single_output.sv
// One reusable memory-driven convolution pixel engine for Conv1 and Conv2.
//
// Trained sizes:
//   Conv1: 3->16, 32x32, 27 MACs
//   Conv2: 16->32, 16x16, 144 MACs
//
// Same FSM as the verified Conv1/Conv2 single-output engines:
//   IDLE -> ISSUE_BIAS -> WAIT_BIAS -> LOAD_BIAS
//       -> ISSUE_OP -> WAIT_MAC  (x MACs_PER_OUTPUT)
//       -> REQUANTIZE -> DONE
//
// Activations are always EXTERNAL (1-cycle sync). Weight/bias/mult/shift
// come from layer-selected ROMs instantiated here (separate banks).

`timescale 1ns / 1ps

module shared_conv_single_output #(
    parameter CONV1_WGT_MEM   = "vectors/conv1_memory/conv1_weights.mem",
    parameter CONV1_BIAS_MEM  = "vectors/conv1_memory/conv1_biases.mem",
    parameter CONV1_MULT_MEM  = "vectors/conv1_memory/conv1_multipliers.mem",
    parameter CONV1_SHIFT_MEM = "vectors/conv1_memory/conv1_shifts.mem",
    parameter CONV2_WGT_MEM   = "vectors/conv2/conv2_weights.mem",
    parameter CONV2_BIAS_MEM  = "vectors/conv2/conv2_biases.mem",
    parameter CONV2_MULT_MEM  = "vectors/conv2/conv2_multipliers.mem",
    parameter CONV2_SHIFT_MEM = "vectors/conv2/conv2_shifts.mem"
) (
    input  logic                clk,
    input  logic                rst,
    input  logic                start,
    input  logic                layer_is_conv2,   // 0=Conv1, 1=Conv2 (stable while busy)

    input  logic [4:0]          output_channel,
    input  logic [4:0]          output_row,
    input  logic [4:0]          output_column,

    input  logic signed [7:0]   external_act_data,

    output logic                busy,
    output logic                done,
    output logic signed [31:0]  final_accumulator,
    output logic signed [7:0]   requantized_output,
    output logic signed [7:0]   relu_output,
    output logic [7:0]          mac_count,

    output logic [3:0]          current_input_channel,
    output logic [1:0]          current_kernel_row,
    output logic [1:0]          current_kernel_column,
    output logic [11:0]         current_activation_address,
    output logic [12:0]         current_weight_address,
    output logic [4:0]          current_bias_address,
    output logic [13:0]         current_output_address,
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
    output logic signed [15:0]  last_product,
    output logic                dbg_layer_is_conv2
);

    localparam logic [3:0] ST_IDLE       = 4'd0;
    localparam logic [3:0] ST_ISSUE_BIAS = 4'd1;
    localparam logic [3:0] ST_WAIT_BIAS  = 4'd2;
    localparam logic [3:0] ST_LOAD_BIAS  = 4'd3;
    localparam logic [3:0] ST_ISSUE_OP   = 4'd4;
    localparam logic [3:0] ST_WAIT_MAC   = 4'd5;
    localparam logic [3:0] ST_REQUANTIZE = 4'd6;
    localparam logic [3:0] ST_DONE       = 4'd7;

    logic [3:0] state, state_n;

    logic layer_r; // latched at start
    logic [4:0] oc_r, or_r, oc_col_r;
    logic [3:0] ic_r;
    logic [1:0] kr_r, kc_r;
    logic [7:0] macs_done_r;

    logic signed [31:0] bias_r, multiplier_r;
    logic [5:0] shift_r;

    logic padding_w;
    logic signed [6:0] in_row_w, in_col_w;
    logic [11:0] act_addr_w;
    logic [12:0] wgt_addr_w;
    logic [4:0]  bias_addr_w;
    logic [13:0] out_addr_w;

    logic        padding_pend;
    logic [11:0] act_addr_pend;
    logic [12:0] wgt_addr_pend;
    logic [3:0]  ic_pend;
    logic [1:0]  kr_pend, kc_pend;

    logic bias_re, mult_re, shift_re, act_re, wgt_re;
    logic load_bias_i, mac_en;
    logic signed [7:0] act_mac, wgt_mac;
    logic signed [15:0] prod_w;
    logic signed [31:0] acc_w;

    logic signed [63:0] wide_product, rounding_offset, rounded_product, shifted_value;
    logic signed [31:0] zero_point_adjusted;
    logic signed [7:0] sat_value, relu_value;

    logic operand_valid_c, last_mac;
    logic [3:0] last_ic;
    logic [7:0] macs_target;

    assign last_ic     = layer_r ? 4'd15 : 4'd2;
    assign macs_target = layer_r ? 8'd144 : 8'd27;
    assign last_mac    = (ic_r == last_ic) && (kr_r == 2'd2) && (kc_r == 2'd2);
    assign dbg_layer_is_conv2 = layer_r;

    shared_conv_address_generator u_agen (
        .layer_is_conv2      (layer_r),
        .output_channel      (oc_r),
        .output_row          (or_r),
        .output_column       (oc_col_r),
        .input_channel       (ic_r),
        .kernel_row          (kr_r),
        .kernel_column       (kc_r),
        .input_row           (in_row_w),
        .input_column        (in_col_w),
        .padding             (padding_w),
        .activation_address  (act_addr_w),
        .weight_address      (wgt_addr_w),
        .bias_address        (bias_addr_w),
        .output_address      (out_addr_w)
    );

    // Separate parameter banks; mux by layer_r
    logic signed [7:0]  wgt1, wgt2, wgt_rom_data;
    logic signed [31:0] bias1, bias2, bias_rom_data;
    logic signed [31:0] mult1, mult2, mult_rom_data;
    logic signed [31:0] sh1, sh2, shift_rom_data;

    logic wgt1_re, wgt2_re, bias1_re, bias2_re, mult1_re, mult2_re, sh1_re, sh2_re;

    assign wgt1_re  = wgt_re  & ~layer_r;
    assign wgt2_re  = wgt_re  &  layer_r;
    assign bias1_re = bias_re & ~layer_r;
    assign bias2_re = bias_re &  layer_r;
    assign mult1_re = mult_re & ~layer_r;
    assign mult2_re = mult_re &  layer_r;
    assign sh1_re   = shift_re & ~layer_r;
    assign sh2_re   = shift_re &  layer_r;

    assign wgt_rom_data  = layer_r ? wgt2  : wgt1;
    assign bias_rom_data = layer_r ? bias2 : bias1;
    assign mult_rom_data = layer_r ? mult2 : mult1;
    assign shift_rom_data= layer_r ? sh2   : sh1;

    int8_sync_rom #(.DEPTH(432), .ADDR_WIDTH(9), .MEM_FILE(CONV1_WGT_MEM)) u_wgt1 (
        .clk(clk), .read_enable(wgt1_re),
        .read_address(wgt_addr_w[8:0]), .read_data(wgt1)
    );
    int8_sync_rom #(.DEPTH(4608), .ADDR_WIDTH(13), .MEM_FILE(CONV2_WGT_MEM)) u_wgt2 (
        .clk(clk), .read_enable(wgt2_re),
        .read_address(wgt_addr_w), .read_data(wgt2)
    );
    int32_sync_rom #(.DEPTH(16), .ADDR_WIDTH(4), .MEM_FILE(CONV1_BIAS_MEM)) u_bias1 (
        .clk(clk), .read_enable(bias1_re),
        .read_address(oc_r[3:0]), .read_data(bias1)
    );
    int32_sync_rom #(.DEPTH(32), .ADDR_WIDTH(5), .MEM_FILE(CONV2_BIAS_MEM)) u_bias2 (
        .clk(clk), .read_enable(bias2_re),
        .read_address(oc_r), .read_data(bias2)
    );
    int32_sync_rom #(.DEPTH(16), .ADDR_WIDTH(4), .MEM_FILE(CONV1_MULT_MEM)) u_mult1 (
        .clk(clk), .read_enable(mult1_re),
        .read_address(oc_r[3:0]), .read_data(mult1)
    );
    int32_sync_rom #(.DEPTH(32), .ADDR_WIDTH(5), .MEM_FILE(CONV2_MULT_MEM)) u_mult2 (
        .clk(clk), .read_enable(mult2_re),
        .read_address(oc_r), .read_data(mult2)
    );
    int32_sync_rom #(.DEPTH(16), .ADDR_WIDTH(4), .MEM_FILE(CONV1_SHIFT_MEM)) u_sh1 (
        .clk(clk), .read_enable(sh1_re),
        .read_address(oc_r[3:0]), .read_data(sh1)
    );
    int32_sync_rom #(.DEPTH(32), .ADDR_WIDTH(5), .MEM_FILE(CONV2_SHIFT_MEM)) u_sh2 (
        .clk(clk), .read_enable(sh2_re),
        .read_address(oc_r), .read_data(sh2)
    );

    assign act_mac = padding_pend ? 8'sd0 : external_act_data;
    assign wgt_mac = wgt_rom_data;

    logic signed [31:0] bias_for_mac;
    assign bias_for_mac = (state == ST_LOAD_BIAS) ? bias_rom_data : bias_r;

    int8_mac u_mac (
        .clk(clk), .rst(rst),
        .load_bias(load_bias_i), .enable(mac_en),
        .activation(act_mac), .weight(wgt_mac), .bias(bias_for_mac),
        .product(prod_w), .accumulator(acc_w)
    );

    requantize u_requant (
        .accumulator(acc_w), .multiplier(multiplier_r), .shift(shift_r),
        .output_zero_point(8'sd0),
        .wide_product(wide_product), .rounding_offset(rounding_offset),
        .rounded_product(rounded_product), .shifted_value(shifted_value),
        .zero_point_adjusted(zero_point_adjusted), .saturated_value(sat_value)
    );

    relu_int8 u_relu (
        .in_value(sat_value), .zero_point(8'sd0), .out_value(relu_value)
    );

    assign operand_valid_c = (state == ST_WAIT_MAC);
    assign operand_valid   = operand_valid_c;
    assign mac_enable      = mac_en;
    assign load_bias       = load_bias_i;
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
            layer_r <= 1'b0;
            oc_r <= 5'd0; or_r <= 5'd0; oc_col_r <= 5'd0;
            ic_r <= 4'd0; kr_r <= 2'd0; kc_r <= 2'd0;
            macs_done_r <= 8'd0;
            bias_r <= 32'sd0; multiplier_r <= 32'sd0; shift_r <= 6'd0;
            padding_pend <= 1'b0;
            act_addr_pend <= 12'd0; wgt_addr_pend <= 13'd0;
            ic_pend <= 4'd0; kr_pend <= 2'd0; kc_pend <= 2'd0;
            final_accumulator <= 32'sd0;
            requantized_output <= 8'sd0; relu_output <= 8'sd0;
            last_product <= 16'sd0;
            done <= 1'b0; busy <= 1'b0;
        end else begin
            state <= state_n;
            done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    busy <= 1'b0;
                    macs_done_r <= 8'd0;
                    ic_r <= 4'd0; kr_r <= 2'd0; kc_r <= 2'd0;
                    if (start) begin
                        busy <= 1'b1;
                        layer_r <= layer_is_conv2;
                        oc_r <= output_channel;
                        or_r <= output_row;
                        oc_col_r <= output_column;
                    end
                end
                ST_LOAD_BIAS: begin
                    bias_r       <= bias_rom_data;
                    multiplier_r <= mult_rom_data;
                    shift_r      <= shift_rom_data[5:0];
                    ic_r <= 4'd0; kr_r <= 2'd0; kc_r <= 2'd0;
                    macs_done_r <= 8'd0;
                end
                ST_ISSUE_OP: begin
                    padding_pend  <= padding_w;
                    act_addr_pend <= act_addr_w;
                    wgt_addr_pend <= wgt_addr_w;
                    ic_pend <= ic_r; kr_pend <= kr_r; kc_pend <= kc_r;
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
        state_n = state;
        load_bias_i = 1'b0; mac_en = 1'b0;
        bias_re = 1'b0; mult_re = 1'b0; shift_re = 1'b0;
        act_re = 1'b0; wgt_re = 1'b0;

        unique case (state)
            ST_IDLE: if (start) state_n = ST_ISSUE_BIAS;
            ST_ISSUE_BIAS: begin
                bias_re = 1'b1; mult_re = 1'b1; shift_re = 1'b1;
                state_n = ST_WAIT_BIAS;
            end
            ST_WAIT_BIAS: state_n = ST_LOAD_BIAS;
            ST_LOAD_BIAS: begin
                load_bias_i = 1'b1; mac_en = 1'b1;
                state_n = ST_ISSUE_OP;
            end
            ST_ISSUE_OP: begin
                act_re = ~padding_w; wgt_re = 1'b1;
                state_n = ST_WAIT_MAC;
            end
            ST_WAIT_MAC: begin
                mac_en = 1'b1;
                state_n = last_mac ? ST_REQUANTIZE : ST_ISSUE_OP;
            end
            ST_REQUANTIZE: state_n = ST_DONE;
            ST_DONE: state_n = ST_IDLE;
            default: state_n = ST_IDLE;
        endcase
    end

    // synopsys translate_off
    always_ff @(posedge clk) begin
        if (!rst) begin
            if (busy && (layer_is_conv2 !== layer_r))
                $error("layer_is_conv2 changed while shared_conv_single_output busy");
            if (done && (macs_done_r != macs_target))
                $error("done with mac_count=%0d expected=%0d", macs_done_r, macs_target);
            if (act_re && padding_w)
                $error("activation read while padding");
        end
    end
    // synopsys translate_on

    /* verilator lint_off UNUSED */
    wire unused_coords = |{in_row_w, in_col_w, bias_addr_w};
    /* verilator lint_on UNUSED */

endmodule
