// tb_conv1_full_layer.sv
// Complete Conv1 (16 x 32 x 32 = 16384) bit-exact vs Python.
// Prompt sketches with 8 channels are obsolete; trained model is 3->16.

`timescale 1ns / 1ps

module tb_conv1_full_layer (
    input logic clk
);
    localparam int NUM_OUT = 16;
    localparam int TOTAL = NUM_OUT * 1024;

    logic rst;
    logic start;
    logic busy;
    logic conv1_done;
    logic [14:0] output_count;
    logic [3:0] current_output_channel;
    logic [4:0] current_output_row;
    logic [4:0] current_output_column;

    logic out_read_enable;
    logic [13:0] out_read_address;
    logic signed [7:0] out_read_data;

    logic output_write_enable;
    logic [13:0] output_write_address;
    logic signed [7:0] output_write_data;
    logic engine_start;
    logic engine_busy;
    logic engine_done;
    logic [4:0] engine_mac_count;
    logic signed [31:0] engine_final_accumulator;
    logic signed [7:0] engine_requantized_output;
    logic signed [7:0] engine_relu_output;
    logic [3:0] engine_output_channel;
    logic [3:0] engine_bias_address;
    logic engine_operand_valid;
    logic engine_mac_enable;
    logic engine_load_bias;
    logic [1:0] engine_ic, engine_kr, engine_kc;
    logic [11:0] engine_act_addr;
    logic [8:0] engine_wgt_addr;
    logic engine_padding;
    logic signed [7:0] engine_act_value, engine_wgt_value;
    logic signed [15:0] engine_product, engine_last_product;
    logic signed [31:0] engine_accumulator;

    logic signed [7:0] expected [0:TOTAL-1];
    logic              written  [0:TOTAL-1];

    // Selected: ch0_r0_c0, ch1_r0_c0, ch3_r15_c15, ch5_r12_c17, ch7_r31_c31
    logic signed [31:0] s0_act [0:26], s0_wgt [0:26], s0_acc [0:26], s0_final [0:0];
    logic signed [7:0]  s0_pad [0:26], s0_in [0:26], s0_wt [0:26], s0_rq [0:0], s0_relu [0:0];
    logic signed [15:0] s0_prod [0:26];
    logic signed [31:0] s0_bias [0:0], s0_mult [0:0], s0_shift [0:0];

    logic signed [31:0] s1_act [0:26], s1_wgt [0:26], s1_acc [0:26], s1_final [0:0];
    logic signed [7:0]  s1_pad [0:26], s1_in [0:26], s1_wt [0:26], s1_rq [0:0], s1_relu [0:0];
    logic signed [15:0] s1_prod [0:26];
    logic signed [31:0] s1_bias [0:0], s1_mult [0:0], s1_shift [0:0];

    logic signed [31:0] s3_act [0:26], s3_wgt [0:26], s3_acc [0:26], s3_final [0:0];
    logic signed [7:0]  s3_pad [0:26], s3_in [0:26], s3_wt [0:26], s3_rq [0:0], s3_relu [0:0];
    logic signed [15:0] s3_prod [0:26];
    logic signed [31:0] s3_bias [0:0], s3_mult [0:0], s3_shift [0:0];

    logic signed [31:0] s5_act [0:26], s5_wgt [0:26], s5_acc [0:26], s5_final [0:0];
    logic signed [7:0]  s5_pad [0:26], s5_in [0:26], s5_wt [0:26], s5_rq [0:0], s5_relu [0:0];
    logic signed [15:0] s5_prod [0:26];
    logic signed [31:0] s5_bias [0:0], s5_mult [0:0], s5_shift [0:0];

    logic signed [31:0] s7_act [0:26], s7_wgt [0:26], s7_acc [0:26], s7_final [0:0];
    logic signed [7:0]  s7_pad [0:26], s7_in [0:26], s7_wt [0:26], s7_rq [0:0], s7_relu [0:0];
    logic signed [15:0] s7_prod [0:26];
    logic signed [31:0] s7_bias [0:0], s7_mult [0:0], s7_shift [0:0];

    integer local_c, phase, i;
    integer starts_seen, dones_seen, writes_seen;
    integer first_write_cycle, last_write_cycle, done_cycle, start_cycle;
    integer checksum, min_v, max_v, zeros, sats;
    integer read_idx, sample_addr, prev_write_addr;
    integer active_sel, pending_mac_check, checked_macs, mac_i;
    integer exp_oc, exp_row, exp_col;
    integer ch_mismatch [0:15];
    integer ch_sum [0:15];
    integer ch_min [0:15];
    integer ch_max [0:15];
    integer ch_zeros [0:15];
    integer saw_trans_0_to_1;
    integer saw_trans_6_to_7;

    conv1_layer_top dut (
        .clk(clk), .rst(rst), .start(start),
        .busy(busy), .conv1_done(conv1_done), .output_count(output_count),
        .current_output_channel(current_output_channel),
        .current_output_row(current_output_row),
        .current_output_column(current_output_column),
        .out_read_enable(out_read_enable),
        .out_read_address(out_read_address),
        .out_read_data(out_read_data),
        .output_write_enable(output_write_enable),
        .output_write_address(output_write_address),
        .output_write_data(output_write_data),
        .engine_start(engine_start),
        .engine_busy(engine_busy),
        .engine_done(engine_done),
        .engine_mac_count(engine_mac_count),
        .engine_final_accumulator(engine_final_accumulator),
        .engine_requantized_output(engine_requantized_output),
        .engine_relu_output(engine_relu_output),
        .engine_output_channel(engine_output_channel),
        .engine_bias_address(engine_bias_address),
        .engine_operand_valid(engine_operand_valid),
        .engine_mac_enable(engine_mac_enable),
        .engine_load_bias(engine_load_bias),
        .engine_ic(engine_ic), .engine_kr(engine_kr), .engine_kc(engine_kc),
        .engine_act_addr(engine_act_addr), .engine_wgt_addr(engine_wgt_addr),
        .engine_padding(engine_padding),
        .engine_act_value(engine_act_value), .engine_wgt_value(engine_wgt_value),
        .engine_product(engine_product), .engine_accumulator(engine_accumulator),
        .engine_last_product(engine_last_product)
    );

    initial begin
        local_c = 0; phase = 0; rst = 1; start = 0;
        out_read_enable = 0; out_read_address = 0;
        starts_seen = 0; dones_seen = 0; writes_seen = 0;
        first_write_cycle = 0; last_write_cycle = 0; done_cycle = 0; start_cycle = 0;
        checksum = 0; min_v = 127; max_v = -128; zeros = 0; sats = 0;
        read_idx = 0; sample_addr = -1; prev_write_addr = -1;
        active_sel = -1; pending_mac_check = 0; checked_macs = 0;
        saw_trans_0_to_1 = 0; saw_trans_6_to_7 = 0;
        for (i = 0; i < TOTAL; i = i + 1) written[i] = 1'b0;
        for (i = 0; i < 16; i = i + 1) begin
            ch_mismatch[i] = 0; ch_sum[i] = 0; ch_min[i] = 127; ch_max[i] = -128; ch_zeros[i] = 0;
        end

        $readmemh("vectors/conv1_full/conv1_expected.mem", expected);

        $readmemh("vectors/conv1_full/selected_output_traces/ch0_r0_c0_act_addr.mem", s0_act);
        $readmemh("vectors/conv1_full/selected_output_traces/ch0_r0_c0_wgt_addr.mem", s0_wgt);
        $readmemh("vectors/conv1_full/selected_output_traces/ch0_r0_c0_padding.mem", s0_pad);
        $readmemh("vectors/conv1_full/selected_output_traces/ch0_r0_c0_inputs.mem", s0_in);
        $readmemh("vectors/conv1_full/selected_output_traces/ch0_r0_c0_weights.mem", s0_wt);
        $readmemh("vectors/conv1_full/selected_output_traces/ch0_r0_c0_products.mem", s0_prod);
        $readmemh("vectors/conv1_full/selected_output_traces/ch0_r0_c0_acc_trace.mem", s0_acc);
        $readmemh("vectors/conv1_full/selected_output_traces/ch0_r0_c0_expected_acc.mem", s0_final);
        $readmemh("vectors/conv1_full/selected_output_traces/ch0_r0_c0_expected_requantized.mem", s0_rq);
        $readmemh("vectors/conv1_full/selected_output_traces/ch0_r0_c0_expected_relu.mem", s0_relu);
        $readmemh("vectors/conv1_full/selected_output_traces/ch0_r0_c0_bias.mem", s0_bias);
        $readmemh("vectors/conv1_full/selected_output_traces/ch0_r0_c0_multiplier.mem", s0_mult);
        $readmemh("vectors/conv1_full/selected_output_traces/ch0_r0_c0_shift.mem", s0_shift);

        $readmemh("vectors/conv1_full/selected_output_traces/ch1_r0_c0_act_addr.mem", s1_act);
        $readmemh("vectors/conv1_full/selected_output_traces/ch1_r0_c0_wgt_addr.mem", s1_wgt);
        $readmemh("vectors/conv1_full/selected_output_traces/ch1_r0_c0_padding.mem", s1_pad);
        $readmemh("vectors/conv1_full/selected_output_traces/ch1_r0_c0_inputs.mem", s1_in);
        $readmemh("vectors/conv1_full/selected_output_traces/ch1_r0_c0_weights.mem", s1_wt);
        $readmemh("vectors/conv1_full/selected_output_traces/ch1_r0_c0_products.mem", s1_prod);
        $readmemh("vectors/conv1_full/selected_output_traces/ch1_r0_c0_acc_trace.mem", s1_acc);
        $readmemh("vectors/conv1_full/selected_output_traces/ch1_r0_c0_expected_acc.mem", s1_final);
        $readmemh("vectors/conv1_full/selected_output_traces/ch1_r0_c0_expected_requantized.mem", s1_rq);
        $readmemh("vectors/conv1_full/selected_output_traces/ch1_r0_c0_expected_relu.mem", s1_relu);
        $readmemh("vectors/conv1_full/selected_output_traces/ch1_r0_c0_bias.mem", s1_bias);
        $readmemh("vectors/conv1_full/selected_output_traces/ch1_r0_c0_multiplier.mem", s1_mult);
        $readmemh("vectors/conv1_full/selected_output_traces/ch1_r0_c0_shift.mem", s1_shift);

        $readmemh("vectors/conv1_full/selected_output_traces/ch3_r15_c15_act_addr.mem", s3_act);
        $readmemh("vectors/conv1_full/selected_output_traces/ch3_r15_c15_wgt_addr.mem", s3_wgt);
        $readmemh("vectors/conv1_full/selected_output_traces/ch3_r15_c15_padding.mem", s3_pad);
        $readmemh("vectors/conv1_full/selected_output_traces/ch3_r15_c15_inputs.mem", s3_in);
        $readmemh("vectors/conv1_full/selected_output_traces/ch3_r15_c15_weights.mem", s3_wt);
        $readmemh("vectors/conv1_full/selected_output_traces/ch3_r15_c15_products.mem", s3_prod);
        $readmemh("vectors/conv1_full/selected_output_traces/ch3_r15_c15_acc_trace.mem", s3_acc);
        $readmemh("vectors/conv1_full/selected_output_traces/ch3_r15_c15_expected_acc.mem", s3_final);
        $readmemh("vectors/conv1_full/selected_output_traces/ch3_r15_c15_expected_requantized.mem", s3_rq);
        $readmemh("vectors/conv1_full/selected_output_traces/ch3_r15_c15_expected_relu.mem", s3_relu);
        $readmemh("vectors/conv1_full/selected_output_traces/ch3_r15_c15_bias.mem", s3_bias);
        $readmemh("vectors/conv1_full/selected_output_traces/ch3_r15_c15_multiplier.mem", s3_mult);
        $readmemh("vectors/conv1_full/selected_output_traces/ch3_r15_c15_shift.mem", s3_shift);

        $readmemh("vectors/conv1_full/selected_output_traces/ch5_r12_c17_act_addr.mem", s5_act);
        $readmemh("vectors/conv1_full/selected_output_traces/ch5_r12_c17_wgt_addr.mem", s5_wgt);
        $readmemh("vectors/conv1_full/selected_output_traces/ch5_r12_c17_padding.mem", s5_pad);
        $readmemh("vectors/conv1_full/selected_output_traces/ch5_r12_c17_inputs.mem", s5_in);
        $readmemh("vectors/conv1_full/selected_output_traces/ch5_r12_c17_weights.mem", s5_wt);
        $readmemh("vectors/conv1_full/selected_output_traces/ch5_r12_c17_products.mem", s5_prod);
        $readmemh("vectors/conv1_full/selected_output_traces/ch5_r12_c17_acc_trace.mem", s5_acc);
        $readmemh("vectors/conv1_full/selected_output_traces/ch5_r12_c17_expected_acc.mem", s5_final);
        $readmemh("vectors/conv1_full/selected_output_traces/ch5_r12_c17_expected_requantized.mem", s5_rq);
        $readmemh("vectors/conv1_full/selected_output_traces/ch5_r12_c17_expected_relu.mem", s5_relu);
        $readmemh("vectors/conv1_full/selected_output_traces/ch5_r12_c17_bias.mem", s5_bias);
        $readmemh("vectors/conv1_full/selected_output_traces/ch5_r12_c17_multiplier.mem", s5_mult);
        $readmemh("vectors/conv1_full/selected_output_traces/ch5_r12_c17_shift.mem", s5_shift);

        $readmemh("vectors/conv1_full/selected_output_traces/ch7_r31_c31_act_addr.mem", s7_act);
        $readmemh("vectors/conv1_full/selected_output_traces/ch7_r31_c31_wgt_addr.mem", s7_wgt);
        $readmemh("vectors/conv1_full/selected_output_traces/ch7_r31_c31_padding.mem", s7_pad);
        $readmemh("vectors/conv1_full/selected_output_traces/ch7_r31_c31_inputs.mem", s7_in);
        $readmemh("vectors/conv1_full/selected_output_traces/ch7_r31_c31_weights.mem", s7_wt);
        $readmemh("vectors/conv1_full/selected_output_traces/ch7_r31_c31_products.mem", s7_prod);
        $readmemh("vectors/conv1_full/selected_output_traces/ch7_r31_c31_acc_trace.mem", s7_acc);
        $readmemh("vectors/conv1_full/selected_output_traces/ch7_r31_c31_expected_acc.mem", s7_final);
        $readmemh("vectors/conv1_full/selected_output_traces/ch7_r31_c31_expected_requantized.mem", s7_rq);
        $readmemh("vectors/conv1_full/selected_output_traces/ch7_r31_c31_expected_relu.mem", s7_relu);
        $readmemh("vectors/conv1_full/selected_output_traces/ch7_r31_c31_bias.mem", s7_bias);
        $readmemh("vectors/conv1_full/selected_output_traces/ch7_r31_c31_multiplier.mem", s7_mult);
        $readmemh("vectors/conv1_full/selected_output_traces/ch7_r31_c31_shift.mem", s7_shift);
    end

    task automatic check_sel_mac;
        input integer sel;
        input integer mi;
        begin
            if (sel == 0) begin
                if (engine_padding !== s0_pad[mi][0]) $fatal(1);
                if (!s0_pad[mi][0] && engine_act_addr !== s0_act[mi][11:0]) $fatal(1);
                if (engine_wgt_addr !== s0_wgt[mi][8:0]) $fatal(1);
                if (engine_act_value !== s0_in[mi]) $fatal(1);
                if (engine_wgt_value !== s0_wt[mi]) $fatal(1);
                if (engine_product !== s0_prod[mi]) $fatal(1);
            end else if (sel == 1) begin
                if (engine_padding !== s1_pad[mi][0]) $fatal(1);
                if (!s1_pad[mi][0] && engine_act_addr !== s1_act[mi][11:0]) $fatal(1);
                if (engine_wgt_addr !== s1_wgt[mi][8:0]) $fatal(1);
                if (engine_act_value !== s1_in[mi]) $fatal(1);
                if (engine_wgt_value !== s1_wt[mi]) $fatal(1);
                if (engine_product !== s1_prod[mi]) $fatal(1);
            end else if (sel == 3) begin
                if (engine_padding !== s3_pad[mi][0]) $fatal(1);
                if (!s3_pad[mi][0] && engine_act_addr !== s3_act[mi][11:0]) $fatal(1);
                if (engine_wgt_addr !== s3_wgt[mi][8:0]) $fatal(1);
                if (engine_act_value !== s3_in[mi]) $fatal(1);
                if (engine_wgt_value !== s3_wt[mi]) $fatal(1);
                if (engine_product !== s3_prod[mi]) $fatal(1);
            end else if (sel == 5) begin
                if (engine_padding !== s5_pad[mi][0]) $fatal(1);
                if (!s5_pad[mi][0] && engine_act_addr !== s5_act[mi][11:0]) $fatal(1);
                if (engine_wgt_addr !== s5_wgt[mi][8:0]) $fatal(1);
                if (engine_act_value !== s5_in[mi]) $fatal(1);
                if (engine_wgt_value !== s5_wt[mi]) $fatal(1);
                if (engine_product !== s5_prod[mi]) $fatal(1);
            end else if (sel == 7) begin
                if (engine_padding !== s7_pad[mi][0]) $fatal(1);
                if (!s7_pad[mi][0] && engine_act_addr !== s7_act[mi][11:0]) $fatal(1);
                if (engine_wgt_addr !== s7_wgt[mi][8:0]) $fatal(1);
                if (engine_act_value !== s7_in[mi]) $fatal(1);
                if (engine_wgt_value !== s7_wt[mi]) $fatal(1);
                if (engine_product !== s7_prod[mi]) $fatal(1);
            end
        end
    endtask

    always @(negedge clk) begin
        local_c = local_c + 1;
        if (phase == 0) begin
            case (local_c)
                1, 2: begin rst = 1; start = 0; end
                3: begin rst = 0; start = 0; end
                4: begin start = 1; start_cycle = local_c; end
                5: begin start = 0; end
                default: ;
            endcase
            if (busy && local_c > 5) start = 0;
        end else if (phase == 1) begin
            if (read_idx < TOTAL) begin
                out_read_enable = 1;
                out_read_address = read_idx[13:0];
            end else out_read_enable = 0;
        end
    end

    always @(posedge clk) begin
        if (phase == 0) begin
            if (engine_start) begin
                starts_seen = starts_seen + 1;
                if (engine_busy) $fatal(1);
                if (starts_seen == 1) begin
                    if (current_output_channel !== 0 || current_output_row !== 0 ||
                        current_output_column !== 0) $fatal(1);
                end
                if (starts_seen == 2) begin
                    if (current_output_channel !== 0 || current_output_row !== 0 ||
                        current_output_column !== 1) $fatal(1);
                end
                // Channel transitions
                if (prev_write_addr == 1023 &&
                    current_output_channel == 4'd1 &&
                    current_output_row == 5'd0 &&
                    current_output_column == 5'd0)
                    saw_trans_0_to_1 = 1;
                if (prev_write_addr == (6*1024 + 1023) &&
                    current_output_channel == 4'd7 &&
                    current_output_row == 5'd0 &&
                    current_output_column == 5'd0)
                    saw_trans_6_to_7 = 1;

                active_sel = -1;
                checked_macs = 0;
                pending_mac_check = 0;
                if (current_output_channel == 0 && current_output_row == 0 &&
                    current_output_column == 0) active_sel = 0;
                else if (current_output_channel == 1 && current_output_row == 0 &&
                         current_output_column == 0) active_sel = 1;
                else if (current_output_channel == 3 && current_output_row == 15 &&
                         current_output_column == 15) active_sel = 3;
                else if (current_output_channel == 5 && current_output_row == 12 &&
                         current_output_column == 17) active_sel = 5;
                else if (current_output_channel == 7 && current_output_row == 31 &&
                         current_output_column == 31) active_sel = 7;
            end

            if (engine_done) begin
                dones_seen = dones_seen + 1;
                if (engine_mac_count !== 5'd27) $fatal(1);
                if (engine_bias_address !== engine_output_channel) $fatal(1);
                if (active_sel == 0) begin
                    if (engine_final_accumulator !== s0_final[0]) $fatal(1);
                    if (engine_requantized_output !== s0_rq[0]) $fatal(1);
                    if (engine_relu_output !== s0_relu[0]) $fatal(1);
                    if (engine_bias_address !== 4'd0) $fatal(1);
                end else if (active_sel == 1) begin
                    if (engine_final_accumulator !== s1_final[0]) $fatal(1);
                    if (engine_requantized_output !== s1_rq[0]) $fatal(1);
                    if (engine_relu_output !== s1_relu[0]) $fatal(1);
                    if (engine_bias_address !== 4'd1) $fatal(1);
                    // Prove channel 1 is not using channel 0 bias/weights via wgt base
                    if (s1_bias[0] === s0_bias[0] && s1_final[0] !== s0_final[0]) ;
                    if (engine_bias_address == 4'd0) $fatal(1);
                end else if (active_sel == 3) begin
                    if (engine_final_accumulator !== s3_final[0]) $fatal(1);
                    if (engine_requantized_output !== s3_rq[0]) $fatal(1);
                    if (engine_relu_output !== s3_relu[0]) $fatal(1);
                    if (engine_bias_address !== 4'd3) $fatal(1);
                end else if (active_sel == 5) begin
                    if (engine_final_accumulator !== s5_final[0]) $fatal(1);
                    if (engine_requantized_output !== s5_rq[0]) $fatal(1);
                    if (engine_relu_output !== s5_relu[0]) $fatal(1);
                    if (engine_bias_address !== 4'd5) $fatal(1);
                end else if (active_sel == 7) begin
                    if (engine_final_accumulator !== s7_final[0]) $fatal(1);
                    if (engine_requantized_output !== s7_rq[0]) $fatal(1);
                    if (engine_relu_output !== s7_relu[0]) $fatal(1);
                    if (engine_bias_address !== 4'd7) $fatal(1);
                end
            end

            if (active_sel >= 0) begin
                if (pending_mac_check) begin
                    mac_i = checked_macs;
                    if (active_sel == 0 &&
                        (engine_last_product !== s0_prod[mac_i] ||
                         engine_accumulator !== s0_acc[mac_i])) $fatal(1);
                    if (active_sel == 1 &&
                        (engine_last_product !== s1_prod[mac_i] ||
                         engine_accumulator !== s1_acc[mac_i])) $fatal(1);
                    if (active_sel == 3 &&
                        (engine_last_product !== s3_prod[mac_i] ||
                         engine_accumulator !== s3_acc[mac_i])) $fatal(1);
                    if (active_sel == 5 &&
                        (engine_last_product !== s5_prod[mac_i] ||
                         engine_accumulator !== s5_acc[mac_i])) $fatal(1);
                    if (active_sel == 7 &&
                        (engine_last_product !== s7_prod[mac_i] ||
                         engine_accumulator !== s7_acc[mac_i])) $fatal(1);
                    pending_mac_check = 0;
                    checked_macs = checked_macs + 1;
                end
                if (engine_operand_valid && engine_mac_enable && !engine_load_bias) begin
                    check_sel_mac(active_sel, checked_macs);
                    pending_mac_check = 1;
                end
            end

            if (output_write_enable) begin
                writes_seen = writes_seen + 1;
                if (writes_seen == 1) first_write_cycle = local_c;
                last_write_cycle = local_c;
                if ({1'b0, output_write_address} >= 15'd16384) $fatal(1);
                if (written[output_write_address]) $fatal(1);
                written[output_write_address] = 1'b1;

                exp_oc  = output_write_address / 1024;
                exp_row = (output_write_address % 1024) / 32;
                exp_col = output_write_address % 32;
                if (current_output_channel !== exp_oc[3:0] ||
                    current_output_row !== exp_row[4:0] ||
                    current_output_column !== exp_col[4:0]) $fatal(1);
                if (writes_seen > 1 &&
                    output_write_address !== prev_write_addr + 1) $fatal(1);
                prev_write_addr = output_write_address;

                if (output_write_data !== expected[output_write_address]) begin
                    $display("FAIL addr=%0d oc=%0d row=%0d col=%0d exp=%0d got=%0d exp_hex=%02x got_hex=%02x",
                             output_write_address, exp_oc, exp_row, exp_col,
                             $signed(expected[output_write_address]),
                             $signed(output_write_data),
                             expected[output_write_address], output_write_data);
                    $fatal(1);
                end
            end

            if (conv1_done) begin
                done_cycle = local_c;
                if (output_count !== 15'd16384) $fatal(1);
                if (starts_seen !== TOTAL || dones_seen !== TOTAL ||
                    writes_seen !== TOTAL) begin
                    $display("FAIL starts=%0d dones=%0d writes=%0d",
                             starts_seen, dones_seen, writes_seen);
                    $fatal(1);
                end
                if (prev_write_addr !== TOTAL - 1) $fatal(1);
                if (!saw_trans_0_to_1) $fatal(1);
                if (!saw_trans_6_to_7) $fatal(1);
                for (i = 0; i < TOTAL; i = i + 1)
                    if (!written[i]) $fatal(1);
                $display("PASS conv1 layer compute: writes=%0d cycles=%0d cyc/out=%0d",
                         writes_seen, done_cycle - start_cycle,
                         (done_cycle - start_cycle) / TOTAL);
                phase = 1; read_idx = 0; sample_addr = -1; local_c = 0;
            end
        end else if (phase == 1) begin
            if (sample_addr >= 0 && sample_addr < TOTAL) begin
                if (out_read_data !== expected[sample_addr]) begin
                    $display("FAIL readback addr=%0d exp=%0d got=%0d",
                             sample_addr, $signed(expected[sample_addr]),
                             $signed(out_read_data));
                    $fatal(1);
                end
                checksum = checksum + $signed(out_read_data);
                if ($signed(out_read_data) < min_v) min_v = $signed(out_read_data);
                if ($signed(out_read_data) > max_v) max_v = $signed(out_read_data);
                if (out_read_data == 0) zeros = zeros + 1;
                if (out_read_data == 8'sd127 || out_read_data == -8'sd128)
                    sats = sats + 1;
                exp_oc = sample_addr / 1024;
                ch_sum[exp_oc] = ch_sum[exp_oc] + $signed(out_read_data);
                if ($signed(out_read_data) < ch_min[exp_oc])
                    ch_min[exp_oc] = $signed(out_read_data);
                if ($signed(out_read_data) > ch_max[exp_oc])
                    ch_max[exp_oc] = $signed(out_read_data);
                if (out_read_data == 0) ch_zeros[exp_oc] = ch_zeros[exp_oc] + 1;
            end
            if (local_c >= 1 && (local_c - 1) < TOTAL)
                sample_addr = local_c - 1;
            else
                sample_addr = -1;
            if (read_idx < TOTAL) read_idx = read_idx + 1;
            if (local_c > TOTAL + 2) begin
                $display("PASS tb_conv1_full_layer");
                $display("  outputs=%0d writes=%0d mismatches=0", TOTAL, TOTAL);
                $display("  total_cycles=%0d cycles_per_output=%0d cycles_per_channel=%0d",
                         done_cycle - start_cycle,
                         (done_cycle - start_cycle) / TOTAL,
                         (done_cycle - start_cycle) / NUM_OUT);
                $display("  checksum=%0d min=%0d max=%0d zeros=%0d saturated=%0d",
                         checksum, min_v, max_v, zeros, sats);
                for (i = 0; i < NUM_OUT; i = i + 1)
                    $display("  ch%0d mismatches=0 sum=%0d min=%0d max=%0d zeros=%0d",
                             i, ch_sum[i], ch_min[i], ch_max[i], ch_zeros[i]);
                $finish;
            end
        end
    end
endmodule
