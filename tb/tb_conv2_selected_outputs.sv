// tb_conv2_selected_outputs.sv
// Bit-exact 144-MAC checks for selected Conv2 pixels (trained 16->32).

`timescale 1ns / 1ps

module tb_conv2_selected_outputs (
    input logic clk
);
    logic rst, start;
    logic [4:0] output_channel;
    logic [3:0] output_row, output_column;

    logic busy, done;
    logic signed [31:0] final_accumulator;
    logic signed [7:0]  requantized_output, relu_output;
    logic [7:0] mac_count;

    logic [3:0] current_input_channel;
    logic [1:0] current_kernel_row, current_kernel_column;
    logic [11:0] current_activation_address;
    logic [12:0] current_weight_address;
    logic [4:0]  current_bias_address;
    logic [12:0] current_output_address;
    logic current_padding;
    logic signed [7:0]  current_activation_value, current_weight_value;
    logic signed [15:0] current_product;
    logic signed [31:0] current_accumulator;
    logic operand_valid, mac_enable, load_bias;
    logic act_read_enable, wgt_read_enable;
    logic signed [15:0] last_product;

    logic signed [31:0] exp_act_addr [0:143];
    logic signed [31:0] exp_wgt_addr [0:143];
    logic signed [7:0]  exp_pad      [0:143];
    logic signed [7:0]  exp_act      [0:143];
    logic signed [7:0]  exp_wgt      [0:143];
    logic signed [15:0] exp_prod     [0:143];
    logic signed [31:0] exp_acc      [0:143];
    logic signed [31:0] exp_final    [0:0];
    logic signed [7:0]  exp_rq       [0:0];
    logic signed [7:0]  exp_relu     [0:0];
    logic signed [31:0] exp_out_addr [0:0];

    integer local_c, phase, mac_i;
    integer pending_mac_check, checked_macs, bias_loads, saw_issue;
    integer cycle_start, cycle_done;

    conv2_memory_single_output dut (
        .clk(clk), .rst(rst), .start(start),
        .output_channel(output_channel),
        .output_row(output_row),
        .output_column(output_column),
        .external_act_data(8'sd0),
        .busy(busy), .done(done),
        .final_accumulator(final_accumulator),
        .requantized_output(requantized_output),
        .relu_output(relu_output),
        .mac_count(mac_count),
        .current_input_channel(current_input_channel),
        .current_kernel_row(current_kernel_row),
        .current_kernel_column(current_kernel_column),
        .current_activation_address(current_activation_address),
        .current_weight_address(current_weight_address),
        .current_bias_address(current_bias_address),
        .current_output_address(current_output_address),
        .current_padding(current_padding),
        .current_activation_value(current_activation_value),
        .current_weight_value(current_weight_value),
        .current_product(current_product),
        .current_accumulator(current_accumulator),
        .operand_valid(operand_valid),
        .mac_enable(mac_enable),
        .load_bias(load_bias),
        .act_read_enable(act_read_enable),
        .wgt_read_enable(wgt_read_enable),
        .last_product(last_product)
    );

    task automatic load_case;
        input integer idx;
        begin
            case (idx)
                0: begin
                    output_channel=5'd0; output_row=4'd0; output_column=4'd0;
                    $readmemh("vectors/conv2/selected_output_traces/ch0_r0_c0_act_addr.mem", exp_act_addr);
                    $readmemh("vectors/conv2/selected_output_traces/ch0_r0_c0_wgt_addr.mem", exp_wgt_addr);
                    $readmemh("vectors/conv2/selected_output_traces/ch0_r0_c0_padding.mem", exp_pad);
                    $readmemh("vectors/conv2/selected_output_traces/ch0_r0_c0_inputs.mem", exp_act);
                    $readmemh("vectors/conv2/selected_output_traces/ch0_r0_c0_weights.mem", exp_wgt);
                    $readmemh("vectors/conv2/selected_output_traces/ch0_r0_c0_products.mem", exp_prod);
                    $readmemh("vectors/conv2/selected_output_traces/ch0_r0_c0_acc_trace.mem", exp_acc);
                    $readmemh("vectors/conv2/selected_output_traces/ch0_r0_c0_expected_acc.mem", exp_final);
                    $readmemh("vectors/conv2/selected_output_traces/ch0_r0_c0_expected_requantized.mem", exp_rq);
                    $readmemh("vectors/conv2/selected_output_traces/ch0_r0_c0_expected_relu.mem", exp_relu);
                    $readmemh("vectors/conv2/selected_output_traces/ch0_r0_c0_out_addr.mem", exp_out_addr);
                end
                1: begin
                    output_channel=5'd0; output_row=4'd7; output_column=4'd7;
                    $readmemh("vectors/conv2/selected_output_traces/ch0_r7_c7_act_addr.mem", exp_act_addr);
                    $readmemh("vectors/conv2/selected_output_traces/ch0_r7_c7_wgt_addr.mem", exp_wgt_addr);
                    $readmemh("vectors/conv2/selected_output_traces/ch0_r7_c7_padding.mem", exp_pad);
                    $readmemh("vectors/conv2/selected_output_traces/ch0_r7_c7_inputs.mem", exp_act);
                    $readmemh("vectors/conv2/selected_output_traces/ch0_r7_c7_weights.mem", exp_wgt);
                    $readmemh("vectors/conv2/selected_output_traces/ch0_r7_c7_products.mem", exp_prod);
                    $readmemh("vectors/conv2/selected_output_traces/ch0_r7_c7_acc_trace.mem", exp_acc);
                    $readmemh("vectors/conv2/selected_output_traces/ch0_r7_c7_expected_acc.mem", exp_final);
                    $readmemh("vectors/conv2/selected_output_traces/ch0_r7_c7_expected_requantized.mem", exp_rq);
                    $readmemh("vectors/conv2/selected_output_traces/ch0_r7_c7_expected_relu.mem", exp_relu);
                    $readmemh("vectors/conv2/selected_output_traces/ch0_r7_c7_out_addr.mem", exp_out_addr);
                end
                2: begin
                    output_channel=5'd1; output_row=4'd0; output_column=4'd0;
                    $readmemh("vectors/conv2/selected_output_traces/ch1_r0_c0_act_addr.mem", exp_act_addr);
                    $readmemh("vectors/conv2/selected_output_traces/ch1_r0_c0_wgt_addr.mem", exp_wgt_addr);
                    $readmemh("vectors/conv2/selected_output_traces/ch1_r0_c0_padding.mem", exp_pad);
                    $readmemh("vectors/conv2/selected_output_traces/ch1_r0_c0_inputs.mem", exp_act);
                    $readmemh("vectors/conv2/selected_output_traces/ch1_r0_c0_weights.mem", exp_wgt);
                    $readmemh("vectors/conv2/selected_output_traces/ch1_r0_c0_products.mem", exp_prod);
                    $readmemh("vectors/conv2/selected_output_traces/ch1_r0_c0_acc_trace.mem", exp_acc);
                    $readmemh("vectors/conv2/selected_output_traces/ch1_r0_c0_expected_acc.mem", exp_final);
                    $readmemh("vectors/conv2/selected_output_traces/ch1_r0_c0_expected_requantized.mem", exp_rq);
                    $readmemh("vectors/conv2/selected_output_traces/ch1_r0_c0_expected_relu.mem", exp_relu);
                    $readmemh("vectors/conv2/selected_output_traces/ch1_r0_c0_out_addr.mem", exp_out_addr);
                end
                3: begin
                    output_channel=5'd5; output_row=4'd7; output_column=4'd10;
                    $readmemh("vectors/conv2/selected_output_traces/ch5_r7_c10_act_addr.mem", exp_act_addr);
                    $readmemh("vectors/conv2/selected_output_traces/ch5_r7_c10_wgt_addr.mem", exp_wgt_addr);
                    $readmemh("vectors/conv2/selected_output_traces/ch5_r7_c10_padding.mem", exp_pad);
                    $readmemh("vectors/conv2/selected_output_traces/ch5_r7_c10_inputs.mem", exp_act);
                    $readmemh("vectors/conv2/selected_output_traces/ch5_r7_c10_weights.mem", exp_wgt);
                    $readmemh("vectors/conv2/selected_output_traces/ch5_r7_c10_products.mem", exp_prod);
                    $readmemh("vectors/conv2/selected_output_traces/ch5_r7_c10_acc_trace.mem", exp_acc);
                    $readmemh("vectors/conv2/selected_output_traces/ch5_r7_c10_expected_acc.mem", exp_final);
                    $readmemh("vectors/conv2/selected_output_traces/ch5_r7_c10_expected_requantized.mem", exp_rq);
                    $readmemh("vectors/conv2/selected_output_traces/ch5_r7_c10_expected_relu.mem", exp_relu);
                    $readmemh("vectors/conv2/selected_output_traces/ch5_r7_c10_out_addr.mem", exp_out_addr);
                end
                4: begin
                    output_channel=5'd9; output_row=4'd12; output_column=4'd4;
                    $readmemh("vectors/conv2/selected_output_traces/ch9_r12_c4_act_addr.mem", exp_act_addr);
                    $readmemh("vectors/conv2/selected_output_traces/ch9_r12_c4_wgt_addr.mem", exp_wgt_addr);
                    $readmemh("vectors/conv2/selected_output_traces/ch9_r12_c4_padding.mem", exp_pad);
                    $readmemh("vectors/conv2/selected_output_traces/ch9_r12_c4_inputs.mem", exp_act);
                    $readmemh("vectors/conv2/selected_output_traces/ch9_r12_c4_weights.mem", exp_wgt);
                    $readmemh("vectors/conv2/selected_output_traces/ch9_r12_c4_products.mem", exp_prod);
                    $readmemh("vectors/conv2/selected_output_traces/ch9_r12_c4_acc_trace.mem", exp_acc);
                    $readmemh("vectors/conv2/selected_output_traces/ch9_r12_c4_expected_acc.mem", exp_final);
                    $readmemh("vectors/conv2/selected_output_traces/ch9_r12_c4_expected_requantized.mem", exp_rq);
                    $readmemh("vectors/conv2/selected_output_traces/ch9_r12_c4_expected_relu.mem", exp_relu);
                    $readmemh("vectors/conv2/selected_output_traces/ch9_r12_c4_out_addr.mem", exp_out_addr);
                end
                5: begin
                    output_channel=5'd15; output_row=4'd15; output_column=4'd15;
                    $readmemh("vectors/conv2/selected_output_traces/ch15_r15_c15_act_addr.mem", exp_act_addr);
                    $readmemh("vectors/conv2/selected_output_traces/ch15_r15_c15_wgt_addr.mem", exp_wgt_addr);
                    $readmemh("vectors/conv2/selected_output_traces/ch15_r15_c15_padding.mem", exp_pad);
                    $readmemh("vectors/conv2/selected_output_traces/ch15_r15_c15_inputs.mem", exp_act);
                    $readmemh("vectors/conv2/selected_output_traces/ch15_r15_c15_weights.mem", exp_wgt);
                    $readmemh("vectors/conv2/selected_output_traces/ch15_r15_c15_products.mem", exp_prod);
                    $readmemh("vectors/conv2/selected_output_traces/ch15_r15_c15_acc_trace.mem", exp_acc);
                    $readmemh("vectors/conv2/selected_output_traces/ch15_r15_c15_expected_acc.mem", exp_final);
                    $readmemh("vectors/conv2/selected_output_traces/ch15_r15_c15_expected_requantized.mem", exp_rq);
                    $readmemh("vectors/conv2/selected_output_traces/ch15_r15_c15_expected_relu.mem", exp_relu);
                    $readmemh("vectors/conv2/selected_output_traces/ch15_r15_c15_out_addr.mem", exp_out_addr);
                end
                default: begin
                    output_channel=5'd31; output_row=4'd15; output_column=4'd15;
                    $readmemh("vectors/conv2/selected_output_traces/ch31_r15_c15_act_addr.mem", exp_act_addr);
                    $readmemh("vectors/conv2/selected_output_traces/ch31_r15_c15_wgt_addr.mem", exp_wgt_addr);
                    $readmemh("vectors/conv2/selected_output_traces/ch31_r15_c15_padding.mem", exp_pad);
                    $readmemh("vectors/conv2/selected_output_traces/ch31_r15_c15_inputs.mem", exp_act);
                    $readmemh("vectors/conv2/selected_output_traces/ch31_r15_c15_weights.mem", exp_wgt);
                    $readmemh("vectors/conv2/selected_output_traces/ch31_r15_c15_products.mem", exp_prod);
                    $readmemh("vectors/conv2/selected_output_traces/ch31_r15_c15_acc_trace.mem", exp_acc);
                    $readmemh("vectors/conv2/selected_output_traces/ch31_r15_c15_expected_acc.mem", exp_final);
                    $readmemh("vectors/conv2/selected_output_traces/ch31_r15_c15_expected_requantized.mem", exp_rq);
                    $readmemh("vectors/conv2/selected_output_traces/ch31_r15_c15_expected_relu.mem", exp_relu);
                    $readmemh("vectors/conv2/selected_output_traces/ch31_r15_c15_out_addr.mem", exp_out_addr);
                end
            endcase
        end
    endtask

    task automatic fail_mac;
        input integer mi;
        begin
            $display("FAIL phase%0d cycle%0d MAC %0d", phase, local_c, mi);
            $display("  oc=%0d r=%0d c=%0d ic=%0d kr=%0d kc=%0d",
                     output_channel, output_row, output_column,
                     current_input_channel, current_kernel_row, current_kernel_column);
            $display("  act_addr exp=%0d got=%0d  wgt_addr exp=%0d got=%0d",
                     exp_act_addr[mi], current_activation_address,
                     exp_wgt_addr[mi], current_weight_address);
            $display("  pad exp=%0d got=%0d act exp=%0d got=%0d wgt exp=%0d got=%0d",
                     exp_pad[mi], current_padding,
                     $signed(exp_act[mi]), $signed(current_activation_value),
                     $signed(exp_wgt[mi]), $signed(current_weight_value));
            $display("  prod exp=%0d got=%0d acc exp=%0d got=%0d",
                     $signed(exp_prod[mi]), $signed(current_product),
                     $signed(exp_acc[mi]), $signed(current_accumulator));
            $fatal(1);
        end
    endtask

    initial begin
        local_c=0; phase=0; rst=1; start=0;
        pending_mac_check=0; checked_macs=0; bias_loads=0; saw_issue=0;
        cycle_start=0; cycle_done=0;
        load_case(0);
    end

    always @(negedge clk) begin
        local_c = local_c + 1;
        if (phase < 7) begin
            case (local_c)
                1, 2: begin rst=1; start=0; end
                3: begin rst=0; end
                4: begin start=1; cycle_start=local_c; end
                5: begin start=0; end
                default: ;
            endcase
            if (busy && local_c > 5) start = 0;
        end
    end

    always @(posedge clk) begin
        if (phase < 7) begin
            if (mac_enable && !load_bias && !operand_valid)
                $fatal(1, "mac_enable without operand_valid");
            if (act_read_enable && current_padding)
                $fatal(1, "act read while padding");
            if (mac_count > 8'd144)
                $fatal(1, "mac_count > 144");
            if (mac_enable && load_bias)
                bias_loads = bias_loads + 1;

            if (pending_mac_check) begin
                mac_i = checked_macs;
                if (last_product !== exp_prod[mac_i]) fail_mac(mac_i);
                if (current_accumulator !== exp_acc[mac_i]) fail_mac(mac_i);
                pending_mac_check = 0;
                checked_macs = checked_macs + 1;
            end

            if (operand_valid && mac_enable && !load_bias) begin
                mac_i = checked_macs;
                if (current_padding !== exp_pad[mac_i][0]) fail_mac(mac_i);
                if (exp_pad[mac_i][0] == 1'b0) begin
                    if (current_activation_address !== exp_act_addr[mac_i][11:0])
                        fail_mac(mac_i);
                end
                if (current_weight_address !== exp_wgt_addr[mac_i][12:0]) fail_mac(mac_i);
                if (current_bias_address !== output_channel) fail_mac(mac_i);
                if (current_activation_value !== exp_act[mac_i]) fail_mac(mac_i);
                if (current_weight_value !== exp_wgt[mac_i]) fail_mac(mac_i);
                if (current_product !== exp_prod[mac_i]) fail_mac(mac_i);
                if (!saw_issue) $fatal(1, "MAC before operand issue");
                pending_mac_check = 1;
            end

            if (wgt_read_enable) saw_issue = 1;

            if (done) begin
                cycle_done = local_c;
                if (mac_count !== 8'd144) $fatal(1, "done mac_count=%0d", mac_count);
                if (bias_loads != 1) $fatal(1, "bias_loads=%0d", bias_loads);
                if (final_accumulator !== exp_final[0])
                    $fatal(1, "final_acc mismatch");
                if (requantized_output !== exp_rq[0])
                    $fatal(1, "requant mismatch");
                if (relu_output !== exp_relu[0])
                    $fatal(1, "relu mismatch");
                if (current_output_address !== exp_out_addr[0][12:0])
                    $fatal(1, "out_addr mismatch");
            end

            if (cycle_done != 0 && local_c == cycle_done + 1) begin
                if (pending_mac_check) begin
                    mac_i = checked_macs;
                    if (last_product !== exp_prod[mac_i] ||
                        current_accumulator !== exp_acc[mac_i])
                        fail_mac(mac_i);
                    pending_mac_check = 0;
                    checked_macs = checked_macs + 1;
                end
                if (checked_macs !== 144)
                    $fatal(1, "checked_macs=%0d", checked_macs);
                $display("PASS selected case phase %0d cycles=%0d",
                         phase, cycle_done - cycle_start);
                phase = phase + 1;
                local_c = 0;
                pending_mac_check = 0;
                checked_macs = 0;
                bias_loads = 0;
                saw_issue = 0;
                cycle_start = 0;
                cycle_done = 0;
                if (phase < 7)
                    load_case(phase);
                else begin
                    $display("PASS: tb_conv2_selected_outputs");
                    $finish;
                end
            end
        end
    end
endmodule
