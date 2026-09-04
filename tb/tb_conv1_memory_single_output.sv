// tb_conv1_memory_single_output.sv
// Full memory-driven Conv1 single-output bit-exact tests + latency checks.

`timescale 1ns / 1ps

module tb_conv1_memory_single_output (
    input logic clk
);
    logic rst;
    logic start;
    logic [3:0] output_channel;
    logic [4:0] output_row;
    logic [4:0] output_column;

    logic busy;
    logic done;
    logic signed [31:0] final_accumulator;
    logic signed [7:0]  requantized_output;
    logic signed [7:0]  relu_output;
    logic [4:0] mac_count;

    logic [1:0] current_input_channel;
    logic [1:0] current_kernel_row;
    logic [1:0] current_kernel_column;
    logic [11:0] current_activation_address;
    logic [8:0]  current_weight_address;
    logic [3:0]  current_bias_address;
    logic current_padding;
    logic signed [7:0]  current_activation_value;
    logic signed [7:0]  current_weight_value;
    logic signed [15:0] current_product;
    logic signed [31:0] current_accumulator;
    logic operand_valid;
    logic mac_enable;
    logic load_bias;
    logic act_read_enable;
    logic wgt_read_enable;
    logic signed [15:0] last_product;

    logic signed [31:0] exp_act_addr [0:26];
    logic signed [31:0] exp_wgt_addr [0:26];
    logic signed [7:0]  exp_pad      [0:26];
    logic signed [7:0]  exp_act      [0:26];
    logic signed [7:0]  exp_wgt      [0:26];
    logic signed [15:0] exp_prod     [0:26];
    logic signed [31:0] exp_acc      [0:26];
    logic signed [31:0] exp_final_mem [0:0];
    logic signed [7:0]  exp_rq_mem    [0:0];
    logic signed [7:0]  exp_relu_mem  [0:0];

    integer local_c;
    integer phase;
    integer mac_i;
    integer cycle_start;
    integer cycle_done;
    integer pending_mac_check;
    integer checked_macs;
    integer bias_loads;
    integer saw_issue;

    conv1_memory_single_output dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .output_channel(output_channel),
        .output_row(output_row),
        .output_column(output_column),
        .external_act_data(8'sd0),
        .busy(busy),
        .done(done),
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

    task automatic setup_center;
        begin
            output_channel = 4'd0; output_row = 5'd10; output_column = 5'd10;
            $readmemh("vectors/conv1_memory/selected_output_traces/center_act_addr.mem", exp_act_addr);
            $readmemh("vectors/conv1_memory/selected_output_traces/center_wgt_addr.mem", exp_wgt_addr);
            $readmemh("vectors/conv1_memory/selected_output_traces/center_padding.mem", exp_pad);
            $readmemh("vectors/conv1_memory/selected_output_traces/center_inputs.mem", exp_act);
            $readmemh("vectors/conv1_memory/selected_output_traces/center_weights.mem", exp_wgt);
            $readmemh("vectors/conv1_memory/selected_output_traces/center_products.mem", exp_prod);
            $readmemh("vectors/conv1_memory/selected_output_traces/center_acc_trace.mem", exp_acc);
            $readmemh("vectors/conv1_memory/selected_output_traces/center_expected_acc.mem", exp_final_mem);
            $readmemh("vectors/conv1_memory/selected_output_traces/center_expected_requantized.mem", exp_rq_mem);
            $readmemh("vectors/conv1_memory/selected_output_traces/center_expected_relu.mem", exp_relu_mem);
        end
    endtask

    task automatic setup_top_left;
        begin
            output_channel = 4'd0; output_row = 5'd0; output_column = 5'd0;
            $readmemh("vectors/conv1_memory/selected_output_traces/top_left_act_addr.mem", exp_act_addr);
            $readmemh("vectors/conv1_memory/selected_output_traces/top_left_wgt_addr.mem", exp_wgt_addr);
            $readmemh("vectors/conv1_memory/selected_output_traces/top_left_padding.mem", exp_pad);
            $readmemh("vectors/conv1_memory/selected_output_traces/top_left_inputs.mem", exp_act);
            $readmemh("vectors/conv1_memory/selected_output_traces/top_left_weights.mem", exp_wgt);
            $readmemh("vectors/conv1_memory/selected_output_traces/top_left_products.mem", exp_prod);
            $readmemh("vectors/conv1_memory/selected_output_traces/top_left_acc_trace.mem", exp_acc);
            $readmemh("vectors/conv1_memory/selected_output_traces/top_left_expected_acc.mem", exp_final_mem);
            $readmemh("vectors/conv1_memory/selected_output_traces/top_left_expected_requantized.mem", exp_rq_mem);
            $readmemh("vectors/conv1_memory/selected_output_traces/top_left_expected_relu.mem", exp_relu_mem);
        end
    endtask

    task automatic setup_bottom_right;
        begin
            output_channel = 4'd0; output_row = 5'd31; output_column = 5'd31;
            $readmemh("vectors/conv1_memory/selected_output_traces/bottom_right_act_addr.mem", exp_act_addr);
            $readmemh("vectors/conv1_memory/selected_output_traces/bottom_right_wgt_addr.mem", exp_wgt_addr);
            $readmemh("vectors/conv1_memory/selected_output_traces/bottom_right_padding.mem", exp_pad);
            $readmemh("vectors/conv1_memory/selected_output_traces/bottom_right_inputs.mem", exp_act);
            $readmemh("vectors/conv1_memory/selected_output_traces/bottom_right_weights.mem", exp_wgt);
            $readmemh("vectors/conv1_memory/selected_output_traces/bottom_right_products.mem", exp_prod);
            $readmemh("vectors/conv1_memory/selected_output_traces/bottom_right_acc_trace.mem", exp_acc);
            $readmemh("vectors/conv1_memory/selected_output_traces/bottom_right_expected_acc.mem", exp_final_mem);
            $readmemh("vectors/conv1_memory/selected_output_traces/bottom_right_expected_requantized.mem", exp_rq_mem);
            $readmemh("vectors/conv1_memory/selected_output_traces/bottom_right_expected_relu.mem", exp_relu_mem);
        end
    endtask

    task automatic setup_channel5;
        begin
            output_channel = 4'd5; output_row = 5'd12; output_column = 5'd17;
            $readmemh("vectors/conv1_memory/selected_output_traces/channel5_act_addr.mem", exp_act_addr);
            $readmemh("vectors/conv1_memory/selected_output_traces/channel5_wgt_addr.mem", exp_wgt_addr);
            $readmemh("vectors/conv1_memory/selected_output_traces/channel5_padding.mem", exp_pad);
            $readmemh("vectors/conv1_memory/selected_output_traces/channel5_inputs.mem", exp_act);
            $readmemh("vectors/conv1_memory/selected_output_traces/channel5_weights.mem", exp_wgt);
            $readmemh("vectors/conv1_memory/selected_output_traces/channel5_products.mem", exp_prod);
            $readmemh("vectors/conv1_memory/selected_output_traces/channel5_acc_trace.mem", exp_acc);
            $readmemh("vectors/conv1_memory/selected_output_traces/channel5_expected_acc.mem", exp_final_mem);
            $readmemh("vectors/conv1_memory/selected_output_traces/channel5_expected_requantized.mem", exp_rq_mem);
            $readmemh("vectors/conv1_memory/selected_output_traces/channel5_expected_relu.mem", exp_relu_mem);
        end
    endtask

    task automatic fail_mac;
        input integer mi;
        begin
            $display("FAIL phase%0d cycle%0d MAC %0d", phase, local_c, mi);
            $display("  output_channel=%0d output_row=%0d output_column=%0d",
                     output_channel, output_row, output_column);
            $display("  input_channel=%0d kernel_row=%0d kernel_column=%0d",
                     current_input_channel, current_kernel_row, current_kernel_column);
            $display("  expected_activation_address=%0d actual=%0d",
                     exp_act_addr[mi], current_activation_address);
            $display("  expected_weight_address=%0d actual=%0d",
                     exp_wgt_addr[mi], current_weight_address);
            $display("  expected_padding=%0d actual=%0d",
                     exp_pad[mi], current_padding);
            $display("  expected_activation=%0d actual=%0d",
                     $signed(exp_act[mi]), $signed(current_activation_value));
            $display("  expected_weight=%0d actual=%0d",
                     $signed(exp_wgt[mi]), $signed(current_weight_value));
            $display("  expected_product=%0d actual=%0d",
                     $signed(exp_prod[mi]), $signed(current_product));
            $display("  expected_accumulator=%0d actual=%0d",
                     $signed(exp_acc[mi]), $signed(current_accumulator));
            $fatal(1);
        end
    endtask

    initial begin
        local_c = 0;
        phase = 0;
        rst = 1;
        start = 0;
        pending_mac_check = 0;
        checked_macs = 0;
        bias_loads = 0;
        saw_issue = 0;
        cycle_start = 0;
        cycle_done = 0;
        setup_center();
    end

    always @(negedge clk) begin
        local_c = local_c + 1;
        if (phase < 4) begin
            case (local_c)
                1, 2: begin rst = 1; start = 0; end
                3: begin rst = 0; start = 0; end
                4: begin start = 1; cycle_start = local_c; end
                5: begin start = 0; end
                default: ;
            endcase
            if (busy && local_c > 5)
                start = 0;
        end
    end

    always @(posedge clk) begin
        if (phase < 4) begin
            if (mac_enable && !load_bias) begin
                if (!operand_valid) begin
                    $display("FAIL mac_enable without operand_valid @c%0d", local_c);
                    $fatal(1);
                end
            end
            if (act_read_enable && current_padding) begin
                $display("FAIL act read while padding @c%0d", local_c);
                $fatal(1);
            end
            if (mac_count > 5'd27) begin
                $display("FAIL mac_count>%0d", mac_count);
                $fatal(1);
            end
            if (mac_enable && load_bias)
                bias_loads = bias_loads + 1;

            if (pending_mac_check) begin
                mac_i = checked_macs;
                if (last_product !== exp_prod[mac_i]) begin
                    $display("product mismatch (registered)");
                    fail_mac(mac_i);
                end
                if (current_accumulator !== exp_acc[mac_i]) begin
                    $display("accumulator mismatch");
                    fail_mac(mac_i);
                end
                pending_mac_check = 0;
                checked_macs = checked_macs + 1;
            end

            if (operand_valid && mac_enable && !load_bias) begin
                mac_i = checked_macs;
                if (current_padding !== exp_pad[mac_i][0]) begin
                    $display("padding mismatch");
                    fail_mac(mac_i);
                end
                if (exp_pad[mac_i][0] == 1'b0) begin
                    if (current_activation_address !== exp_act_addr[mac_i][11:0]) begin
                        $display("act_addr mismatch");
                        fail_mac(mac_i);
                    end
                end
                if (current_weight_address !== exp_wgt_addr[mac_i][8:0]) begin
                    $display("wgt_addr mismatch");
                    fail_mac(mac_i);
                end
                if (current_bias_address !== output_channel) begin
                    $display("bias_addr mismatch");
                    fail_mac(mac_i);
                end
                if (current_activation_value !== exp_act[mac_i]) begin
                    $display("activation mismatch");
                    fail_mac(mac_i);
                end
                if (current_weight_value !== exp_wgt[mac_i]) begin
                    $display("weight mismatch");
                    fail_mac(mac_i);
                end
                if (current_product !== exp_prod[mac_i]) begin
                    $display("product (combo) mismatch");
                    fail_mac(mac_i);
                end
                if (!saw_issue) begin
                    $display("FAIL MAC before any operand issue");
                    $fatal(1);
                end
                pending_mac_check = 1;
            end

            if (wgt_read_enable)
                saw_issue = 1;

            if (done) begin
                cycle_done = local_c;
                if (mac_count !== 5'd27) begin
                    $display("FAIL done mac_count=%0d", mac_count);
                    $fatal(1);
                end
                if (bias_loads != 1) begin
                    $display("FAIL bias_loads=%0d (expected 1)", bias_loads);
                    $fatal(1);
                end
                if (final_accumulator !== exp_final_mem[0]) begin
                    $display("FAIL final_acc got %0d exp %0d",
                             final_accumulator, exp_final_mem[0]);
                    $fatal(1);
                end
                if (requantized_output !== exp_rq_mem[0]) begin
                    $display("FAIL requant got %0d exp %0d",
                             requantized_output, exp_rq_mem[0]);
                    $fatal(1);
                end
                if (relu_output !== exp_relu_mem[0]) begin
                    $display("FAIL relu got %0d exp %0d",
                             relu_output, exp_relu_mem[0]);
                    $fatal(1);
                end
            end

            if (cycle_done != 0 && local_c == cycle_done + 1) begin
                if (pending_mac_check) begin
                    mac_i = checked_macs;
                    if (last_product !== exp_prod[mac_i] || current_accumulator !== exp_acc[mac_i])
                        fail_mac(mac_i);
                    pending_mac_check = 0;
                    checked_macs = checked_macs + 1;
                end
                if (checked_macs !== 27) begin
                    $display("FAIL checked_macs=%0d", checked_macs);
                    $fatal(1);
                end
                $display("PASS memory case phase %0d cycles_from_start=%0d",
                         phase, cycle_done - cycle_start);
                phase = phase + 1;
                local_c = 0;
                pending_mac_check = 0;
                checked_macs = 0;
                bias_loads = 0;
                saw_issue = 0;
                cycle_start = 0;
                cycle_done = 0;
                if (phase == 1) setup_top_left();
                else if (phase == 2) setup_bottom_right();
                else if (phase == 3) setup_channel5();
                else begin
                    $display("PASS tb_conv1_memory_single_output");
                    $finish;
                end
            end
        end
    end
endmodule
