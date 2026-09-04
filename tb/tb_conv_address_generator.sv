// tb_conv_address_generator.sv
// Bit-exact address / padding checks vs Python-exported vectors.

`timescale 1ns / 1ps

module tb_conv_address_generator (
    input logic clk
);
    logic [3:0] output_channel;
    logic [4:0] output_row;
    logic [4:0] output_column;
    logic [1:0] input_channel;
    logic [1:0] kernel_row;
    logic [1:0] kernel_column;

    logic signed [6:0] input_row;
    logic signed [6:0] input_column;
    logic              padding;
    logic [11:0]       activation_address;
    logic [8:0]        weight_address;
    logic [3:0]        bias_address;

    logic signed [31:0] exp_act_addr [0:26];
    logic signed [31:0] exp_wgt_addr [0:26];
    logic signed [7:0]  exp_in_row   [0:26];
    logic signed [7:0]  exp_in_col   [0:26];
    logic signed [7:0]  exp_pad      [0:26];

    integer local_c;
    integer phase;
    integer mac_i;
    integer errors;
    integer case_oc;

    conv_address_generator dut (
        .output_channel     (output_channel),
        .output_row         (output_row),
        .output_column      (output_column),
        .input_channel      (input_channel),
        .kernel_row         (kernel_row),
        .kernel_column      (kernel_column),
        .input_row          (input_row),
        .input_column       (input_column),
        .padding            (padding),
        .activation_address (activation_address),
        .weight_address     (weight_address),
        .bias_address       (bias_address)
    );

    task automatic load_center;
        begin
            case_oc = 0;
            output_channel = 4'd0; output_row = 5'd10; output_column = 5'd10;
            $readmemh("vectors/conv1_memory/selected_output_traces/center_act_addr.mem", exp_act_addr);
            $readmemh("vectors/conv1_memory/selected_output_traces/center_wgt_addr.mem", exp_wgt_addr);
            $readmemh("vectors/conv1_memory/selected_output_traces/center_input_row.mem", exp_in_row);
            $readmemh("vectors/conv1_memory/selected_output_traces/center_input_col.mem", exp_in_col);
            $readmemh("vectors/conv1_memory/selected_output_traces/center_padding.mem", exp_pad);
        end
    endtask

    task automatic load_top_left;
        begin
            case_oc = 0;
            output_channel = 4'd0; output_row = 5'd0; output_column = 5'd0;
            $readmemh("vectors/conv1_memory/selected_output_traces/top_left_act_addr.mem", exp_act_addr);
            $readmemh("vectors/conv1_memory/selected_output_traces/top_left_wgt_addr.mem", exp_wgt_addr);
            $readmemh("vectors/conv1_memory/selected_output_traces/top_left_input_row.mem", exp_in_row);
            $readmemh("vectors/conv1_memory/selected_output_traces/top_left_input_col.mem", exp_in_col);
            $readmemh("vectors/conv1_memory/selected_output_traces/top_left_padding.mem", exp_pad);
        end
    endtask

    task automatic load_top_right;
        begin
            case_oc = 0;
            output_channel = 4'd0; output_row = 5'd0; output_column = 5'd31;
            $readmemh("vectors/conv1_memory/selected_output_traces/top_right_act_addr.mem", exp_act_addr);
            $readmemh("vectors/conv1_memory/selected_output_traces/top_right_wgt_addr.mem", exp_wgt_addr);
            $readmemh("vectors/conv1_memory/selected_output_traces/top_right_input_row.mem", exp_in_row);
            $readmemh("vectors/conv1_memory/selected_output_traces/top_right_input_col.mem", exp_in_col);
            $readmemh("vectors/conv1_memory/selected_output_traces/top_right_padding.mem", exp_pad);
        end
    endtask

    task automatic load_bottom_left;
        begin
            case_oc = 0;
            output_channel = 4'd0; output_row = 5'd31; output_column = 5'd0;
            $readmemh("vectors/conv1_memory/selected_output_traces/bottom_left_act_addr.mem", exp_act_addr);
            $readmemh("vectors/conv1_memory/selected_output_traces/bottom_left_wgt_addr.mem", exp_wgt_addr);
            $readmemh("vectors/conv1_memory/selected_output_traces/bottom_left_input_row.mem", exp_in_row);
            $readmemh("vectors/conv1_memory/selected_output_traces/bottom_left_input_col.mem", exp_in_col);
            $readmemh("vectors/conv1_memory/selected_output_traces/bottom_left_padding.mem", exp_pad);
        end
    endtask

    task automatic load_bottom_right;
        begin
            case_oc = 0;
            output_channel = 4'd0; output_row = 5'd31; output_column = 5'd31;
            $readmemh("vectors/conv1_memory/selected_output_traces/bottom_right_act_addr.mem", exp_act_addr);
            $readmemh("vectors/conv1_memory/selected_output_traces/bottom_right_wgt_addr.mem", exp_wgt_addr);
            $readmemh("vectors/conv1_memory/selected_output_traces/bottom_right_input_row.mem", exp_in_row);
            $readmemh("vectors/conv1_memory/selected_output_traces/bottom_right_input_col.mem", exp_in_col);
            $readmemh("vectors/conv1_memory/selected_output_traces/bottom_right_padding.mem", exp_pad);
        end
    endtask

    task automatic load_channel5;
        begin
            case_oc = 5;
            output_channel = 4'd5; output_row = 5'd12; output_column = 5'd17;
            $readmemh("vectors/conv1_memory/selected_output_traces/channel5_act_addr.mem", exp_act_addr);
            $readmemh("vectors/conv1_memory/selected_output_traces/channel5_wgt_addr.mem", exp_wgt_addr);
            $readmemh("vectors/conv1_memory/selected_output_traces/channel5_input_row.mem", exp_in_row);
            $readmemh("vectors/conv1_memory/selected_output_traces/channel5_input_col.mem", exp_in_col);
            $readmemh("vectors/conv1_memory/selected_output_traces/channel5_padding.mem", exp_pad);
        end
    endtask

    initial begin
        local_c = 0;
        phase = 0;
        errors = 0;
        input_channel = 0;
        kernel_row = 0;
        kernel_column = 0;
        load_center();
    end

    always @(negedge clk) begin
        local_c = local_c + 1;
        if (phase < 6) begin
            if (local_c >= 2 && local_c <= 28) begin
                mac_i = local_c - 2;
                input_channel  = mac_i / 9;
                kernel_row     = (mac_i % 9) / 3;
                kernel_column  = mac_i % 3;
            end
        end
    end

    always @(posedge clk) begin
        if (phase < 6 && local_c >= 2 && local_c <= 28) begin
            mac_i = local_c - 2;
            if ($signed(input_row) !== $signed({{1{exp_in_row[mac_i][7]}}, exp_in_row[mac_i]})) begin
                $display("FAIL phase%0d MAC%0d input_row got %0d exp %0d",
                         phase, mac_i, $signed(input_row), $signed(exp_in_row[mac_i]));
                errors = errors + 1;
            end
            if ($signed(input_column) !== $signed({{1{exp_in_col[mac_i][7]}}, exp_in_col[mac_i]})) begin
                $display("FAIL phase%0d MAC%0d input_col got %0d exp %0d",
                         phase, mac_i, $signed(input_column), $signed(exp_in_col[mac_i]));
                errors = errors + 1;
            end
            if (padding !== exp_pad[mac_i][0]) begin
                $display("FAIL phase%0d MAC%0d padding got %0d exp %0d",
                         phase, mac_i, padding, exp_pad[mac_i]);
                errors = errors + 1;
            end
            if (exp_pad[mac_i][0] == 1'b0) begin
                if (activation_address !== exp_act_addr[mac_i][11:0]) begin
                    $display("FAIL phase%0d MAC%0d act_addr got %0d exp %0d",
                             phase, mac_i, activation_address, exp_act_addr[mac_i]);
                    errors = errors + 1;
                end
            end
            if (weight_address !== exp_wgt_addr[mac_i][8:0]) begin
                $display("FAIL phase%0d MAC%0d wgt_addr got %0d exp %0d",
                         phase, mac_i, weight_address, exp_wgt_addr[mac_i]);
                errors = errors + 1;
            end
            if (bias_address !== case_oc[3:0]) begin
                $display("FAIL phase%0d MAC%0d bias_addr got %0d exp %0d",
                         phase, mac_i, bias_address, case_oc);
                errors = errors + 1;
            end
            if (local_c == 28) begin
                if (errors != 0) begin
                    $display("FAIL tb_conv_address_generator phase %0d", phase);
                    $fatal(1);
                end
                $display("PASS address phase %0d", phase);
                phase = phase + 1;
                local_c = 0;
                errors = 0;
                if (phase == 1) load_top_left();
                else if (phase == 2) load_top_right();
                else if (phase == 3) load_bottom_left();
                else if (phase == 4) load_bottom_right();
                else if (phase == 5) load_channel5();
                else begin
                    $display("PASS tb_conv_address_generator");
                    $finish;
                end
            end
        end
    end
endmodule
