// tb_conv1_channel.sv
// Full Conv1 output channel 0: 1024 bit-exact RAM writes vs Python.

`timescale 1ns / 1ps

module tb_conv1_channel (
    input logic clk
);
    logic rst;
    logic start;
    logic busy;
    logic channel_done;
    logic [10:0] output_count;
    logic [4:0] current_output_row;
    logic [4:0] current_output_column;

    logic out_read_enable;
    logic [9:0] out_read_address;
    logic signed [7:0] out_read_data;

    logic output_write_enable;
    logic [9:0] output_write_address;
    logic signed [7:0] output_write_data;
    logic engine_start;
    logic engine_busy;
    logic engine_done;
    logic [4:0] engine_mac_count;
    logic signed [31:0] engine_final_accumulator;
    logic signed [7:0] engine_requantized_output;
    logic signed [7:0] engine_relu_output;
    logic [3:0] engine_output_channel;
    logic engine_operand_valid;
    logic engine_mac_enable;
    logic engine_load_bias;
    logic [1:0] engine_ic;
    logic [1:0] engine_kr;
    logic [1:0] engine_kc;
    logic [11:0] engine_act_addr;
    logic [8:0] engine_wgt_addr;
    logic engine_padding;
    logic signed [7:0] engine_act_value;
    logic signed [7:0] engine_wgt_value;
    logic signed [15:0] engine_product;
    logic signed [31:0] engine_accumulator;
    logic signed [15:0] engine_last_product;

    logic signed [7:0] expected [0:1023];
    logic              written  [0:1023];

    logic signed [31:0] sel0_act_addr [0:26];
    logic signed [31:0] sel0_wgt_addr [0:26];
    logic signed [7:0]  sel0_pad      [0:26];
    logic signed [15:0] sel0_prod     [0:26];
    logic signed [31:0] sel0_acc      [0:26];
    logic signed [31:0] sel0_final    [0:0];
    logic signed [7:0]  sel0_rq       [0:0];
    logic signed [7:0]  sel0_relu     [0:0];

    logic signed [31:0] sel1_act_addr [0:26];
    logic signed [31:0] sel1_wgt_addr [0:26];
    logic signed [7:0]  sel1_pad      [0:26];
    logic signed [15:0] sel1_prod     [0:26];
    logic signed [31:0] sel1_acc      [0:26];
    logic signed [31:0] sel1_final    [0:0];
    logic signed [7:0]  sel1_rq       [0:0];
    logic signed [7:0]  sel1_relu     [0:0];

    logic signed [31:0] sel2_act_addr [0:26];
    logic signed [31:0] sel2_wgt_addr [0:26];
    logic signed [7:0]  sel2_pad      [0:26];
    logic signed [15:0] sel2_prod     [0:26];
    logic signed [31:0] sel2_acc      [0:26];
    logic signed [31:0] sel2_final    [0:0];
    logic signed [7:0]  sel2_rq       [0:0];
    logic signed [7:0]  sel2_relu     [0:0];

    integer local_c;
    integer phase;
    integer i;
    integer mismatches;
    integer starts_seen;
    integer dones_seen;
    integer writes_seen;
    integer first_write_cycle;
    integer last_write_cycle;
    integer channel_done_cycle;
    integer start_cycle;
    integer checksum;
    integer min_v;
    integer max_v;
    integer read_idx;
    integer pending_mac_check;
    integer checked_macs;
    integer active_sel;
    integer mac_i;
    integer exp_row;
    integer exp_col;
    integer prev_write_addr;
    integer sample_addr;

    conv1_channel_top dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .busy(busy),
        .channel_done(channel_done),
        .output_count(output_count),
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
        .engine_operand_valid(engine_operand_valid),
        .engine_mac_enable(engine_mac_enable),
        .engine_load_bias(engine_load_bias),
        .engine_ic(engine_ic),
        .engine_kr(engine_kr),
        .engine_kc(engine_kc),
        .engine_act_addr(engine_act_addr),
        .engine_wgt_addr(engine_wgt_addr),
        .engine_padding(engine_padding),
        .engine_act_value(engine_act_value),
        .engine_wgt_value(engine_wgt_value),
        .engine_product(engine_product),
        .engine_accumulator(engine_accumulator),
        .engine_last_product(engine_last_product)
    );

    initial begin
        local_c = 0;
        phase = 0;
        rst = 1;
        start = 0;
        out_read_enable = 0;
        out_read_address = 0;
        mismatches = 0;
        starts_seen = 0;
        dones_seen = 0;
        writes_seen = 0;
        first_write_cycle = 0;
        last_write_cycle = 0;
        channel_done_cycle = 0;
        start_cycle = 0;
        checksum = 0;
        min_v = 127;
        max_v = -128;
        read_idx = 0;
        pending_mac_check = 0;
        checked_macs = 0;
        active_sel = -1;
        prev_write_addr = -1;
        sample_addr = -1;

        for (i = 0; i < 1024; i = i + 1)
            written[i] = 1'b0;

        $readmemh("vectors/conv1_channel/conv1_channel0_expected.mem", expected);

        $readmemh("vectors/conv1_channel/selected_pixel_traces/r0_c0_act_addr.mem", sel0_act_addr);
        $readmemh("vectors/conv1_channel/selected_pixel_traces/r0_c0_wgt_addr.mem", sel0_wgt_addr);
        $readmemh("vectors/conv1_channel/selected_pixel_traces/r0_c0_padding.mem", sel0_pad);
        $readmemh("vectors/conv1_channel/selected_pixel_traces/r0_c0_products.mem", sel0_prod);
        $readmemh("vectors/conv1_channel/selected_pixel_traces/r0_c0_acc_trace.mem", sel0_acc);
        $readmemh("vectors/conv1_channel/selected_pixel_traces/r0_c0_expected_acc.mem", sel0_final);
        $readmemh("vectors/conv1_channel/selected_pixel_traces/r0_c0_expected_requantized.mem", sel0_rq);
        $readmemh("vectors/conv1_channel/selected_pixel_traces/r0_c0_expected_relu.mem", sel0_relu);

        $readmemh("vectors/conv1_channel/selected_pixel_traces/r10_c10_act_addr.mem", sel1_act_addr);
        $readmemh("vectors/conv1_channel/selected_pixel_traces/r10_c10_wgt_addr.mem", sel1_wgt_addr);
        $readmemh("vectors/conv1_channel/selected_pixel_traces/r10_c10_padding.mem", sel1_pad);
        $readmemh("vectors/conv1_channel/selected_pixel_traces/r10_c10_products.mem", sel1_prod);
        $readmemh("vectors/conv1_channel/selected_pixel_traces/r10_c10_acc_trace.mem", sel1_acc);
        $readmemh("vectors/conv1_channel/selected_pixel_traces/r10_c10_expected_acc.mem", sel1_final);
        $readmemh("vectors/conv1_channel/selected_pixel_traces/r10_c10_expected_requantized.mem", sel1_rq);
        $readmemh("vectors/conv1_channel/selected_pixel_traces/r10_c10_expected_relu.mem", sel1_relu);

        $readmemh("vectors/conv1_channel/selected_pixel_traces/r31_c31_act_addr.mem", sel2_act_addr);
        $readmemh("vectors/conv1_channel/selected_pixel_traces/r31_c31_wgt_addr.mem", sel2_wgt_addr);
        $readmemh("vectors/conv1_channel/selected_pixel_traces/r31_c31_padding.mem", sel2_pad);
        $readmemh("vectors/conv1_channel/selected_pixel_traces/r31_c31_products.mem", sel2_prod);
        $readmemh("vectors/conv1_channel/selected_pixel_traces/r31_c31_acc_trace.mem", sel2_acc);
        $readmemh("vectors/conv1_channel/selected_pixel_traces/r31_c31_expected_acc.mem", sel2_final);
        $readmemh("vectors/conv1_channel/selected_pixel_traces/r31_c31_expected_requantized.mem", sel2_rq);
        $readmemh("vectors/conv1_channel/selected_pixel_traces/r31_c31_expected_relu.mem", sel2_relu);
    end

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
            if (busy && local_c > 5)
                start = 0;
        end else if (phase == 1) begin
            if (read_idx < 1024) begin
                out_read_enable = 1;
                out_read_address = read_idx[9:0];
            end else begin
                out_read_enable = 0;
            end
        end
    end

    always @(posedge clk) begin
        if (phase == 0) begin
            if (engine_output_channel !== 4'd0) begin
                $display("FAIL engine_output_channel=%0d", engine_output_channel);
                $fatal(1);
            end

            if (engine_start) begin
                starts_seen = starts_seen + 1;
                if (engine_busy) begin
                    $display("FAIL start while engine busy @c%0d", local_c);
                    $fatal(1);
                end
                if (starts_seen == 1) begin
                    if (current_output_row !== 5'd0 || current_output_column !== 5'd0) begin
                        $display("FAIL first coord got (%0d,%0d)",
                                 current_output_row, current_output_column);
                        $fatal(1);
                    end
                end
                if (starts_seen == 2) begin
                    if (current_output_row !== 5'd0 || current_output_column !== 5'd1) begin
                        $display("FAIL second coord got (%0d,%0d)",
                                 current_output_row, current_output_column);
                        $fatal(1);
                    end
                end
                active_sel = -1;
                checked_macs = 0;
                pending_mac_check = 0;
                if (current_output_row == 5'd0 && current_output_column == 5'd0)
                    active_sel = 0;
                else if (current_output_row == 5'd10 && current_output_column == 5'd10)
                    active_sel = 1;
                else if (current_output_row == 5'd31 && current_output_column == 5'd31)
                    active_sel = 2;
            end

            if (engine_done) begin
                dones_seen = dones_seen + 1;
                if (engine_mac_count !== 5'd27) begin
                    $display("FAIL mac_count=%0d at done", engine_mac_count);
                    $fatal(1);
                end
                if (active_sel == 0) begin
                    if (engine_final_accumulator !== sel0_final[0]) $fatal(1, "sel0 acc");
                    if (engine_requantized_output !== sel0_rq[0]) $fatal(1, "sel0 rq");
                    if (engine_relu_output !== sel0_relu[0]) $fatal(1, "sel0 relu");
                end else if (active_sel == 1) begin
                    if (engine_final_accumulator !== sel1_final[0]) $fatal(1, "sel1 acc");
                    if (engine_requantized_output !== sel1_rq[0]) $fatal(1, "sel1 rq");
                    if (engine_relu_output !== sel1_relu[0]) $fatal(1, "sel1 relu");
                end else if (active_sel == 2) begin
                    if (engine_final_accumulator !== sel2_final[0]) $fatal(1, "sel2 acc");
                    if (engine_requantized_output !== sel2_rq[0]) $fatal(1, "sel2 rq");
                    if (engine_relu_output !== sel2_relu[0]) $fatal(1, "sel2 relu");
                end
            end

            if (active_sel >= 0) begin
                if (pending_mac_check) begin
                    mac_i = checked_macs;
                    if (active_sel == 0) begin
                        if (engine_last_product !== sel0_prod[mac_i] ||
                            engine_accumulator !== sel0_acc[mac_i])
                            $fatal(1, "sel0 mac registered");
                    end else if (active_sel == 1) begin
                        if (engine_last_product !== sel1_prod[mac_i] ||
                            engine_accumulator !== sel1_acc[mac_i])
                            $fatal(1, "sel1 mac registered");
                    end else begin
                        if (engine_last_product !== sel2_prod[mac_i] ||
                            engine_accumulator !== sel2_acc[mac_i])
                            $fatal(1, "sel2 mac registered");
                    end
                    pending_mac_check = 0;
                    checked_macs = checked_macs + 1;
                end
                if (engine_operand_valid && engine_mac_enable && !engine_load_bias) begin
                    mac_i = checked_macs;
                    if (active_sel == 0) begin
                        if (engine_padding !== sel0_pad[mac_i][0]) $fatal(1, "sel0 pad");
                        if (!sel0_pad[mac_i][0] &&
                            engine_act_addr !== sel0_act_addr[mac_i][11:0])
                            $fatal(1, "sel0 act_addr");
                        if (engine_wgt_addr !== sel0_wgt_addr[mac_i][8:0])
                            $fatal(1, "sel0 wgt_addr");
                        if (engine_product !== sel0_prod[mac_i]) $fatal(1, "sel0 prod");
                    end else if (active_sel == 1) begin
                        if (engine_padding !== sel1_pad[mac_i][0]) $fatal(1, "sel1 pad");
                        if (!sel1_pad[mac_i][0] &&
                            engine_act_addr !== sel1_act_addr[mac_i][11:0])
                            $fatal(1, "sel1 act_addr");
                        if (engine_wgt_addr !== sel1_wgt_addr[mac_i][8:0])
                            $fatal(1, "sel1 wgt_addr");
                        if (engine_product !== sel1_prod[mac_i]) $fatal(1, "sel1 prod");
                    end else begin
                        if (engine_padding !== sel2_pad[mac_i][0]) $fatal(1, "sel2 pad");
                        if (!sel2_pad[mac_i][0] &&
                            engine_act_addr !== sel2_act_addr[mac_i][11:0])
                            $fatal(1, "sel2 act_addr");
                        if (engine_wgt_addr !== sel2_wgt_addr[mac_i][8:0])
                            $fatal(1, "sel2 wgt_addr");
                        if (engine_product !== sel2_prod[mac_i]) $fatal(1, "sel2 prod");
                    end
                    pending_mac_check = 1;
                end
            end

            if (output_write_enable) begin
                writes_seen = writes_seen + 1;
                if (writes_seen == 1)
                    first_write_cycle = local_c;
                last_write_cycle = local_c;

                if ({1'b0, output_write_address} >= 11'd1024)
                    $fatal(1);
                if (written[output_write_address])
                    $fatal(1, "duplicate write");
                written[output_write_address] = 1'b1;

                exp_row = output_write_address / 32;
                exp_col = output_write_address % 32;
                if (current_output_row !== exp_row[4:0] ||
                    current_output_column !== exp_col[4:0])
                    $fatal(1, "write coord mismatch");
                if (writes_seen > 1 && output_write_address !== prev_write_addr + 1)
                    $fatal(1, "non-sequential write");
                prev_write_addr = output_write_address;

                if (output_write_data !== expected[output_write_address]) begin
                    $display("FAIL write addr=%0d row=%0d col=%0d exp=%0d got=%0d exp_hex=%02x got_hex=%02x",
                             output_write_address, exp_row, exp_col,
                             $signed(expected[output_write_address]),
                             $signed(output_write_data),
                             expected[output_write_address],
                             output_write_data);
                    $fatal(1);
                end
            end

            if (channel_done) begin
                channel_done_cycle = local_c;
                if (output_count !== 11'd1024)
                    $fatal(1, "bad output_count");
                if (writes_seen !== 1024 || starts_seen !== 1024 || dones_seen !== 1024) begin
                    $display("FAIL starts=%0d dones=%0d writes=%0d",
                             starts_seen, dones_seen, writes_seen);
                    $fatal(1);
                end
                if (prev_write_addr !== 1023)
                    $fatal(1, "last addr");
                for (i = 0; i < 1024; i = i + 1) begin
                    if (!written[i])
                        $fatal(1, "missing write");
                end
                $display("PASS channel compute: writes=%0d cycles=%0d cycles_per_output=%0d",
                         writes_seen,
                         channel_done_cycle - start_cycle,
                         (channel_done_cycle - start_cycle) / 1024);
                phase = 1;
                read_idx = 0;
                sample_addr = -1;
                local_c = 0;
            end
        end else if (phase == 1) begin
            // After addr issued on prior negedge, data valid this posedge (1-cycle latency)
            if (sample_addr >= 0 && sample_addr < 1024) begin
                if (out_read_data !== expected[sample_addr]) begin
                    $display("FAIL RAM readback addr=%0d row=%0d col=%0d exp=%0d got=%0d exp_hex=%02x got_hex=%02x",
                             sample_addr, sample_addr / 32, sample_addr % 32,
                             $signed(expected[sample_addr]), $signed(out_read_data),
                             expected[sample_addr], out_read_data);
                    $fatal(1);
                end
                checksum = checksum + $signed(out_read_data);
                if ($signed(out_read_data) < min_v)
                    min_v = $signed(out_read_data);
                if ($signed(out_read_data) > max_v)
                    max_v = $signed(out_read_data);
            end

            // Address presented at negedge with local_c==N becomes valid on posedge N+1
            // At posedge local_c: sample previous address (local_c-1) if it was issued
            if (local_c >= 1 && (local_c - 1) < 1024)
                sample_addr = local_c - 1;
            else
                sample_addr = -1;

            if (read_idx < 1024)
                read_idx = read_idx + 1;

            if (local_c > 1025) begin
                $display("PASS tb_conv1_channel");
                $display("  total_outputs=1024 mismatches=0");
                $display("  total_cycles=%0d cycles_per_output=%0d",
                         channel_done_cycle - start_cycle,
                         (channel_done_cycle - start_cycle) / 1024);
                $display("  first_write_cycle=%0d last_write_cycle=%0d",
                         first_write_cycle, last_write_cycle);
                $display("  checksum=%0d min=%0d max=%0d", checksum, min_v, max_v);
                $finish;
            end
        end
    end
endmodule
