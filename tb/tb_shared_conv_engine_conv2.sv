// tb_shared_conv_engine_conv2.sv
// Shared convolution engine in Conv2 mode vs Python golden (8192 outputs).

`timescale 1ns / 1ps

module tb_shared_conv_engine_conv2 (
    input logic clk
);
    localparam int IN_N  = 4096;
    localparam int OUT_N = 8192;

    logic rst, start, busy, done;
    logic layer_is_conv2;
    logic [14:0] output_count;
    logic act_re;
    logic [11:0] act_raddr;
    logic signed [7:0] act_rdata;
    logic out_we;
    logic [13:0] out_waddr;
    logic signed [7:0] out_wdata;

    logic in_we;
    logic [11:0] in_waddr;
    logic signed [7:0] in_wdata;

    logic [4:0] cur_oc, cur_row, cur_col;
    logic dbg_layer;
    logic [7:0] eng_macs;
    logic eng_done_dbg;

    logic signed [7:0] exp_out [0:OUT_N-1];
    logic signed [7:0] exp_in  [0:IN_N-1];
    logic seen [0:OUT_N-1];

    integer local_c, phase, i, mismatches, wr_count;

    int8_sync_ram #(.DEPTH(4096), .ADDR_WIDTH(12)) u_in (
        .clk(clk),
        .write_enable(in_we),
        .write_address(in_waddr),
        .write_data(in_wdata),
        .read_enable(act_re),
        .read_address(act_raddr),
        .read_data(act_rdata)
    );

    shared_conv_engine dut (
        .clk(clk), .rst(rst), .start(start),
        .layer_is_conv2(layer_is_conv2),
        .external_act_data(act_rdata),
        .busy(busy), .done(done),
        .output_count(output_count),
        .current_output_channel(cur_oc),
        .current_output_row(cur_row),
        .current_output_column(cur_col),
        .dbg_layer_is_conv2(dbg_layer),
        .act_read_enable(act_re),
        .act_read_address(act_raddr),
        .output_write_enable(out_we),
        .output_write_address(out_waddr),
        .output_write_data(out_wdata),
        .dbg_engine_mac_count(eng_macs),
        .dbg_engine_done(eng_done_dbg)
    );

    initial begin
        local_c = 0; phase = 0; i = 0; mismatches = 0; wr_count = 0;
        rst = 1; start = 0; layer_is_conv2 = 1;
        in_we = 0; in_waddr = 0; in_wdata = 0;
        for (i = 0; i < OUT_N; i = i + 1) seen[i] = 0;
        i = 0;
        $readmemh("vectors/end_to_end/pool1_expected.mem", exp_in);
        $readmemh("vectors/end_to_end/conv2_expected.mem", exp_out);
    end

    always @(negedge clk) begin
        local_c = local_c + 1;
        in_we = 0; start = 0;
        if (phase == 0) begin
            if (local_c <= 3) begin
                rst = 1;
            end else if (local_c == 4) begin
                rst = 0; i = 0;
            end else if (i < IN_N) begin
                in_we = 1;
                in_waddr = i[11:0];
                in_wdata = exp_in[i];
                i = i + 1;
            end else begin
                phase = 1; local_c = 0;
            end
        end else if (phase == 1) begin
            if (local_c == 1) start = 1;
            else phase = 2;
        end
    end

    always @(posedge clk) begin
        if (phase == 2 && out_we) begin
            if (seen[out_waddr]) $fatal(1, "dup write %0d", out_waddr);
            if (out_waddr >= OUT_N) $fatal(1, "OOB write");
            seen[out_waddr] = 1;
            wr_count = wr_count + 1;
            if (out_wdata !== exp_out[out_waddr]) begin
                $display("FAIL addr=%0d exp=%0d got=%0d",
                         out_waddr, exp_out[out_waddr], out_wdata);
                mismatches = mismatches + 1;
                if (mismatches > 8) $fatal(1, "too many mismatches");
            end
        end
        if (phase == 2 && done) begin
            if (wr_count !== OUT_N) $fatal(1, "write count %0d", wr_count);
            if (output_count !== 15'd8192) $fatal(1, "output_count %0d", output_count);
            if (mismatches !== 0) $fatal(1, "mismatches=%0d", mismatches);
            $display("PASS: tb_shared_conv_engine_conv2 mismatches=0 writes=%0d", wr_count);
            $finish;
        end
        if (local_c > 100000000) $fatal(1, "timeout");
    end
endmodule
