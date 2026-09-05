// tb_cnn_bd_wrapper_compare.sv
// Cycle-accurate transparency: BD wrapper vs synth wrapper must match
// bit-for-bit on control/status/results (zeroed Activation RAM A).

`timescale 1ns / 1ps

module tb_cnn_bd_wrapper_compare (
    input logic clk
);

    logic rst;
    logic start;
    logic input_write_enable;
    logic [11:0] input_write_address;
    logic signed [7:0] input_write_data;

    logic busy_s, done_s, busy_b, done_b;
    logic [2:0] class_s, class_b;
    logic signed [31:0] max_s, max_b;
    logic signed [31:0] l0_s, l1_s, l2_s, l3_s, l4_s;
    logic signed [31:0] l0_b, l1_b, l2_b, l3_b, l4_b;
    logic [63:0] cyc_s, cyc_b;

    int unsigned cycle;
    int unsigned errors;
    int unsigned started;
    int unsigned finished;

    cnn_accelerator_synth_wrapper #(
        .LOGIT_WIDTH(32)
    ) u_synth (
        .clk(clk),
        .rst(rst),
        .start(start),
        .busy(busy_s),
        .done(done_s),
        .predicted_class(class_s),
        .maximum_logit(max_s),
        .logit_0(l0_s),
        .logit_1(l1_s),
        .logit_2(l2_s),
        .logit_3(l3_s),
        .logit_4(l4_s),
        .cycle_count(cyc_s),
        .input_write_enable(input_write_enable),
        .input_write_address(input_write_address),
        .input_write_data(input_write_data)
    );

    cnn_accelerator_bd_wrapper u_bd (
        .clk(clk),
        .rst(rst),
        .start(start),
        .busy(busy_b),
        .done(done_b),
        .predicted_class(class_b),
        .maximum_logit(max_b),
        .logit_0(l0_b),
        .logit_1(l1_b),
        .logit_2(l2_b),
        .logit_3(l3_b),
        .logit_4(l4_b),
        .cycle_count(cyc_b),
        .input_write_enable(input_write_enable),
        .input_write_address(input_write_address),
        .input_write_data(input_write_data)
    );

    always_ff @(posedge clk) begin
        if (cycle == 0) begin
            errors <= 0;
            started <= 0;
            finished <= 0;
            rst <= 1'b1;
            start <= 1'b0;
            input_write_enable <= 1'b0;
            input_write_address <= 12'd0;
            input_write_data <= 8'sd0;
        end else if (cycle == 5) begin
            rst <= 1'b0;
        end else if (cycle == 10) begin
            start <= 1'b1;
            started <= 1;
        end else if (cycle == 11) begin
            start <= 1'b0;
        end else begin
            if (started && cycle > 11 && !finished) begin
                if (busy_b !== busy_s) begin
                    $display("FAIL busy @%0d: bd=%0d synth=%0d", cycle, busy_b, busy_s);
                    errors <= errors + 1;
                end
                if (done_b !== done_s) begin
                    $display("FAIL done @%0d: bd=%0d synth=%0d", cycle, done_b, done_s);
                    errors <= errors + 1;
                end
                if (class_b !== class_s) begin
                    $display("FAIL predicted_class @%0d: bd=%0d synth=%0d", cycle, class_b, class_s);
                    errors <= errors + 1;
                end
                if (max_b !== max_s) begin
                    $display("FAIL maximum_logit @%0d", cycle);
                    errors <= errors + 1;
                end
                if (l0_b !== l0_s || l1_b !== l1_s || l2_b !== l2_s ||
                    l3_b !== l3_s || l4_b !== l4_s) begin
                    $display("FAIL logits @%0d", cycle);
                    errors <= errors + 1;
                end
                if (cyc_b !== cyc_s) begin
                    $display("FAIL cycle_count @%0d: bd=%0d synth=%0d", cycle, cyc_b, cyc_s);
                    errors <= errors + 1;
                end
            end

            if (done_s && done_b && !finished) begin
                finished <= 1;
                $display("Both wrappers reached done at cycle %0d class=%0d cyc=%0d",
                         cycle, class_s, cyc_s);
                if (errors == 0)
                    $display("PASS: tb_cnn_bd_wrapper_compare (bit-exact vs synth_wrapper)");
                else
                    $display("FAIL: tb_cnn_bd_wrapper_compare errors=%0d", errors);
                $finish;
            end

            if (cycle > 20000000) begin
                $display("FAIL timeout waiting for done");
                $finish;
            end
        end

        cycle <= cycle + 1;
    end

    initial cycle = 0;

endmodule
