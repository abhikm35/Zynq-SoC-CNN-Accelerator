// tb_cnn_bd_wrapper_elab.sv
// Compile/elaboration smoke test for cnn_accelerator_bd_wrapper.v
// Does not run a full inference.

`timescale 1ns / 1ps

module tb_cnn_bd_wrapper_elab (
    input logic clk
);

    logic rst;
    logic start;
    logic busy, done;
    logic [2:0] predicted_class;
    logic signed [31:0] maximum_logit;
    logic signed [31:0] logit_0, logit_1, logit_2, logit_3, logit_4;
    logic [63:0] cycle_count;
    logic input_write_enable;
    logic [11:0] input_write_address;
    logic signed [7:0] input_write_data;

    int unsigned cycle;
    int unsigned errors;

    cnn_accelerator_bd_wrapper dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .busy(busy),
        .done(done),
        .predicted_class(predicted_class),
        .maximum_logit(maximum_logit),
        .logit_0(logit_0),
        .logit_1(logit_1),
        .logit_2(logit_2),
        .logit_3(logit_3),
        .logit_4(logit_4),
        .cycle_count(cycle_count),
        .input_write_enable(input_write_enable),
        .input_write_address(input_write_address),
        .input_write_data(input_write_data)
    );

    always_ff @(posedge clk) begin
        if (cycle == 0) begin
            errors = 0;
            rst <= 1'b1;
            start <= 1'b0;
            input_write_enable <= 1'b0;
            input_write_address <= 12'd0;
            input_write_data <= 8'sd0;
        end else if (cycle == 4) begin
            rst <= 1'b0;
        end else if (cycle == 8) begin
            if (busy !== 1'b0 || done !== 1'b0) begin
                $display("FAIL idle after reset: busy=%0d done=%0d", busy, done);
                errors++;
            end
            // Exercise input-load port wiring (one write pulse)
            input_write_enable <= 1'b1;
            input_write_address <= 12'd0;
            input_write_data <= 8'sd7;
        end else if (cycle == 9) begin
            input_write_enable <= 1'b0;
            start <= 1'b1;
        end else if (cycle == 10) begin
            start <= 1'b0;
        end else if (cycle == 12) begin
            if (busy !== 1'b1) begin
                $display("FAIL expected busy after start");
                errors++;
            end else begin
                $display("PASS bd_wrapper elaborates; busy rises after start");
            end
            if (errors == 0)
                $display("PASS: tb_cnn_bd_wrapper_elab");
            else
                $display("FAIL: tb_cnn_bd_wrapper_elab errors=%0d", errors);
            $finish;
        end
        cycle <= cycle + 1;
    end

    initial cycle = 0;

endmodule
