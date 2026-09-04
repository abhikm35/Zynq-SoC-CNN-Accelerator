// tb_fully_connected_selected_class.sv
// Test selected classes (0, 2, 4) bit-exact vs Python traces.

`timescale 1ns / 1ps

module tb_fully_connected_selected_class (
    input logic clk
);
    localparam int NSEL = 3;
    localparam int SEL0 = 0;
    localparam int SEL1 = 2;
    localparam int SEL2 = 4;

    logic rst, start, busy, fc_done;
    logic [2:0] class_index;
    logic [3:0] class_count;
    logic [7:0] total_mac_count;

    logic signed [31:0] engine_logit, engine_accumulator;
    logic [5:0] engine_mac_count;
    logic [4:0] engine_input_index;
    logic signed [15:0] engine_product;
    logic signed [31:0] engine_running_acc;

    logic logit_write_enable;
    logic [2:0] logit_write_address;
    logic signed [31:0] logit_write_data;

    logic logit_read_enable;
    logic [2:0] logit_read_address;
    logic signed [31:0] logit_read_data;

    logic signed [31:0] expected_logit [0:4];
    logic signed [31:0] expected_acc [0:4];

    integer local_c, phase, i;
    integer writes_seen;
    integer mismatches;

    fully_connected_top dut (
        .clk(clk), .rst(rst), .start(start),
        .busy(busy), .fc_done(fc_done),
        .class_index(class_index),
        .class_count(class_count),
        .total_mac_count(total_mac_count),
        .engine_logit(engine_logit),
        .engine_accumulator(engine_accumulator),
        .engine_mac_count(engine_mac_count),
        .engine_input_index(engine_input_index),
        .engine_product(engine_product),
        .engine_running_acc(engine_running_acc),
        .logit_write_enable(logit_write_enable),
        .logit_write_address(logit_write_address),
        .logit_write_data(logit_write_data),
        .logit_read_enable(logit_read_enable),
        .logit_read_address(logit_read_address),
        .logit_read_data(logit_read_data)
    );

    initial begin
        local_c=0; phase=0; rst=1; start=0;
        logit_read_enable=0; logit_read_address=0;
        writes_seen=0; mismatches=0;
        $readmemh("vectors/fc/fc_logits_expected.mem", expected_logit);
        $readmemh("vectors/fc/fc_accumulators_expected.mem", expected_acc);
    end

    always @(negedge clk) begin
        local_c = local_c + 1;
        if (phase == 0) begin
            case (local_c)
                1,2: begin rst=1; start=0; end
                3: begin rst=0; end
                4: begin start=1; end
                5: begin start=0; phase=1; end
                default: ;
            endcase
        end
    end

    always @(posedge clk) begin
        if (logit_write_enable) begin
            writes_seen = writes_seen + 1;
            if (logit_write_data !== expected_logit[logit_write_address]) begin
                $display("FAIL class=%0d logit got=%0d exp=%0d acc=%0d exp_acc=%0d",
                         logit_write_address,
                         $signed(logit_write_data),
                         $signed(expected_logit[logit_write_address]),
                         $signed(engine_accumulator),
                         $signed(expected_acc[logit_write_address]));
                mismatches = mismatches + 1;
                $fatal(1, "logit mismatch");
            end
            if (engine_accumulator !== expected_acc[logit_write_address]) begin
                $display("FAIL class=%0d accumulator got=%0d exp=%0d",
                         logit_write_address,
                         $signed(engine_accumulator),
                         $signed(expected_acc[logit_write_address]));
                $fatal(1, "accumulator mismatch");
            end
            if (logit_write_address == SEL0 ||
                logit_write_address == SEL1 ||
                logit_write_address == SEL2) begin
                $display("PASS selected class=%0d acc=%0d logit=%0d macs=%0d",
                         logit_write_address,
                         $signed(engine_accumulator),
                         $signed(logit_write_data),
                         engine_mac_count);
            end
        end

        if (fc_done) begin
            if (writes_seen !== 5)
                $fatal(1, "writes=%0d", writes_seen);
            if (total_mac_count !== 8'd160)
                $fatal(1, "total_mac=%0d", total_mac_count);
            if (mismatches !== 0)
                $fatal(1, "mismatches");
            $display("PASS: tb_fully_connected_selected_class classes=%0d macs=%0d",
                     NSEL, total_mac_count);
            $finish;
        end

        if (local_c > 200000)
            $fatal(1, "timeout");
    end

endmodule
