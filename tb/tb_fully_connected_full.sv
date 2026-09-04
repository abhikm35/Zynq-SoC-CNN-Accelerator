// tb_fully_connected_full.sv
// Complete FC layer (5 INT32 logits) bit-exact vs Python.

`timescale 1ns / 1ps

module tb_fully_connected_full (
    input logic clk
);
    localparam int NUM_CLASSES = 5;
    localparam int TOTAL_MACS = 160;

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

    logic signed [31:0] expected [0:NUM_CLASSES-1];
    logic signed [31:0] expected_acc [0:NUM_CLASSES-1];
    logic written [0:NUM_CLASSES-1];
    integer class_acc [0:NUM_CLASSES-1];
    integer class_logit [0:NUM_CLASSES-1];

    integer local_c, phase, i;
    integer writes_seen, macs_seen;
    integer start_cycle, done_cycle;
    integer mismatches;
    integer read_idx, pending_compare, compare_addr;
    integer total_cycles, avg_cycles, avg_mac_cycles;
    integer got, expv;

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
        writes_seen=0; macs_seen=0;
        start_cycle=0; done_cycle=0;
        mismatches=0;
        read_idx=0; pending_compare=0; compare_addr=0;
        for (i=0; i<NUM_CLASSES; i=i+1) begin
            written[i]=1'b0;
            class_acc[i]=0;
            class_logit[i]=0;
        end
        $readmemh("vectors/fc/fc_logits_expected.mem", expected);
        $readmemh("vectors/fc/fc_accumulators_expected.mem", expected_acc);
    end

    always @(negedge clk) begin
        local_c = local_c + 1;
        if (phase == 0) begin
            case (local_c)
                1,2: begin rst=1; start=0; end
                3: begin rst=0; end
                4: begin start=1; start_cycle=local_c; end
                5: begin start=0; phase=1; end
                default: ;
            endcase
        end else if (phase == 2) begin
            if (read_idx < NUM_CLASSES) begin
                logit_read_enable = 1;
                logit_read_address = read_idx[2:0];
            end else logit_read_enable = 0;
        end
    end

    always @(posedge clk) begin
        if (logit_write_enable) begin
            writes_seen = writes_seen + 1;
            if ({1'b0, logit_write_address} >= 4'd5)
                $fatal(1, "logit write OOB");
            if (written[logit_write_address])
                $fatal(1, "duplicate write %0d", logit_write_address);
            written[logit_write_address] = 1'b1;
            class_acc[logit_write_address] = engine_accumulator;
            class_logit[logit_write_address] = $signed(logit_write_data);
            macs_seen = macs_seen + engine_mac_count;

            if (logit_write_data !== expected[logit_write_address]) begin
                $display("MISMATCH class=%0d exp=%0d(%08h) got=%0d(%08h)",
                         logit_write_address,
                         $signed(expected[logit_write_address]),
                         expected[logit_write_address],
                         $signed(logit_write_data),
                         logit_write_data);
                mismatches = mismatches + 1;
            end
            if (engine_accumulator !== expected_acc[logit_write_address]) begin
                $display("ACC MISMATCH class=%0d exp=%0d got=%0d",
                         logit_write_address,
                         $signed(expected_acc[logit_write_address]),
                         $signed(engine_accumulator));
                mismatches = mismatches + 1;
            end
        end

        if (phase == 1 && fc_done) begin
            done_cycle = local_c;
            total_cycles = done_cycle - start_cycle;
            avg_cycles = total_cycles / NUM_CLASSES;
            avg_mac_cycles = total_cycles / TOTAL_MACS;
            if (writes_seen !== NUM_CLASSES)
                $fatal(1, "writes=%0d", writes_seen);
            if (total_mac_count !== TOTAL_MACS)
                $fatal(1, "dut macs=%0d", total_mac_count);
            if (macs_seen !== TOTAL_MACS)
                $fatal(1, "counted macs=%0d", macs_seen);
            for (i=0; i<NUM_CLASSES; i=i+1)
                if (!written[i])
                    $fatal(1, "missing write %0d", i);
            phase = 2;
            read_idx = 0;
            pending_compare = 0;
        end

        if (phase == 2) begin
            if (pending_compare) begin
                got = $signed(logit_read_data);
                expv = $signed(expected[compare_addr]);
                if (got !== expv) begin
                    $display("READBACK FAIL class=%0d exp=%0d got=%0d",
                             compare_addr, expv, got);
                    mismatches = mismatches + 1;
                end
                pending_compare = 0;
                read_idx = read_idx + 1;
            end
            if (logit_read_enable) begin
                pending_compare = 1;
                compare_addr = logit_read_address;
            end
            if (!pending_compare && read_idx >= NUM_CLASSES) begin
                $display("==== FC full-layer summary ====");
                $display("Expected class calcs  : %0d", NUM_CLASSES);
                $display("Actual class calcs    : %0d", writes_seen);
                $display("Expected MAC count    : %0d", TOTAL_MACS);
                $display("Actual MAC count      : %0d", total_mac_count);
                $display("Expected logit writes : %0d", NUM_CLASSES);
                $display("Actual logit writes   : %0d", writes_seen);
                $display("Total mismatches      : %0d", mismatches);
                $display("Total cycles          : %0d", total_cycles);
                $display("Avg cycles / class    : %0d", avg_cycles);
                $display("Avg cycles / MAC      : %0d", avg_mac_cycles);
                $display("");
                $display("  class    acc   logit");
                for (i=0; i<NUM_CLASSES; i=i+1)
                    $display("  %5d %6d %7d", i, class_acc[i], class_logit[i]);
                if (mismatches !== 0)
                    $fatal(1, "FC mismatches=%0d", mismatches);
                $display("PASS: tb_fully_connected_full (%0d logits bit-exact)", NUM_CLASSES);
                $finish;
            end
        end

        if (local_c > 200000)
            $fatal(1, "timeout");
    end

endmodule
