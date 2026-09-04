// tb_argmax_real_logits.sv
// Sequential argmax over verified FC sample_000 logits.

`timescale 1ns / 1ps

module tb_argmax_real_logits (
    input logic clk
);
    logic rst, start, busy, argmax_done;
    logic [2:0] predicted_class;
    logic signed [31:0] maximum_logit;
    logic [3:0] read_count;
    logic [2:0] current_index;

    logic logit_read_enable;
    logic [2:0] logit_read_address;
    logic signed [31:0] logit_read_data;

    logic signed [31:0] expected_logits [0:4];
    integer expected_class;
    integer expected_max;
    integer local_c, phase, i;
    integer start_cycle, done_cycle, total_cycles;
    integer rd_addr, rd_wait;

    argmax_top #(
        .LOGIT_MEM_FILE("vectors/argmax/logits_input.mem")
    ) dut (
        .clk(clk), .rst(rst), .start(start),
        .busy(busy), .argmax_done(argmax_done),
        .predicted_class(predicted_class),
        .maximum_logit(maximum_logit),
        .read_count(read_count),
        .current_index(current_index),
        .logit_read_enable(logit_read_enable),
        .logit_read_address(logit_read_address),
        .logit_read_data(logit_read_data)
    );

    initial begin
        local_c=0; phase=0; rst=1; start=0;
        logit_read_enable=0; logit_read_address=0;
        start_cycle=0; done_cycle=0; total_cycles=0;
        rd_addr=0; rd_wait=0;
        $readmemh("vectors/argmax/logits_input.mem", expected_logits);
        expected_max = expected_logits[0];
        expected_class = 0;
        for (i=1; i<5; i=i+1) begin
            if ($signed(expected_logits[i]) > expected_max) begin
                expected_max = $signed(expected_logits[i]);
                expected_class = i;
            end
        end
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
            // ISSUE
            logit_read_enable = 1;
            logit_read_address = rd_addr[2:0];
            phase = 3;
            rd_wait = 0;
        end else if (phase == 3) begin
            // WAIT one cycle for sync ROM
            logit_read_enable = 0;
            phase = 4;
        end else if (phase == 4) begin
            // hold for capture on posedge
            ;
        end
    end

    always @(posedge clk) begin
        if (phase == 1 && argmax_done) begin
            done_cycle = local_c;
            total_cycles = done_cycle - start_cycle;
            if (read_count !== 4'd5)
                $fatal(1, "read_count=%0d", read_count);
            if ($signed(maximum_logit) !== expected_max) begin
                $display("FAIL max got=%0d exp=%0d", $signed(maximum_logit), expected_max);
                $fatal(1, "max mismatch");
            end
            if (predicted_class !== expected_class[2:0]) begin
                $display("FAIL class got=%0d exp=%0d", predicted_class, expected_class);
                $fatal(1, "class mismatch");
            end
            $display("==== Argmax real FC logits ====");
            $display("logits: %0d %0d %0d %0d %0d",
                     $signed(expected_logits[0]), $signed(expected_logits[1]),
                     $signed(expected_logits[2]), $signed(expected_logits[3]),
                     $signed(expected_logits[4]));
            $display("expected max=%0d class=%0d", expected_max, expected_class);
            $display("actual   max=%0d class=%0d", $signed(maximum_logit), predicted_class);
            $display("tie-breaking: lowest index (strict >)");
            $display("total cycles=%0d reads=%0d", total_cycles, read_count);
            phase = 2;
            rd_addr = 0;
        end

        if (phase == 4) begin
            if (logit_read_data !== expected_logits[rd_addr]) begin
                $display("FAIL logit[%0d] got=%0d exp=%0d",
                         rd_addr, $signed(logit_read_data),
                         $signed(expected_logits[rd_addr]));
                $fatal(1, "logit storage mutated");
            end
            rd_addr = rd_addr + 1;
            if (rd_addr >= 5) begin
                $display("PASS: tb_argmax_real_logits (logits unchanged)");
                $finish;
            end else begin
                phase = 2;
            end
        end

        if (local_c > 10000)
            $fatal(1, "timeout");
    end

endmodule
