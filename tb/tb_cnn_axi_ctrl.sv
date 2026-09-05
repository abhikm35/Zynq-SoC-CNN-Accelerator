// tb_cnn_axi_ctrl.sv
// AXI4-Lite register interface test for cnn_axi_ctrl (stubbed CNN ports).
// Drives the real AXI bus; does not poke internal slave registers.

`timescale 1ns / 1ps

module tb_cnn_axi_ctrl (
    input logic clk
);

    localparam int AW = 6;
    localparam int DW = 32;

    logic aresetn;

    logic [AW-1:0] awaddr;
    logic [2:0]    awprot;
    logic          awvalid;
    logic          awready;
    logic [DW-1:0] wdata;
    logic [3:0]    wstrb;
    logic          wvalid;
    logic          wready;
    logic [1:0]    bresp;
    logic          bvalid;
    logic          bready;
    logic [AW-1:0] araddr;
    logic [2:0]    arprot;
    logic          arvalid;
    logic          arready;
    logic [DW-1:0] rdata;
    logic [1:0]    rresp;
    logic          rvalid;
    logic          rready;

    logic          cnn_start;
    logic          cnn_busy;
    logic          cnn_done;
    logic [2:0]    cnn_predicted_class;
    logic signed [31:0] cnn_maximum_logit;
    logic signed [31:0] cnn_logit_0;
    logic signed [31:0] cnn_logit_1;
    logic signed [31:0] cnn_logit_2;
    logic signed [31:0] cnn_logit_3;
    logic signed [31:0] cnn_logit_4;
    logic [63:0]   cnn_cycle_count;

    int unsigned errors;
    int unsigned start_pulse_count;
    logic        prev_start;

    cnn_axi_ctrl #(
        .C_CNN_LOGIT_WIDTH(32),
        .C_S00_AXI_DATA_WIDTH(DW),
        .C_S00_AXI_ADDR_WIDTH(AW)
    ) dut (
        .cnn_start(cnn_start),
        .cnn_busy(cnn_busy),
        .cnn_done(cnn_done),
        .cnn_predicted_class(cnn_predicted_class),
        .cnn_maximum_logit(cnn_maximum_logit),
        .cnn_logit_0(cnn_logit_0),
        .cnn_logit_1(cnn_logit_1),
        .cnn_logit_2(cnn_logit_2),
        .cnn_logit_3(cnn_logit_3),
        .cnn_logit_4(cnn_logit_4),
        .cnn_cycle_count(cnn_cycle_count),
        .s00_axi_aclk(clk),
        .s00_axi_aresetn(aresetn),
        .s00_axi_awaddr(awaddr),
        .s00_axi_awprot(awprot),
        .s00_axi_awvalid(awvalid),
        .s00_axi_awready(awready),
        .s00_axi_wdata(wdata),
        .s00_axi_wstrb(wstrb),
        .s00_axi_wvalid(wvalid),
        .s00_axi_wready(wready),
        .s00_axi_bresp(bresp),
        .s00_axi_bvalid(bvalid),
        .s00_axi_bready(bready),
        .s00_axi_araddr(araddr),
        .s00_axi_arprot(arprot),
        .s00_axi_arvalid(arvalid),
        .s00_axi_arready(arready),
        .s00_axi_rdata(rdata),
        .s00_axi_rresp(rresp),
        .s00_axi_rvalid(rvalid),
        .s00_axi_rready(rready)
    );

    task automatic axi_write(input logic [AW-1:0] addr, input logic [DW-1:0] data,
                             input logic [3:0] strb = 4'hF);
        int guard;
        begin
            awaddr  = addr;
            awprot  = 3'b000;
            wdata   = data;
            wstrb   = strb;
            awvalid = 1'b1;
            wvalid  = 1'b1;
            bready  = 1'b1;
            guard = 0;
            // Wait until address+data accepted (same cycle or data follows)
            while (!(awvalid && awready && wvalid && wready)) begin
                @(posedge clk);
                guard++;
                if (guard > 1000) begin
                    $error("axi_write timeout waiting for AW/W ready");
                    errors++;
                    awvalid = 0; wvalid = 0;
                    return;
                end
            end
            @(posedge clk);
            awvalid = 1'b0;
            wvalid  = 1'b0;
            guard = 0;
            while (!bvalid) begin
                @(posedge clk);
                guard++;
                if (guard > 1000) begin
                    $error("axi_write timeout waiting for BVALID");
                    errors++;
                    bready = 0;
                    return;
                end
            end
            @(posedge clk);
            bready = 1'b0;
        end
    endtask

    task automatic axi_read(input logic [AW-1:0] addr, output logic [DW-1:0] data);
        int guard;
        begin
            araddr  = addr;
            arprot  = 3'b000;
            arvalid = 1'b1;
            rready  = 1'b1;
            guard = 0;
            while (!(arvalid && arready)) begin
                @(posedge clk);
                guard++;
                if (guard > 1000) begin
                    $error("axi_read timeout waiting for ARREADY");
                    errors++;
                    arvalid = 0;
                    data = '0;
                    return;
                end
            end
            @(posedge clk);
            arvalid = 1'b0;
            guard = 0;
            while (!rvalid) begin
                @(posedge clk);
                guard++;
                if (guard > 1000) begin
                    $error("axi_read timeout waiting for RVALID");
                    errors++;
                    data = '0;
                    return;
                end
            end
            data = rdata;
            @(posedge clk);
            rready = 1'b0;
        end
    endtask

    task automatic expect_eq(input string name, input logic [DW-1:0] got,
                             input logic [DW-1:0] exp);
        begin
            if (got !== exp) begin
                $display("FAIL %s: got 0x%08h exp 0x%08h", name, got, exp);
                errors++;
            end else begin
                $display("PASS %s: 0x%08h", name, got);
            end
        end
    endtask

    // Count rising edges of cnn_start
    always_ff @(posedge clk) begin
        if (!aresetn) begin
            start_pulse_count <= 0;
            prev_start <= 1'b0;
        end else begin
            if (cnn_start && !prev_start)
                start_pulse_count <= start_pulse_count + 1;
            prev_start <= cnn_start;
        end
    end

    initial begin
        logic [DW-1:0] rd;
        int unsigned pulses_before;

        errors = 0;
        aresetn = 1'b0;
        awaddr = '0; awprot = '0; awvalid = 0;
        wdata = '0; wstrb = '0; wvalid = 0; bready = 0;
        araddr = '0; arprot = '0; arvalid = 0; rready = 0;

        cnn_busy = 1'b0;
        cnn_done = 1'b0;
        cnn_predicted_class = 3'd0;
        cnn_maximum_logit = 32'sd0;
        cnn_logit_0 = 32'sd0;
        cnn_logit_1 = 32'sd0;
        cnn_logit_2 = 32'sd0;
        cnn_logit_3 = 32'sd0;
        cnn_logit_4 = 32'sd0;
        cnn_cycle_count = 64'd0;

        repeat (8) @(posedge clk);
        aresetn = 1'b1;
        repeat (8) @(posedge clk);

        // ---- Reset / idle ----
        if (cnn_start !== 1'b0) begin
            $display("FAIL reset: cnn_start != 0");
            errors++;
        end else $display("PASS reset: cnn_start == 0");

        axi_read(6'h04, rd);
        expect_eq("STATUS after reset", rd, 32'h0);
        axi_read(6'h00, rd);
        expect_eq("CONTROL after reset", rd, 32'h0);

        // ---- START pulse ----
        pulses_before = start_pulse_count;
        axi_write(6'h00, 32'h0000_0001, 4'hF);
        repeat (8) @(posedge clk);
        if (start_pulse_count != pulses_before + 1) begin
            $display("FAIL START: expected exactly one rising edge, got %0d",
                     start_pulse_count - pulses_before);
            errors++;
        end else $display("PASS START: exactly one pulse");

        if (cnn_start !== 1'b0) begin
            $display("FAIL START did not return low");
            errors++;
        end else $display("PASS START returned low without software clear");

        // Ensure no spontaneous second pulse
        pulses_before = start_pulse_count;
        repeat (10) @(posedge clk);
        if (start_pulse_count != pulses_before) begin
            $display("FAIL START: unexpected extra pulse");
            errors++;
        end else $display("PASS START: no second pulse");

        // ---- WSTRB: lane 0 disabled must not START ----
        pulses_before = start_pulse_count;
        axi_write(6'h00, 32'h0000_0001, 4'hE); // strobe excludes byte0
        repeat (4) @(posedge clk);
        if (start_pulse_count != pulses_before) begin
            $display("FAIL WSTRB: START fired without byte0 strobe");
            errors++;
        end else $display("PASS WSTRB: no START without byte0");

        // ---- START while busy ignored ----
        cnn_busy = 1'b1;
        pulses_before = start_pulse_count;
        axi_write(6'h00, 32'h0000_0001, 4'hF);
        repeat (4) @(posedge clk);
        if (start_pulse_count != pulses_before) begin
            $display("FAIL START-while-busy: pulse generated");
            errors++;
        end else $display("PASS START-while-busy: ignored");
        cnn_busy = 1'b0;
        axi_write(6'h00, 32'h0000_0001, 4'hF);
        repeat (4) @(posedge clk);
        if (start_pulse_count != pulses_before + 1) begin
            $display("FAIL START after busy clear");
            errors++;
        end else $display("PASS START after busy clear");

        // ---- STATUS.BUSY ----
        cnn_busy = 1'b0;
        axi_read(6'h04, rd);
        expect_eq("STATUS busy=0", rd, 32'h0);
        cnn_busy = 1'b1;
        axi_read(6'h04, rd);
        expect_eq("STATUS busy=1", rd, 32'h1);
        cnn_busy = 1'b0;

        // ---- Sticky DONE ----
        // Clear sticky via accepted START
        axi_write(6'h00, 32'h0000_0001, 4'hF);
        repeat (4) @(posedge clk);
        axi_read(6'h04, rd);
        if (rd[1]) begin
            $display("FAIL sticky DONE not cleared on START");
            errors++;
        end else $display("PASS sticky DONE cleared on START");

        cnn_done = 1'b1;
        @(posedge clk);
        cnn_done = 1'b0;
        repeat (8) @(posedge clk);
        axi_read(6'h04, rd);
        expect_eq("STATUS sticky DONE after one-cycle done", rd, 32'h2);

        axi_write(6'h00, 32'h0000_0001, 4'hF);
        repeat (4) @(posedge clk);
        axi_read(6'h04, rd);
        expect_eq("STATUS DONE cleared by new START", rd, 32'h0);

        // ---- Predicted class zero-extend ----
        cnn_predicted_class = 3'd3;
        axi_read(6'h08, rd);
        expect_eq("PREDICTED_CLASS=3", rd, 32'h0000_0003);
        cnn_predicted_class = 3'd4;
        axi_read(6'h08, rd);
        expect_eq("PREDICTED_CLASS=4", rd, 32'h0000_0004);

        // ---- Signed logits ----
        cnn_maximum_logit = 32'sd0;
        cnn_logit_0 = 32'sd42;
        cnn_logit_1 = -32'sd1;
        cnn_logit_2 = 32'sd2147483647;
        cnn_logit_3 = -32'sd2147483648;
        cnn_logit_4 = -32'sd2;
        axi_read(6'h0C, rd); expect_eq("MAX_LOGIT 0", rd, 32'h0000_0000);
        axi_read(6'h10, rd); expect_eq("LOGIT_0 +42", rd, 32'h0000_002A);
        axi_read(6'h14, rd); expect_eq("LOGIT_1 -1", rd, 32'hFFFF_FFFF);
        axi_read(6'h18, rd); expect_eq("LOGIT_2 max+", rd, 32'h7FFF_FFFF);
        axi_read(6'h1C, rd); expect_eq("LOGIT_3 max-", rd, 32'h8000_0000);
        axi_read(6'h20, rd); expect_eq("LOGIT_4 -2", rd, 32'hFFFF_FFFE);

        // ---- Cycle count 64-bit ----
        cnn_cycle_count = 64'h0000_00AB_1234_5678;
        axi_read(6'h24, rd); expect_eq("CYCLE_COUNT_LOW", rd, 32'h1234_5678);
        axi_read(6'h28, rd); expect_eq("CYCLE_COUNT_HIGH", rd, 32'h0000_00AB);

        // ---- Read-only writes must not change CNN-facing inputs/readback ----
        axi_write(6'h04, 32'hFFFF_FFFF, 4'hF);
        axi_write(6'h08, 32'hDEAD_BEEF, 4'hF);
        axi_write(6'h0C, 32'h1111_1111, 4'hF);
        axi_write(6'h10, 32'h2222_2222, 4'hF);
        axi_write(6'h24, 32'h3333_3333, 4'hF);
        axi_write(6'h28, 32'h4444_4444, 4'hF);
        axi_read(6'h08, rd); expect_eq("RO write ignored CLASS", rd, 32'h0000_0004);
        axi_read(6'h14, rd); expect_eq("RO write ignored LOGIT_1", rd, 32'hFFFF_FFFF);
        axi_read(6'h24, rd); expect_eq("RO write ignored CYCLE_L", rd, 32'h1234_5678);

        // ---- Reserved 0x2C ----
        axi_read(6'h2C, rd); expect_eq("RESERVED 0x2C", rd, 32'h0);
        axi_write(6'h2C, 32'hCAFE_BABE, 4'hF);
        axi_read(6'h2C, rd); expect_eq("RESERVED after write", rd, 32'h0);

        if (errors == 0)
            $display("PASS: tb_cnn_axi_ctrl");
        else
            $display("FAIL: tb_cnn_axi_ctrl errors=%0d", errors);

        $finish;
    end

endmodule
