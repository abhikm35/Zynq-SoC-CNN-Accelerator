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
    logic          cnn_input_write_enable;
    logic [11:0]   cnn_input_write_address;
    logic [7:0]    cnn_input_write_data;

    int unsigned errors;
    int unsigned start_pulse_count;
    int unsigned we_pulse_count;
    logic        prev_start;
    logic        prev_we;

    // Capture last WE cycle (addr/data sampled while enable high)
    logic [11:0] last_we_addr;
    logic [7:0]  last_we_data;
    int unsigned last_we_high_cycles;

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
        .cnn_input_write_enable(cnn_input_write_enable),
        .cnn_input_write_address(cnn_input_write_address),
        .cnn_input_write_data(cnn_input_write_data),
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

    // Count rising edges of cnn_start / input WE
    always_ff @(posedge clk) begin
        if (!aresetn) begin
            start_pulse_count <= 0;
            prev_start <= 1'b0;
            we_pulse_count <= 0;
            prev_we <= 1'b0;
            last_we_high_cycles <= 0;
        end else begin
            if (cnn_start && !prev_start)
                start_pulse_count <= start_pulse_count + 1;
            prev_start <= cnn_start;

            if (cnn_input_write_enable && !prev_we) begin
                we_pulse_count <= we_pulse_count + 1;
                last_we_addr <= cnn_input_write_address;
                last_we_data <= cnn_input_write_data;
                last_we_high_cycles <= 1;
            end else if (cnn_input_write_enable) begin
                last_we_high_cycles <= last_we_high_cycles + 1;
            end
            prev_we <= cnn_input_write_enable;
        end
    end

    task automatic do_input_write(input logic [11:0] addr, input logic [7:0] data);
        begin
            axi_write(6'h30, {20'h0, addr}, 4'hF);
            axi_write(6'h34, {24'h0, data}, 4'hF);
            axi_write(6'h38, 32'h0000_0001, 4'hF);
            repeat (4) @(posedge clk);
        end
    endtask

    initial begin
        logic [DW-1:0] rd;
        int unsigned pulses_before;
        int unsigned we_before;
        logic [7:0] golden_mem [0:3071];
        int i;
        int load_errs;

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

        if (cnn_input_write_enable !== 1'b0 ||
            cnn_input_write_address !== 12'd0 ||
            cnn_input_write_data !== 8'd0) begin
            $display("FAIL reset: input loader not cleared");
            errors++;
        end else $display("PASS reset: input loader cleared");

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

        // ============================================================
        // INPUT loader (0x30 / 0x34 / 0x38)
        // ============================================================

        // Test A - address holding register
        axi_write(6'h30, 32'd1234, 4'hF);
        repeat (2) @(posedge clk);
        if (cnn_input_write_address !== 12'd1234) begin
            $display("FAIL INPUT_ADDRESS port: got %0d", cnn_input_write_address);
            errors++;
        end else $display("PASS INPUT_ADDRESS port == 1234");
        axi_read(6'h30, rd);
        expect_eq("INPUT_ADDRESS readback", rd, 32'd1234);

        // Test B - data holding register (0xFE as raw INT8 bits)
        axi_write(6'h34, 32'h0000_00FE, 4'hF);
        repeat (2) @(posedge clk);
        if (cnn_input_write_data !== 8'hFE) begin
            $display("FAIL INPUT_DATA port: got 0x%02h", cnn_input_write_data);
            errors++;
        end else $display("PASS INPUT_DATA port == 0xFE");
        axi_read(6'h34, rd);
        expect_eq("INPUT_DATA readback", rd, 32'h0000_00FE);

        // Test C/D - one-cycle WRITE pulse, not sticky
        we_before = we_pulse_count;
        axi_write(6'h38, 32'h0000_0001, 4'hF);
        repeat (8) @(posedge clk);
        if (we_pulse_count != we_before + 1) begin
            $display("FAIL INPUT_COMMAND: expected one WE rising edge, got %0d",
                     we_pulse_count - we_before);
            errors++;
        end else $display("PASS INPUT_COMMAND: exactly one WE pulse");
        if (last_we_high_cycles != 1) begin
            $display("FAIL INPUT_COMMAND: WE high for %0d cycles (want 1)",
                     last_we_high_cycles);
            errors++;
        end else $display("PASS INPUT_COMMAND: WE high exactly one cycle");
        if (last_we_addr !== 12'd1234 || last_we_data !== 8'hFE) begin
            $display("FAIL INPUT_COMMAND capture: addr=%0d data=0x%02h",
                     last_we_addr, last_we_data);
            errors++;
        end else $display("PASS INPUT_COMMAND: addr/data stable during WE");
        if (cnn_input_write_enable !== 1'b0) begin
            $display("FAIL INPUT_COMMAND sticky WE");
            errors++;
        end else $display("PASS INPUT_COMMAND: WE returned low");
        axi_read(6'h38, rd);
        expect_eq("INPUT_COMMAND readback", rd, 32'h0);

        // Test E - busy protection
        cnn_busy = 1'b1;
        we_before = we_pulse_count;
        axi_write(6'h30, 32'd10, 4'hF);
        axi_write(6'h34, 32'h55, 4'hF);
        axi_write(6'h38, 32'h1, 4'hF);
        repeat (4) @(posedge clk);
        if (we_pulse_count != we_before) begin
            $display("FAIL busy protect: WE while busy");
            errors++;
        end else $display("PASS busy protect: WE ignored while busy");
        // Holding regs still update while busy
        axi_read(6'h30, rd);
        expect_eq("INPUT_ADDRESS while busy", rd, 32'd10);
        axi_read(6'h34, rd);
        expect_eq("INPUT_DATA while busy", rd, 32'h55);
        cnn_busy = 1'b0;

        // Test F - invalid address 3072
        we_before = we_pulse_count;
        axi_write(6'h30, 32'd3072, 4'hF);
        axi_write(6'h34, 32'hAA, 4'hF);
        axi_write(6'h38, 32'h1, 4'hF);
        repeat (4) @(posedge clk);
        if (we_pulse_count != we_before) begin
            $display("FAIL invalid addr: WE for 3072");
            errors++;
        end else $display("PASS invalid addr: WE ignored for 3072");

        // Test G - boundary 0 and 3071
        we_before = we_pulse_count;
        do_input_write(12'd0, 8'h11);
        if (we_pulse_count != we_before + 1 || last_we_addr !== 12'd0 ||
            last_we_data !== 8'h11) begin
            $display("FAIL boundary addr 0");
            errors++;
        end else $display("PASS boundary addr 0");

        we_before = we_pulse_count;
        do_input_write(12'd3071, 8'h22);
        if (we_pulse_count != we_before + 1 || last_we_addr !== 12'd3071 ||
            last_we_data !== 8'h22) begin
            $display("FAIL boundary addr 3071");
            errors++;
        end else $display("PASS boundary addr 3071");

        // Test H - START still works after input loader use
        pulses_before = start_pulse_count;
        we_before = we_pulse_count;
        axi_write(6'h00, 32'h1, 4'hF);
        repeat (4) @(posedge clk);
        if (start_pulse_count != pulses_before + 1) begin
            $display("FAIL START after loader");
            errors++;
        end else $display("PASS START after loader");
        if (we_pulse_count != we_before) begin
            $display("FAIL START must not pulse WE");
            errors++;
        end else $display("PASS START independent of WE");

        // Loading must not START
        pulses_before = start_pulse_count;
        do_input_write(12'd5, 8'h33);
        if (start_pulse_count != pulses_before) begin
            $display("FAIL loader must not START");
            errors++;
        end else $display("PASS loader independent of START");

        // Test I - short image loader sequence
        begin
            logic [7:0] seq_data [0:3];
            logic [11:0] seq_addr [0:3];
            seq_addr[0] = 0; seq_data[0] = 8'h12;
            seq_addr[1] = 1; seq_data[1] = 8'hFE;
            seq_addr[2] = 2; seq_data[2] = 8'h7F;
            seq_addr[3] = 3; seq_data[3] = 8'h80;
            for (i = 0; i < 4; i++) begin
                we_before = we_pulse_count;
                do_input_write(seq_addr[i], seq_data[i]);
                if (we_pulse_count != we_before + 1 ||
                    last_we_addr !== seq_addr[i] ||
                    last_we_data !== seq_data[i] ||
                    last_we_high_cycles != 1) begin
                    $display("FAIL short load i=%0d addr=%0d data=%02h",
                             i, last_we_addr, last_we_data);
                    errors++;
                end
            end
            if (errors == 0)
                $display("PASS short image loader sequence (4 bytes)");
        end

        // Full 3072-byte golden vector through AXI loader interface
        $readmemh("vectors/end_to_end/input_image.mem", golden_mem);
        load_errs = 0;
        we_before = we_pulse_count;
        for (i = 0; i < 3072; i++) begin
            axi_write(6'h30, i, 4'hF);
            axi_write(6'h34, {24'h0, golden_mem[i]}, 4'hF);
            axi_write(6'h38, 32'h1, 4'hF);
            repeat (4) @(posedge clk);
            if (we_pulse_count != we_before + i + 1 ||
                last_we_addr !== i[11:0] ||
                last_we_data !== golden_mem[i] ||
                last_we_high_cycles != 1) begin
                load_errs++;
                if (load_errs <= 4)
                    $display("FAIL full load @%0d: we=%0d addr=%0d data=%02h (exp %02h)",
                             i, we_pulse_count, last_we_addr, last_we_data,
                             golden_mem[i]);
            end
        end
        if (we_pulse_count != we_before + 3072 || load_errs != 0) begin
            $display("FAIL full 3072 load: we_delta=%0d load_errs=%0d",
                     we_pulse_count - we_before, load_errs);
            errors++;
        end else $display("PASS full 3072-byte golden input load");

        // Status/results still intact after loader stress
        axi_read(6'h04, rd);
        expect_eq("STATUS after loader", rd, 32'h0);
        axi_read(6'h08, rd);
        expect_eq("CLASS after loader", rd, 32'h0000_0004);
        axi_read(6'h14, rd);
        expect_eq("LOGIT_1 after loader", rd, 32'hFFFF_FFFF);

        if (errors == 0)
            $display("PASS: tb_cnn_axi_ctrl");
        else
            $display("FAIL: tb_cnn_axi_ctrl errors=%0d", errors);

        $finish;
    end

endmodule
