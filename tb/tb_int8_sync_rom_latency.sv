// tb_int8_sync_rom_latency.sv
// Verifies 1-cycle synchronous ROM read latency and enable gating.

`timescale 1ns / 1ps

module tb_int8_sync_rom_latency (
    input logic clk
);
    logic read_enable;
    logic [11:0] read_address;
    logic signed [7:0] read_data;
    logic signed [7:0] prev_data;
    integer local_c;
    integer fd;
    integer scan_code;
    reg [31:0] hv;
    logic signed [7:0] expected0;
    logic signed [7:0] expected10;

    int8_sync_rom #(
        .DEPTH(3072),
        .ADDR_WIDTH(12),
        .MEM_FILE("vectors/conv1_memory/input_image.mem")
    ) dut (
        .clk(clk),
        .read_enable(read_enable),
        .read_address(read_address),
        .read_data(read_data)
    );

    initial begin
        local_c = 0;
        read_enable = 0;
        read_address = 0;
        prev_data = 8'sd0;
        fd = $fopen("vectors/conv1_memory/input_image.mem", "r");
        if (fd == 0) $fatal(1, "missing input_image.mem");
        scan_code = $fscanf(fd, "%h", hv);
        expected0 = hv[7:0];
        // skip to index 10
        begin : skip
            integer i;
            for (i = 1; i <= 10; i = i + 1)
                scan_code = $fscanf(fd, "%h", hv);
        end
        expected10 = hv[7:0];
        $fclose(fd);
    end

    always @(negedge clk) begin
        local_c = local_c + 1;
        case (local_c)
            1: begin
                read_enable = 1;
                read_address = 12'd0;
            end
            2: begin
                // Data for addr0 must NOT be required yet at start of cycle 2;
                // after posedge of cycle2 transition it becomes valid.
                read_enable = 1;
                read_address = 12'd10;
            end
            3: begin
                read_enable = 0;
                read_address = 12'd0;
            end
            4: begin
                // Hold: no enable — data must remain expected10
                read_enable = 0;
            end
            default: ;
        endcase
    end

    always @(posedge clk) begin
        if (local_c == 1) begin
            // Address issued this cycle; data still stale / X — do not compare yet
            prev_data = read_data;
        end
        if (local_c == 2) begin
            if (read_data !== expected0) begin
                $display("FAIL latency: cycle2 data got %0d exp %0d (addr0)",
                         read_data, expected0);
                $fatal(1);
            end
        end
        if (local_c == 3) begin
            if (read_data !== expected10) begin
                $display("FAIL latency: cycle3 data got %0d exp %0d (addr10)",
                         read_data, expected10);
                $fatal(1);
            end
        end
        if (local_c == 4) begin
            if (read_data !== expected10) begin
                $display("FAIL latency: data changed without read_enable");
                $fatal(1);
            end
            $display("PASS tb_int8_sync_rom_latency");
            $finish;
        end
    end
endmodule
