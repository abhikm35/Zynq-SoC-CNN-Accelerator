// int32_sync_rom.sv
// Synchronous ROM for signed INT32 (bias / multiplier / shift tables).
// Same 1-cycle read latency as int8_sync_rom.

`timescale 1ns / 1ps

module int32_sync_rom #(
    parameter int DEPTH = 1,
    parameter int ADDR_WIDTH = 1,
    parameter MEM_FILE = ""
) (
    input  logic                      clk,
    input  logic                      read_enable,
    input  logic [ADDR_WIDTH-1:0]     read_address,
    output logic signed [31:0]        read_data
);

    logic signed [31:0] mem [0:DEPTH-1];

    initial begin
        if (MEM_FILE != "") begin
            $readmemh(MEM_FILE, mem);
        end
    end

    always_ff @(posedge clk) begin
        if (read_enable) begin
            read_data <= mem[read_address];
        end
    end

endmodule
