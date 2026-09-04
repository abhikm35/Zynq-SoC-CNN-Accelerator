// int8_sync_rom.sv
// Synchronous ROM / BRAM model: 1-cycle read latency.
//
// Cycle N:   read_enable and read_address presented
// Cycle N+1: read_data updates with mem[address] (after the posedge)

`timescale 1ns / 1ps

module int8_sync_rom #(
    parameter int DEPTH = 1,
    parameter int ADDR_WIDTH = 1,
    parameter MEM_FILE = ""
) (
    input  logic                      clk,
    input  logic                      read_enable,
    input  logic [ADDR_WIDTH-1:0]     read_address,
    output logic signed [7:0]         read_data
);

    logic signed [7:0] mem [0:DEPTH-1];

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
