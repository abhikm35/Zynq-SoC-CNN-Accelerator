// int32_sync_ram.sv
// Synchronous dual-port INT32 RAM (write + read), 1-cycle read latency.

`timescale 1ns / 1ps

module int32_sync_ram #(
    parameter int DEPTH = 5,
    parameter int ADDR_WIDTH = 3
) (
    input  logic                    clk,

    input  logic                    write_enable,
    input  logic [ADDR_WIDTH-1:0]   write_address,
    input  logic signed [31:0]      write_data,

    input  logic                    read_enable,
    input  logic [ADDR_WIDTH-1:0]   read_address,
    output logic signed [31:0]      read_data
);

    logic signed [31:0] mem [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (write_enable) begin
            mem[write_address] <= write_data;
        end
        if (read_enable) begin
            read_data <= mem[read_address];
        end
    end

endmodule
