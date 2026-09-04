// int8_sync_ram.sv
// Synchronous dual-port INT8 RAM (write + read), 1-cycle read latency.
// Parameterized depth: 1024 (one channel) or 16384 (full Conv1 16x32x32).

`timescale 1ns / 1ps

module int8_sync_ram #(
    parameter int DEPTH = 1024,
    parameter int ADDR_WIDTH = 10
) (
    input  logic                    clk,

    input  logic                    write_enable,
    input  logic [ADDR_WIDTH-1:0]   write_address,
    input  logic signed [7:0]       write_data,

    input  logic                    read_enable,
    input  logic [ADDR_WIDTH-1:0]   read_address,
    output logic signed [7:0]       read_data
);

    logic signed [7:0] mem [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (write_enable) begin
            mem[write_address] <= write_data;
        end
        if (read_enable) begin
            read_data <= mem[read_address];
        end
    end

endmodule
