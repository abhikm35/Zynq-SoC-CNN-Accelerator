// activation_ram.sv
// Ping-pong activation buffer: 16384 x signed INT8.
//
// Sized for the largest trained activation tensor (Conv1: 16 x 32 x 32).
// Thin wrapper around int8_sync_ram — preserves 1-cycle synchronous read
// latency expected by all verified stage controllers.
//
// BRAM-friendly: synchronous write + synchronous read, no async read,
// no reset of the memory array.

`timescale 1ns / 1ps

module activation_ram #(
    parameter int DEPTH      = 16384,
    parameter int ADDR_WIDTH = 14
) (
    input  logic                    clk,

    input  logic                    write_enable,
    input  logic [ADDR_WIDTH-1:0]   write_address,
    input  logic signed [7:0]       write_data,

    input  logic                    read_enable,
    input  logic [ADDR_WIDTH-1:0]   read_address,
    output logic signed [7:0]       read_data
);

    int8_sync_ram #(
        .DEPTH     (DEPTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) u_ram (
        .clk           (clk),
        .write_enable  (write_enable),
        .write_address (write_address),
        .write_data    (write_data),
        .read_enable   (read_enable),
        .read_address  (read_address),
        .read_data     (read_data)
    );

endmodule
