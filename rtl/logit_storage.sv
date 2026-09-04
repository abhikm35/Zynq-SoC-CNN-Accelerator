// logit_storage.sv
// Five-entry signed INT32 storage for classifier logits.
// Thin wrapper around int32_sync_ram (1-cycle sync read).

`timescale 1ns / 1ps

module logit_storage #(
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

    int32_sync_ram #(
        .DEPTH(DEPTH),
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
