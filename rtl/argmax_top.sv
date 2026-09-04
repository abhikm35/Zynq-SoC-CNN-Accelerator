// argmax_top.sv
// Sequential signed argmax over five verified FC INT32 logits.
// Logits are held in a sync ROM (standalone tests) and remain readable
// without modification by argmax.

`timescale 1ns / 1ps

module argmax_top #(
    parameter int LOGIT_WIDTH = 32,
    parameter int NUM_CLASSES = 5,
    parameter LOGIT_MEM_FILE = "vectors/argmax/logits_input.mem"
) (
    input  logic                          clk,
    input  logic                          rst,
    input  logic                          start,

    output logic                          busy,
    output logic                          argmax_done,
    output logic [2:0]                    predicted_class,
    output logic signed [LOGIT_WIDTH-1:0] maximum_logit,
    output logic [3:0]                    read_count,
    output logic [2:0]                    current_index,

    // Preserve software / TB read access to original logits
    input  logic                          logit_read_enable,
    input  logic [2:0]                    logit_read_address,
    output logic signed [LOGIT_WIDTH-1:0] logit_read_data
);

    logic ctrl_re;
    logic [2:0] ctrl_addr;
    logic signed [LOGIT_WIDTH-1:0] mem_data;

    logic mux_re;
    logic [2:0] mux_addr;

    // Controller owns the port while busy; otherwise TB/software may read.
    assign mux_re   = busy ? ctrl_re   : logit_read_enable;
    assign mux_addr = busy ? ctrl_addr : logit_read_address;

    int32_sync_rom #(
        .DEPTH(NUM_CLASSES),
        .ADDR_WIDTH(3),
        .MEM_FILE(LOGIT_MEM_FILE)
    ) u_logit_rom (
        .clk          (clk),
        .read_enable  (mux_re),
        .read_address (mux_addr),
        .read_data    (mem_data)
    );

    assign logit_read_data = mem_data;

    signed_argmax5_controller #(
        .LOGIT_WIDTH(LOGIT_WIDTH)
    ) u_ctrl (
        .clk                (clk),
        .rst                (rst),
        .start              (start),
        .busy               (busy),
        .done               (argmax_done),
        .logit_read_enable  (ctrl_re),
        .logit_read_address (ctrl_addr),
        .logit_read_data    (mem_data),
        .predicted_class    (predicted_class),
        .maximum_logit      (maximum_logit),
        .current_index      (current_index),
        .read_count         (read_count)
    );

endmodule
