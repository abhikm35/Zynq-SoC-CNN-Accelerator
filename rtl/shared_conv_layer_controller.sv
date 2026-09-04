// shared_conv_layer_controller.sv
// Full-layer iterator for the shared convolution engine.
//
// Conv1 (layer_is_conv2=0): oc=0..15, row/col=0..31, 16384 outputs
// Conv2 (layer_is_conv2=1): oc=0..31, row/col=0..15, 8192 outputs
//
// Same FSM as verified Conv1/Conv2 layer controllers:
//   IDLE -> START_OUTPUT -> WAIT_OUTPUT -> WRITE_OUTPUT -> ADVANCE -> (next|DONE)

`timescale 1ns / 1ps

module shared_conv_layer_controller (
    input  logic                   clk,
    input  logic                   rst,
    input  logic                   start,
    input  logic                   layer_is_conv2,

    output logic                   engine_start,
    output logic [4:0]             engine_output_channel,
    output logic [4:0]             engine_output_row,
    output logic [4:0]             engine_output_column,
    input  logic                   engine_busy,
    input  logic                   engine_done,
    input  logic signed [7:0]      engine_relu_output,
    input  logic [7:0]             engine_mac_count,

    output logic                   busy,
    output logic                   layer_done,
    output logic [14:0]            output_count,
    output logic [4:0]             current_output_channel,
    output logic [4:0]             current_output_row,
    output logic [4:0]             current_output_column,
    output logic                   dbg_layer_is_conv2,

    output logic                   output_write_enable,
    output logic [13:0]            output_write_address,
    output logic signed [7:0]      output_write_data
);

    localparam logic [2:0] ST_IDLE         = 3'd0;
    localparam logic [2:0] ST_START_OUTPUT = 3'd1;
    localparam logic [2:0] ST_WAIT_OUTPUT  = 3'd2;
    localparam logic [2:0] ST_WRITE_OUTPUT = 3'd3;
    localparam logic [2:0] ST_ADVANCE      = 3'd4;
    localparam logic [2:0] ST_LAYER_DONE   = 3'd5;

    logic [2:0] state, state_n;
    logic layer_r;
    logic [4:0] oc_r, row_r, col_r;
    logic [14:0] count_r;
    logic signed [7:0] result_r;

    logic [4:0] last_oc, last_row, last_col;
    logic [14:0] total_count;
    logic last_coord;

    assign last_oc     = layer_r ? 5'd31 : 5'd15;
    assign last_row    = layer_r ? 5'd15 : 5'd31;
    assign last_col    = layer_r ? 5'd15 : 5'd31;
    assign total_count = layer_r ? 15'd8192 : 15'd16384;

    assign engine_output_channel  = oc_r;
    assign engine_output_row      = row_r;
    assign engine_output_column   = col_r;
    assign current_output_channel = oc_r;
    assign current_output_row     = row_r;
    assign current_output_column  = col_r;
    assign output_count           = count_r;
    assign output_write_data      = result_r;
    assign dbg_layer_is_conv2     = layer_r;

    // Conv1: oc*1024+row*32+col ; Conv2: oc*256+row*16+col
    assign output_write_address = layer_r
        ? ({1'b0, oc_r, 8'd0} + {6'd0, row_r[3:0], 4'd0} + {10'd0, col_r[3:0]})
        : ({oc_r[3:0], 10'd0} + {5'd0, row_r, 5'd0} + {9'd0, col_r});

    assign last_coord =
        (oc_r == last_oc) && (row_r == last_row) && (col_r == last_col);

    assign engine_start        = (state == ST_START_OUTPUT);
    assign output_write_enable = (state == ST_WRITE_OUTPUT);

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= ST_IDLE;
            layer_r <= 1'b0;
            oc_r <= 5'd0; row_r <= 5'd0; col_r <= 5'd0;
            count_r <= 15'd0; result_r <= 8'sd0;
            busy <= 1'b0; layer_done <= 1'b0;
        end else begin
            state <= state_n;
            layer_done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        layer_r <= layer_is_conv2;
                        oc_r <= 5'd0; row_r <= 5'd0; col_r <= 5'd0;
                        count_r <= 15'd0;
                    end
                end
                ST_WAIT_OUTPUT: begin
                    if (engine_done)
                        result_r <= engine_relu_output;
                end
                ST_WRITE_OUTPUT: begin
                    count_r <= count_r + 15'd1;
                end
                ST_ADVANCE: begin
                    if (!last_coord) begin
                        if (col_r == last_col) begin
                            col_r <= 5'd0;
                            if (row_r == last_row) begin
                                row_r <= 5'd0;
                                oc_r <= oc_r + 5'd1;
                            end else begin
                                row_r <= row_r + 5'd1;
                            end
                        end else begin
                            col_r <= col_r + 5'd1;
                        end
                    end
                end
                ST_LAYER_DONE: begin
                    layer_done <= 1'b1;
                    busy <= 1'b0;
                end
                default: ;
            endcase
        end
    end

    always_comb begin
        state_n = state;
        unique case (state)
            ST_IDLE: if (start) state_n = ST_START_OUTPUT;
            ST_START_OUTPUT: state_n = ST_WAIT_OUTPUT;
            ST_WAIT_OUTPUT: if (engine_done) state_n = ST_WRITE_OUTPUT;
            ST_WRITE_OUTPUT: state_n = ST_ADVANCE;
            ST_ADVANCE: state_n = last_coord ? ST_LAYER_DONE : ST_START_OUTPUT;
            ST_LAYER_DONE: state_n = ST_IDLE;
            default: state_n = ST_IDLE;
        endcase
    end

    // synopsys translate_off
    always_ff @(posedge clk) begin
        if (!rst) begin
            if (busy && (layer_is_conv2 !== layer_r))
                $error("layer_is_conv2 changed while layer controller busy");
            if (engine_start && engine_busy)
                $error("engine_start while engine_busy");
            if (layer_done && (count_r != total_count))
                $error("layer_done with bad count %0d", count_r);
            if (engine_done) begin
                if (!layer_r && (engine_mac_count != 8'd27))
                    $error("Conv1 mac_count != 27");
                if (layer_r && (engine_mac_count != 8'd144))
                    $error("Conv2 mac_count != 144");
            end
        end
    end
    // synopsys translate_on

endmodule
