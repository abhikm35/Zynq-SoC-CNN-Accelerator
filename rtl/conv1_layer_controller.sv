// conv1_layer_controller.sv
// Full Conv1 layer: iterate all output channels x 32 x 32.
//
// Trained model: NUM_OUT_CHANNELS = 16 (prompt sketches with 8 are obsolete).
// Each output still runs the memory-driven single-output engine (27 MACs,
// all 3 input channels). Bias / multiplier / shift are selected by oc inside
// the engine (per-output-channel quantization).
//
// Counter order (column fastest):
//   for oc = 0 .. NUM_OUT_CHANNELS-1
//     for row = 0 .. 31
//       for col = 0 .. 31
//
// Output address:
//   output_write_address = oc * 1024 + row * 32 + col
//   (1024 = 32*32 spatial map size)

`timescale 1ns / 1ps

module conv1_layer_controller #(
    parameter int NUM_OUT_CHANNELS = 16
) (
    input  logic                   clk,
    input  logic                   rst,
    input  logic                   start,

    output logic                   engine_start,
    output logic [3:0]             engine_output_channel,
    output logic [4:0]             engine_output_row,
    output logic [4:0]             engine_output_column,
    input  logic                   engine_busy,
    input  logic                   engine_done,
    input  logic signed [7:0]      engine_relu_output,
    input  logic [4:0]             engine_mac_count,

    output logic                   busy,
    output logic                   conv1_done,
    output logic [14:0]            output_count, // 0 .. 16384
    output logic [3:0]             current_output_channel,
    output logic [4:0]             current_output_row,
    output logic [4:0]             current_output_column,

    output logic                   output_write_enable,
    output logic [13:0]            output_write_address,
    output logic signed [7:0]      output_write_data
);

    localparam int TOTAL_OUTPUTS = NUM_OUT_CHANNELS * 1024;
    // Trained Conv1: 16 channels -> last oc = 15
    localparam logic [3:0] LAST_OC = 4'd15;
    localparam logic [14:0] TOTAL_COUNT = 15'd16384;

    localparam logic [2:0] ST_IDLE         = 3'd0;
    localparam logic [2:0] ST_START_OUTPUT = 3'd1;
    localparam logic [2:0] ST_WAIT_OUTPUT  = 3'd2;
    localparam logic [2:0] ST_WRITE_OUTPUT = 3'd3;
    localparam logic [2:0] ST_ADVANCE      = 3'd4;
    localparam logic [2:0] ST_CONV1_DONE   = 3'd5;

    logic [2:0] state;
    logic [2:0] state_n;

    logic [3:0] oc_r;
    logic [4:0] row_r;
    logic [4:0] col_r;
    logic [14:0] count_r;
    logic signed [7:0] result_r;
    logic last_coord;

    assign engine_output_channel  = oc_r;
    assign engine_output_row      = row_r;
    assign engine_output_column   = col_r;
    assign current_output_channel = oc_r;
    assign current_output_row     = row_r;
    assign current_output_column  = col_r;
    assign output_count           = count_r;
    assign output_write_data      = result_r;

    // oc*1024 + row*32 + col
    assign output_write_address =
        ({oc_r, 10'd0}) + ({5'd0, row_r, 5'd0}) + {9'd0, col_r};

    assign last_coord =
        (oc_r == LAST_OC) && (row_r == 5'd31) && (col_r == 5'd31);

    assign engine_start        = (state == ST_START_OUTPUT);
    assign output_write_enable = (state == ST_WRITE_OUTPUT);

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= ST_IDLE;
            oc_r <= 4'd0;
            row_r <= 5'd0;
            col_r <= 5'd0;
            count_r <= 15'd0;
            result_r <= 8'sd0;
            busy <= 1'b0;
            conv1_done <= 1'b0;
        end else begin
            state <= state_n;
            conv1_done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        oc_r <= 4'd0;
                        row_r <= 5'd0;
                        col_r <= 5'd0;
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
                        if (col_r == 5'd31) begin
                            col_r <= 5'd0;
                            if (row_r == 5'd31) begin
                                row_r <= 5'd0;
                                oc_r <= oc_r + 4'd1;
                            end else begin
                                row_r <= row_r + 5'd1;
                            end
                        end else begin
                            col_r <= col_r + 5'd1;
                        end
                    end
                end
                ST_CONV1_DONE: begin
                    conv1_done <= 1'b1;
                    busy <= 1'b0;
                end
                default: ;
            endcase
        end
    end

    always_comb begin
        state_n = state;
        case (state)
            ST_IDLE: begin
                if (start)
                    state_n = ST_START_OUTPUT;
            end
            ST_START_OUTPUT: begin
                state_n = ST_WAIT_OUTPUT;
            end
            ST_WAIT_OUTPUT: begin
                if (engine_done)
                    state_n = ST_WRITE_OUTPUT;
            end
            ST_WRITE_OUTPUT: begin
                state_n = ST_ADVANCE;
            end
            ST_ADVANCE: begin
                if (last_coord)
                    state_n = ST_CONV1_DONE;
                else
                    state_n = ST_START_OUTPUT;
            end
            ST_CONV1_DONE: begin
                state_n = ST_IDLE;
            end
            default: state_n = ST_IDLE;
        endcase
    end

    // synopsys translate_off
    always_ff @(posedge clk) begin
        if (!rst) begin
            if (engine_start && engine_busy)
                $error("engine_start while engine_busy");
            if (output_write_enable &&
                ({1'b0, output_write_address} >= TOTAL_COUNT))
                $error("write address OOB");
            if (count_r > TOTAL_COUNT)
                $error("output_count exceeded");
            if (conv1_done && (count_r != TOTAL_COUNT))
                $error("conv1_done with bad count");
            if (conv1_done &&
                ((oc_r != LAST_OC) || (row_r != 5'd31) || (col_r != 5'd31)))
                $error("conv1_done with wrong final coordinate");
            if (engine_done && (engine_mac_count != 5'd27))
                $error("mac_count != 27");
        end
    end
    // synopsys translate_on

endmodule
