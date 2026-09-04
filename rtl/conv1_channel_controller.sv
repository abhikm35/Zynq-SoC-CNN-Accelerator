// conv1_channel_controller.sv
// Iterates all 32x32 spatial outputs of Conv1 output_channel = 0.
//
// For each (row, col):
//   start memory-driven single-output engine -> wait done -> write ReLU to RAM
// Column increments fastest. Exactly 1024 writes; then channel_done.
//
// Each output still combines all 3 input channels (27 MACs) inside the engine.
//
// FSM:
//   IDLE -> START_OUTPUT -> WAIT_OUTPUT -> WRITE_OUTPUT -> ADVANCE
//        -> (more coords ? START_OUTPUT : CHANNEL_DONE) -> IDLE
//
// Output address (channel-local for oc=0):
//   output_write_address = output_row * 32 + output_column
// Future full Conv1:
//   full_output_address = output_channel * 1024 + output_row * 32 + output_column

`timescale 1ns / 1ps

module conv1_channel_controller (
    input  logic                   clk,
    input  logic                   rst,
    input  logic                   start,

    // Handshake to conv1_memory_single_output
    output logic                   engine_start,
    output logic [3:0]             engine_output_channel, // fixed 0
    output logic [4:0]             engine_output_row,
    output logic [4:0]             engine_output_column,
    input  logic                   engine_busy,
    input  logic                   engine_done,
    input  logic signed [7:0]      engine_relu_output,
    input  logic [4:0]             engine_mac_count,

    output logic                   busy,
    output logic                   channel_done,
    output logic [10:0]            output_count, // 0 .. 1024
    output logic [4:0]             current_output_row,
    output logic [4:0]             current_output_column,

    output logic                   output_write_enable,
    output logic [9:0]             output_write_address,
    output logic signed [7:0]      output_write_data
);

    localparam logic [2:0] ST_IDLE         = 3'd0;
    localparam logic [2:0] ST_START_OUTPUT = 3'd1;
    localparam logic [2:0] ST_WAIT_OUTPUT  = 3'd2;
    localparam logic [2:0] ST_WRITE_OUTPUT = 3'd3;
    localparam logic [2:0] ST_ADVANCE      = 3'd4;
    localparam logic [2:0] ST_CHANNEL_DONE = 3'd5;

    logic [2:0] state;
    logic [2:0] state_n;

    logic [4:0] row_r;
    logic [4:0] col_r;
    logic [10:0] count_r;
    logic signed [7:0] result_r;
    logic last_coord;

    assign engine_output_channel = 4'd0;
    assign engine_output_row     = row_r;
    assign engine_output_column  = col_r;
    assign current_output_row    = row_r;
    assign current_output_column = col_r;
    assign output_count          = count_r;
    // Local address for output channel 0: row*32 + col (no oc*1024 yet)
    assign output_write_address  = {row_r, 5'd0} + {5'd0, col_r};
    assign output_write_data     = result_r;
    assign last_coord            = (row_r == 5'd31) && (col_r == 5'd31);

    // One-cycle pulses while in the corresponding state
    assign engine_start        = (state == ST_START_OUTPUT);
    assign output_write_enable = (state == ST_WRITE_OUTPUT);

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= ST_IDLE;
            row_r <= 5'd0;
            col_r <= 5'd0;
            count_r <= 11'd0;
            result_r <= 8'sd0;
            busy <= 1'b0;
            channel_done <= 1'b0;
        end else begin
            state <= state_n;
            channel_done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        row_r <= 5'd0;
                        col_r <= 5'd0;
                        count_r <= 11'd0;
                    end
                end
                ST_WAIT_OUTPUT: begin
                    if (engine_done)
                        result_r <= engine_relu_output;
                end
                ST_WRITE_OUTPUT: begin
                    count_r <= count_r + 11'd1;
                end
                ST_ADVANCE: begin
                    if (!last_coord) begin
                        if (col_r == 5'd31) begin
                            col_r <= 5'd0;
                            row_r <= row_r + 5'd1;
                        end else begin
                            col_r <= col_r + 5'd1;
                        end
                    end
                end
                ST_CHANNEL_DONE: begin
                    channel_done <= 1'b1;
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
                    state_n = ST_CHANNEL_DONE;
                else
                    state_n = ST_START_OUTPUT;
            end
            ST_CHANNEL_DONE: begin
                state_n = ST_IDLE;
            end
            default: state_n = ST_IDLE;
        endcase
    end

    // Simulation assertions
    // synopsys translate_off
    always_ff @(posedge clk) begin
        if (!rst) begin
            if (output_write_enable && ({1'b0, output_write_address} >= 11'd1024))
                $error("write address >= 1024");
            if (count_r > 11'd1024)
                $error("output_count exceeded 1024");
            if (channel_done && (count_r != 11'd1024))
                $error("channel_done with count != 1024");
            if (state != ST_IDLE) begin
                if (row_r > 5'd31 || col_r > 5'd31)
                    $error("coordinate out of range");
            end
            if (engine_done && (engine_mac_count != 5'd27))
                $error("engine finished with mac_count != 27");
            if (engine_output_channel != 4'd0)
                $error("output channel must remain 0");
            if (engine_start && engine_busy)
                $error("engine_start while engine_busy");
        end
    end
    // synopsys translate_on

endmodule
