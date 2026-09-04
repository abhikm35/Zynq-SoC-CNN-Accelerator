// shared_maxpool_engine.sv
// One reusable signed INT8 2x2 stride-2 max-pool for Pool1 and Pool2.
//
// Trained sizes (obsolete 8/16 prompt sketches are not used):
//   layer_is_pool2=0 -> Pool1: 16 x 32x32 -> 16 x 16x16 (4096 outs, 16384 reads)
//   layer_is_pool2=1 -> Pool2: 32 x 16x16 -> 32 x  8x8  (2048 outs,  8192 reads)
//
// Same FSM as verified MaxPool1/MaxPool2 controllers:
//   ISSUE -> WAIT -> CAPTURE  (window 0..3)
//   COMPARE -> WRITE -> ADVANCE -> (next | DONE)
//
// max4_int8 compares all four captured values (no init-to-zero).
// Activations: EXTERNAL 1-cycle sync RAM.

`timescale 1ns / 1ps

module shared_maxpool_engine (
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    start,
    input  logic                    layer_is_pool2,  // 0=Pool1, 1=Pool2

    output logic                    busy,
    output logic                    done,

    output logic                    activation_read_enable,
    output logic [13:0]             activation_read_address,
    input  logic signed [7:0]       activation_read_data,

    output logic                    output_write_enable,
    output logic [11:0]             output_write_address,
    output logic signed [7:0]       output_write_data,

    output logic [4:0]              current_channel,
    output logic [3:0]              current_pool_row,
    output logic [3:0]              current_pool_column,
    output logic [1:0]              current_window_index,
    output logic [12:0]             output_count,
    output logic [14:0]             read_count,
    output logic                    dbg_layer_is_pool2,

    output logic signed [7:0]       value_a,
    output logic signed [7:0]       value_b,
    output logic signed [7:0]       value_c,
    output logic signed [7:0]       value_d,
    output logic signed [7:0]       maximum_value
);

    localparam logic [3:0] ST_IDLE     = 4'd0;
    localparam logic [3:0] ST_ISSUE    = 4'd1;
    localparam logic [3:0] ST_WAIT     = 4'd2;
    localparam logic [3:0] ST_CAPTURE  = 4'd3;
    localparam logic [3:0] ST_COMPARE  = 4'd4;
    localparam logic [3:0] ST_WRITE    = 4'd5;
    localparam logic [3:0] ST_ADVANCE  = 4'd6;
    localparam logic [3:0] ST_DONE     = 4'd7;

    logic [3:0] state, state_n;
    logic layer_r;

    logic [4:0] ch_r;
    logic [3:0] pr_r, pc_r;
    logic [1:0] win_r;
    logic [12:0] count_r;
    logic [14:0] reads_r;

    logic signed [7:0] a_r, b_r, c_r, d_r;
    logic signed [7:0] max_w;

    logic [4:0] in_row_w, in_col_w;
    logic [13:0] act_addr_w;
    logic [11:0] pool_addr_w;

    logic [4:0] last_ch;
    logic [3:0] last_pr, last_pc;
    logic [12:0] total_count;
    logic [14:0] total_reads;
    logic last_coord;

    assign last_ch     = layer_r ? 5'd31 : 5'd15;
    assign last_pr     = layer_r ? 4'd7  : 4'd15;
    assign last_pc     = layer_r ? 4'd7  : 4'd15;
    assign total_count = layer_r ? 13'd2048 : 13'd4096;
    assign total_reads = layer_r ? 15'd8192 : 15'd16384;

    assign last_coord =
        (ch_r == last_ch) && (pr_r == last_pr) && (pc_r == last_pc);

    shared_maxpool_address_generator u_agen (
        .layer_is_pool2           (layer_r),
        .channel                  (ch_r),
        .pool_row                 (pr_r),
        .pool_column              (pc_r),
        .window_index             (win_r),
        .input_row                (in_row_w),
        .input_column             (in_col_w),
        .activation_read_address  (act_addr_w),
        .pool_write_address       (pool_addr_w)
    );

    max4_int8 u_max4 (
        .value_a (a_r),
        .value_b (b_r),
        .value_c (c_r),
        .value_d (d_r),
        .maximum (max_w)
    );

    assign current_channel        = ch_r;
    assign current_pool_row       = pr_r;
    assign current_pool_column    = pc_r;
    assign current_window_index   = win_r;
    assign output_count           = count_r;
    assign read_count             = reads_r;
    assign dbg_layer_is_pool2     = layer_r;
    assign activation_read_address = act_addr_w;
    assign output_write_address   = pool_addr_w;
    assign output_write_data      = max_w;
    assign value_a = a_r;
    assign value_b = b_r;
    assign value_c = c_r;
    assign value_d = d_r;
    assign maximum_value = max_w;

    assign activation_read_enable = (state == ST_ISSUE);
    assign output_write_enable    = (state == ST_WRITE);

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= ST_IDLE;
            layer_r <= 1'b0;
            ch_r <= 5'd0;
            pr_r <= 4'd0;
            pc_r <= 4'd0;
            win_r <= 2'd0;
            count_r <= 13'd0;
            reads_r <= 15'd0;
            a_r <= 8'sd0;
            b_r <= 8'sd0;
            c_r <= 8'sd0;
            d_r <= 8'sd0;
            busy <= 1'b0;
            done <= 1'b0;
        end else begin
            state <= state_n;
            done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        layer_r <= layer_is_pool2;
                        ch_r <= 5'd0;
                        pr_r <= 4'd0;
                        pc_r <= 4'd0;
                        win_r <= 2'd0;
                        count_r <= 13'd0;
                        reads_r <= 15'd0;
                    end
                end
                ST_ISSUE: begin
                    reads_r <= reads_r + 15'd1;
                end
                ST_CAPTURE: begin
                    unique case (win_r)
                        2'd0: a_r <= activation_read_data;
                        2'd1: b_r <= activation_read_data;
                        2'd2: c_r <= activation_read_data;
                        default: d_r <= activation_read_data;
                    endcase
                    if (win_r != 2'd3)
                        win_r <= win_r + 2'd1;
                end
                ST_WRITE: begin
                    count_r <= count_r + 13'd1;
                end
                ST_ADVANCE: begin
                    win_r <= 2'd0;
                    if (!last_coord) begin
                        if (pc_r == last_pc) begin
                            pc_r <= 4'd0;
                            if (pr_r == last_pr) begin
                                pr_r <= 4'd0;
                                ch_r <= ch_r + 5'd1;
                            end else begin
                                pr_r <= pr_r + 4'd1;
                            end
                        end else begin
                            pc_r <= pc_r + 4'd1;
                        end
                    end
                end
                ST_DONE: begin
                    done <= 1'b1;
                    busy <= 1'b0;
                end
                default: ;
            endcase
        end
    end

    always_comb begin
        state_n = state;
        unique case (state)
            ST_IDLE: if (start) state_n = ST_ISSUE;
            ST_ISSUE: state_n = ST_WAIT;
            ST_WAIT: state_n = ST_CAPTURE;
            ST_CAPTURE: state_n = (win_r == 2'd3) ? ST_COMPARE : ST_ISSUE;
            ST_COMPARE: state_n = ST_WRITE;
            ST_WRITE: state_n = ST_ADVANCE;
            ST_ADVANCE: state_n = last_coord ? ST_DONE : ST_ISSUE;
            ST_DONE: state_n = ST_IDLE;
            default: state_n = ST_IDLE;
        endcase
    end

    // synopsys translate_off
    always_ff @(posedge clk) begin
        if (!rst) begin
            if (busy && (layer_is_pool2 !== layer_r))
                $error("layer_is_pool2 changed while shared_maxpool busy");
            if (busy && start)
                $error("shared_maxpool start while busy");
            if (!layer_r && activation_read_enable &&
                ({1'b0, activation_read_address} >= 15'd16384))
                $error("Pool1 read OOB %0d", activation_read_address);
            if (layer_r && activation_read_enable &&
                (activation_read_address >= 14'd8192))
                $error("Pool2 read OOB %0d", activation_read_address);
            if (!layer_r && output_write_enable &&
                ({1'b0, output_write_address} >= 13'd4096))
                $error("Pool1 write OOB");
            if (layer_r && output_write_enable &&
                ({1'b0, output_write_address} >= 13'd2048))
                $error("Pool2 write OOB");
            if (done && (count_r != total_count))
                $error("done with bad count %0d", count_r);
            if (done && (reads_r != total_reads))
                $error("done with bad read_count %0d", reads_r);
            if (done &&
                ((ch_r != last_ch) || (pr_r != last_pr) || (pc_r != last_pc)))
                $error("done with wrong final coordinate");
        end
    end
    // synopsys translate_on

    /* verilator lint_off UNUSED */
    wire unused_coords = |{in_row_w, in_col_w};
    /* verilator lint_on UNUSED */

endmodule
