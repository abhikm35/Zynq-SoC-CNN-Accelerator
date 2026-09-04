// signed_argmax5.sv
// Combinational signed argmax over five logits.
//
// Tie-breaking matches numpy.argmax / IntegerTinyCNN.predict:
//   update only when current_logit > maximum_logit (strict greater).
//   Equal maxima keep the lowest class index.
//
// No softmax, no requantization, no ReLU. No floating-point.

`timescale 1ns / 1ps

module signed_argmax5 #(
    parameter int LOGIT_WIDTH = 32
) (
    input  logic signed [LOGIT_WIDTH-1:0] logit_0,
    input  logic signed [LOGIT_WIDTH-1:0] logit_1,
    input  logic signed [LOGIT_WIDTH-1:0] logit_2,
    input  logic signed [LOGIT_WIDTH-1:0] logit_3,
    input  logic signed [LOGIT_WIDTH-1:0] logit_4,

    output logic [2:0]                    predicted_class,
    output logic signed [LOGIT_WIDTH-1:0] maximum_logit
);

    logic signed [LOGIT_WIDTH-1:0] max_w;
    logic [2:0] pred_w;

    always_comb begin
        max_w  = logit_0;
        pred_w = 3'd0;

        if (logit_1 > max_w) begin
            max_w  = logit_1;
            pred_w = 3'd1;
        end
        if (logit_2 > max_w) begin
            max_w  = logit_2;
            pred_w = 3'd2;
        end
        if (logit_3 > max_w) begin
            max_w  = logit_3;
            pred_w = 3'd3;
        end
        if (logit_4 > max_w) begin
            max_w  = logit_4;
            pred_w = 3'd4;
        end
    end

    assign maximum_logit   = max_w;
    assign predicted_class = pred_w;

endmodule
