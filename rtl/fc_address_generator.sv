// fc_address_generator.sv
// Fully connected address equations for Linear 32 -> 5.
//
//   gap_read_address    = input_index
//   fc_weight_address   = class_index * 32 + input_index
//   fc_bias_address     = class_index
//   logit_write_address = class_index
//
// Combinational. No floating-point.

`timescale 1ns / 1ps

module fc_address_generator (
    input  logic [2:0]  class_index,   // 0 .. 4
    input  logic [4:0]  input_index,   // 0 .. 31

    output logic [4:0]  gap_read_address,     // 0 .. 31
    output logic [7:0]  fc_weight_address,    // 0 .. 159
    output logic [2:0]  fc_bias_address,      // 0 .. 4
    output logic [2:0]  logit_write_address   // 0 .. 4
);

    assign gap_read_address    = input_index;
    // class_index * 32 == class_index << 5
    assign fc_weight_address   = {class_index, input_index};
    assign fc_bias_address     = class_index;
    assign logit_write_address = class_index;

endmodule
