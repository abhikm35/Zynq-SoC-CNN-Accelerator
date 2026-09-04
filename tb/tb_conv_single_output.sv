// tb_conv_single_output.sv
`timescale 1ns / 1ps

module tb_conv_single_output (
    input logic clk
);
    logic rst;
    logic start;
    logic signed [31:0] bias;
    logic signed [31:0] multiplier;
    logic [5:0] shift;
    logic signed [7:0] output_zero_point;
    logic signed [7:0] activation;
    logic signed [7:0] weight;

    logic [4:0] mac_sel;
    logic signed [15:0] product;
    logic signed [15:0] last_product;
    logic signed [31:0] accumulator;
    logic signed [31:0] final_accumulator;
    logic signed [7:0] requantized_output;
    logic signed [7:0] relu_output;
    logic [4:0] mac_count;
    logic busy;
    logic done;

    logic signed [7:0]  act_mem  [0:26];
    logic signed [7:0]  wgt_mem  [0:26];
    logic signed [15:0] prod_mem [0:26];
    logic signed [31:0] acc_mem  [0:26];
    logic signed [31:0] exp_acc;
    logic signed [7:0]  exp_rq;
    logic signed [7:0]  exp_relu;

    integer local_c;
    integer phase;
    integer mac_i;
    integer errors;
    integer fd;
    integer idx;
    integer scan_code;
    reg [31:0] hv;

    conv_single_output dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .bias(bias),
        .multiplier(multiplier),
        .shift(shift),
        .output_zero_point(output_zero_point),
        .activation(activation),
        .weight(weight),
        .mac_sel(mac_sel),
        .product(product),
        .last_product(last_product),
        .accumulator(accumulator),
        .final_accumulator(final_accumulator),
        .requantized_output(requantized_output),
        .relu_output(relu_output),
        .mac_count(mac_count),
        .busy(busy),
        .done(done)
    );

    always_comb begin
        activation = act_mem[mac_sel];
        weight     = wgt_mem[mac_sel];
    end

    task automatic load_center;
        begin
            fd = $fopen("vectors/conv1_single_output/center_inputs.mem", "r");
            if (fd == 0) begin $display("FAIL open center inputs"); $fatal(1); end
            for (idx = 0; idx < 27; idx = idx + 1) begin
                scan_code = $fscanf(fd, "%h", hv); act_mem[idx] = hv[7:0];
            end
            $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/center_weights.mem", "r");
            for (idx = 0; idx < 27; idx = idx + 1) begin
                scan_code = $fscanf(fd, "%h", hv); wgt_mem[idx] = hv[7:0];
            end
            $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/center_products.mem", "r");
            for (idx = 0; idx < 27; idx = idx + 1) begin
                scan_code = $fscanf(fd, "%h", hv); prod_mem[idx] = hv[15:0];
            end
            $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/center_acc_trace.mem", "r");
            for (idx = 0; idx < 27; idx = idx + 1) begin
                scan_code = $fscanf(fd, "%h", hv); acc_mem[idx] = hv;
            end
            $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/center_bias.mem", "r");
            scan_code = $fscanf(fd, "%h", hv); bias = hv; $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/center_multiplier.mem", "r");
            scan_code = $fscanf(fd, "%h", hv); multiplier = hv; $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/center_shift.mem", "r");
            scan_code = $fscanf(fd, "%h", hv); shift = hv[5:0]; $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/center_expected_acc.mem", "r");
            scan_code = $fscanf(fd, "%h", hv); exp_acc = hv; $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/center_expected_requantized.mem", "r");
            scan_code = $fscanf(fd, "%h", hv); exp_rq = hv[7:0]; $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/center_expected_relu.mem", "r");
            scan_code = $fscanf(fd, "%h", hv); exp_relu = hv[7:0]; $fclose(fd);
            output_zero_point = 0;
        end
    endtask

    task automatic load_corner;
        begin
            fd = $fopen("vectors/conv1_single_output/corner_inputs.mem", "r");
            if (fd == 0) $fatal(1);
            for (idx = 0; idx < 27; idx = idx + 1) begin
                scan_code = $fscanf(fd, "%h", hv); act_mem[idx] = hv[7:0];
            end
            $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/corner_weights.mem", "r");
            for (idx = 0; idx < 27; idx = idx + 1) begin
                scan_code = $fscanf(fd, "%h", hv); wgt_mem[idx] = hv[7:0];
            end
            $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/corner_products.mem", "r");
            for (idx = 0; idx < 27; idx = idx + 1) begin
                scan_code = $fscanf(fd, "%h", hv); prod_mem[idx] = hv[15:0];
            end
            $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/corner_acc_trace.mem", "r");
            for (idx = 0; idx < 27; idx = idx + 1) begin
                scan_code = $fscanf(fd, "%h", hv); acc_mem[idx] = hv;
            end
            $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/corner_bias.mem", "r");
            scan_code = $fscanf(fd, "%h", hv); bias = hv; $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/corner_multiplier.mem", "r");
            scan_code = $fscanf(fd, "%h", hv); multiplier = hv; $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/corner_shift.mem", "r");
            scan_code = $fscanf(fd, "%h", hv); shift = hv[5:0]; $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/corner_expected_acc.mem", "r");
            scan_code = $fscanf(fd, "%h", hv); exp_acc = hv; $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/corner_expected_requantized.mem", "r");
            scan_code = $fscanf(fd, "%h", hv); exp_rq = hv[7:0]; $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/corner_expected_relu.mem", "r");
            scan_code = $fscanf(fd, "%h", hv); exp_relu = hv[7:0]; $fclose(fd);
            output_zero_point = 0;
        end
    endtask

    task automatic load_channel3;
        begin
            fd = $fopen("vectors/conv1_single_output/channel3_inputs.mem", "r");
            if (fd == 0) $fatal(1);
            for (idx = 0; idx < 27; idx = idx + 1) begin
                scan_code = $fscanf(fd, "%h", hv); act_mem[idx] = hv[7:0];
            end
            $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/channel3_weights.mem", "r");
            for (idx = 0; idx < 27; idx = idx + 1) begin
                scan_code = $fscanf(fd, "%h", hv); wgt_mem[idx] = hv[7:0];
            end
            $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/channel3_products.mem", "r");
            for (idx = 0; idx < 27; idx = idx + 1) begin
                scan_code = $fscanf(fd, "%h", hv); prod_mem[idx] = hv[15:0];
            end
            $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/channel3_acc_trace.mem", "r");
            for (idx = 0; idx < 27; idx = idx + 1) begin
                scan_code = $fscanf(fd, "%h", hv); acc_mem[idx] = hv;
            end
            $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/channel3_bias.mem", "r");
            scan_code = $fscanf(fd, "%h", hv); bias = hv; $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/channel3_multiplier.mem", "r");
            scan_code = $fscanf(fd, "%h", hv); multiplier = hv; $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/channel3_shift.mem", "r");
            scan_code = $fscanf(fd, "%h", hv); shift = hv[5:0]; $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/channel3_expected_acc.mem", "r");
            scan_code = $fscanf(fd, "%h", hv); exp_acc = hv; $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/channel3_expected_requantized.mem", "r");
            scan_code = $fscanf(fd, "%h", hv); exp_rq = hv[7:0]; $fclose(fd);
            fd = $fopen("vectors/conv1_single_output/channel3_expected_relu.mem", "r");
            scan_code = $fscanf(fd, "%h", hv); exp_relu = hv[7:0]; $fclose(fd);
            output_zero_point = 0;
        end
    endtask

    initial begin
        local_c = 0;
        phase = 0;
        errors = 0;
        rst = 1;
        start = 0;
        load_center();
    end

    always @(negedge clk) begin
        local_c = local_c + 1;
        if (phase < 3) begin
            case (local_c)
                1, 2: begin rst = 1; start = 0; end
                3: begin rst = 0; start = 0; end
                4: begin start = 1; end
                5: begin start = 0; end
                default: ;
            endcase
        end
    end

    always @(posedge clk) begin
        if (phase < 3) begin
            // Sample prior MAC after the following posedge (avoid same-edge race).
            // local_c=5: LOAD_BIAS
            // local_c=6: MAC0
            // local_c=7: MAC1 — check MAC0
            // ...
            // local_c=32: MAC26 — check MAC25
            // local_c=33: REQUANTIZE — check MAC26
            // local_c=34: DONE (done pulse)
            // local_c=35: check outputs (done was registered)
            if (local_c >= 7 && local_c <= 33) begin
                mac_i = local_c - 7;
                if (last_product !== prod_mem[mac_i]) begin
                    $display("FAIL phase%0d MAC %0d product got %0d exp %0d",
                             phase, mac_i, last_product, prod_mem[mac_i]);
                    errors = errors + 1;
                end
                if (accumulator !== acc_mem[mac_i]) begin
                    $display("FAIL phase%0d MAC %0d acc got %0d exp %0d",
                             phase, mac_i, accumulator, acc_mem[mac_i]);
                    errors = errors + 1;
                end
            end
            if (local_c == 35) begin
                if (!done) begin $display("FAIL phase%0d done", phase); errors = errors + 1; end
                if (mac_count !== 5'd27) begin
                    $display("FAIL phase%0d mac_count %0d", phase, mac_count);
                    errors = errors + 1;
                end
                if (final_accumulator !== exp_acc) begin
                    $display("FAIL phase%0d acc %0d exp %0d", phase, final_accumulator, exp_acc);
                    errors = errors + 1;
                end
                if (requantized_output !== exp_rq) begin
                    $display("FAIL phase%0d rq %0d exp %0d", phase, requantized_output, exp_rq);
                    errors = errors + 1;
                end
                if (relu_output !== exp_relu) begin
                    $display("FAIL phase%0d relu %0d exp %0d", phase, relu_output, exp_relu);
                    errors = errors + 1;
                end
                if (errors != 0) begin
                    $display("FAIL tb_conv_single_output phase %0d", phase);
                    $fatal(1);
                end
                $display("PASS phase %0d", phase);
                phase = phase + 1;
                local_c = 0;
                errors = 0;
                if (phase == 1) load_corner();
                else if (phase == 2) load_channel3();
                else begin
                    $display("PASS tb_conv_single_output");
                    $finish;
                end
            end
        end
    end
endmodule
