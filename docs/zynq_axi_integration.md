# Zynq AXI / Block Design integration paths

## Standalone synthesis path (timing-closed)

```text
cnn_accelerator_synth_wrapper.sv
        |
        v
timing-closed CNN (~83 MHz on xc7z010)
```

## Zynq Block Design path

```text
zynq_cnn_system.bd
        |
        v
cnn_accelerator_bd_wrapper.v      ← Add Module (plain Verilog top)
        |
        v
cnn_accelerator_synth_wrapper.sv
        |
        v
CNN
```

## AXI control + packed INPUT_WRITE image loader

```text
                         Zynq ARM
                            |
                            | AXI4-Lite
                            v
                    +---------------+
                    | cnn_axi_ctrl  |
                    |               |
                    | CONTROL       |------> CNN start
                    | STATUS        |<------ busy/done
                    | RESULTS       |<------ logits/class
                    |               |
                    | INPUT_WRITE   |------+
                    +---------------+      |
                                           |
                      +--------------------+
                      |
                      v
              input_write_enable
              input_write_address
              input_write_data
                      |
                      v
                 Activation RAM A
                      |
                      v
                     CNN
```

This AXI-Lite path is intentionally simple for first board bring-up.
A future optimization may use AXI BRAM Controller or DMA — not in this milestone.

Register details: `docs/cnn_axi_register_map.md`.

### Control / result wiring

```text
cnn_axi_ctrl_0.cnn_start              -> bd_wrapper.start
bd_wrapper.busy                       -> cnn_axi_ctrl_0.cnn_busy
bd_wrapper.done                       -> cnn_axi_ctrl_0.cnn_done
bd_wrapper.predicted_class            -> cnn_axi_ctrl_0.cnn_predicted_class
bd_wrapper.maximum_logit              -> cnn_axi_ctrl_0.cnn_maximum_logit
bd_wrapper.logit_0..4                 -> cnn_axi_ctrl_0.cnn_logit_0..4
bd_wrapper.cycle_count                -> cnn_axi_ctrl_0.cnn_cycle_count
```

### Input-load wiring

```text
cnn_axi_ctrl_0.cnn_input_write_enable
    -> cnn_accelerator_bd_wrapper.input_write_enable

cnn_axi_ctrl_0.cnn_input_write_address[11:0]
    -> cnn_accelerator_bd_wrapper.input_write_address[11:0]

cnn_axi_ctrl_0.cnn_input_write_data[7:0]
    -> cnn_accelerator_bd_wrapper.input_write_data[7:0]
```

CNN `rst` is **active-high** (invert PS active-low reset once in the BD).

### INPUT_WRITE packing (`0x2C`)

```text
packed = (address & 0xFFF) | ((uint8_t)value << 12)
Xil_Out32(CNN_BASE + 0x2C, packed)
```

One write → one-cycle `input_write_enable` (ignored if busy or address ≥ 3072).

## First-board software flow

```text
known INT8 input[3072]   // cnn_test_image / input_image.mem

for i = 0..3071:
    packed = (i & 0xFFF) | ((uint8_t)input[i] << 12)
    Xil_Out32(CNN_BASE + CNN_INPUT_WRITE_OFFSET, packed)

write START
poll DONE
read logit_0..4, maximum_logit, predicted_class, cycle_count
```

Target: INT8 Python == RTL == Arty Z7 FPGA.

## Windows: refresh IP + connect ports

1. Commit/push Cursor changes; on Windows `git pull`.
2. Open editable `cnn_axi_ctrl` packaged-IP project.
3. Refresh source/file groups if stale.
4. Confirm ports: `cnn_input_write_enable`, `_address[11:0]`, `_data[7:0]`.
5. Review and Package / Re-Package the IP.
6. Main CNN project: refresh/upgrade `cnn_axi_ctrl_0`.
7. Open `zynq_cnn_system.bd`.
8. Connect the three `cnn_input_write_*` pins to the BD wrapper (table above).
9. Save → Validate Design (unconnected `input_write_*` warnings should clear).

## Next milestones

**If board shows all logits = 0 but cycle_count exact:** see
`docs/fpga_zero_logits_debug.md` and run `scripts/apply_bd_mem_generics.tcl`
before rebuilding the bitstream (parameter `$readmemh` paths).

Then: Assign AXI address → generate BD wrapper → synth/impl @ ~83.333 MHz →
bitstream → XSA → Vitis load golden image and compare.
