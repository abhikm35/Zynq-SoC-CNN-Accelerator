# Zynq AXI / Block Design integration paths

## Standalone synthesis path (timing-closed)

```text
cnn_accelerator_synth_wrapper.sv
        |
        v
timing-closed CNN (~83 MHz on xc7z010)
```

Use this top for standalone synth / implement / timing characterization.
Do **not** add this file as a Vivado Module Reference top — Vivado rejects
SystemVerilog as the Module Reference top file.

## Zynq Block Design path

```text
zynq_cnn_system.bd
        |
        v
cnn_accelerator_bd_wrapper.v      ← Add Module (plain Verilog top)
        |
        v
cnn_accelerator_synth_wrapper.sv  ← unchanged timing-closed wrapper
        |
        v
CNN
```

`cnn_accelerator_bd_wrapper.v` contains **no** accelerator computation. It is
only a Verilog-2001 wire shell so IP Integrator Module Reference can accept a
non-SystemVerilog top.

## Current AXI control + input-loader architecture

```text
                   Zynq PS
                     ARM
                      |
                      | AXI4-Lite (M_AXI_GP0 → SmartConnect)
                      v
             +------------------+
             | cnn_axi_ctrl     |
             |                  |
             | START ---------->|------+
             | STATUS <---------|      |
             | RESULTS <--------|------+----> cnn_accelerator_bd_wrapper
             |                  |
             | INPUT_ADDRESS ---|------+
             | INPUT_DATA ------|------|----> input_write_*
             | INPUT_COMMAND ---|------+
             +------------------+
```

Register map details: `docs/cnn_axi_register_map.md`.

### Control / result wiring

```text
cnn_axi_ctrl_0.cnn_start
    -> cnn_accelerator_bd_wrapper.start

cnn_accelerator_bd_wrapper.busy
    -> cnn_axi_ctrl_0.cnn_busy

cnn_accelerator_bd_wrapper.done
    -> cnn_axi_ctrl_0.cnn_done

cnn_accelerator_bd_wrapper.predicted_class
    -> cnn_axi_ctrl_0.cnn_predicted_class

cnn_accelerator_bd_wrapper.maximum_logit
    -> cnn_axi_ctrl_0.cnn_maximum_logit

cnn_accelerator_bd_wrapper.logit_0 .. logit_4
    -> cnn_axi_ctrl_0.cnn_logit_0 .. cnn_logit_4

cnn_accelerator_bd_wrapper.cycle_count
    -> cnn_axi_ctrl_0.cnn_cycle_count
```

### Input-load wiring (clears BD unconnected warnings)

```text
cnn_axi_ctrl_0.cnn_input_write_enable
    -> cnn_accelerator_bd_wrapper.input_write_enable

cnn_axi_ctrl_0.cnn_input_write_address[11:0]
    -> cnn_accelerator_bd_wrapper.input_write_address[11:0]

cnn_axi_ctrl_0.cnn_input_write_data[7:0]
    -> cnn_accelerator_bd_wrapper.input_write_data[7:0]
```

CNN `rst` is **active-high**. If `proc_sys_reset` / AXI reset is active-low,
invert once in the Block Design (`rst = ~peripheral_aresetn`).

## First software inference sequence

1. Load 3072 INT8 values via INPUT_ADDRESS / INPUT_DATA / INPUT_COMMAND  
   (`cnn_load_input` + `cnn_test_image` from `software/driver/`).  
2. Confirm CNN idle (STATUS.BUSY = 0).  
3. Write START (CONTROL bit 0).  
4. Poll DONE (STATUS bit 1 sticky).  
5. Read logits.  
6. Read predicted class.  
7. Read cycle count.  
8. Compare against INT8 Python and RTL golden.

## Why Vivado rejects the synth wrapper as Module Reference

Typical message:

```text
Reference 'cnn_accelerator_synth_wrapper' contains top file
'.../cnn_accelerator_synth_wrapper.sv' of type SystemVerilog.
This type is not allowed as the top file in the reference.
```

Additional reasons the synth wrapper is a poor Module Reference boundary:

- many string `.mem` path parameters
- parameter-dependent port width (`LOGIT_WIDTH`)

## Windows: refresh IP + connect input ports

1. `git pull`
2. Open the existing Vivado CNN project.
3. Open/Edit the packaged `cnn_axi_ctrl` IP (`ip_repo/cnn_axi_ctrl_1_0`).
4. Refresh/reload the changed HDL (`cnn_axi_ctrl.v`, slave lite).
5. Re-package the AXI IP (let the packager update ports; avoid hand-editing
   `component.xml` unless required).
6. Return to `zynq_cnn_system` block design.
7. Refresh `cnn_axi_ctrl_0` if Vivado marks it stale.
8. Confirm new ports appear:

   - `cnn_input_write_enable`
   - `cnn_input_write_address[11:0]`
   - `cnn_input_write_data[7:0]`

9. Connect them to the BD wrapper `input_write_*` ports (table above).
10. Save block design.
11. Validate Design — unconnected `input_write_*` warnings should disappear.

More Module Reference detail: `docs/cnn_bd_wrapper.md`.

## Next milestones (after Validate Design is clean)

1. Assign/check AXI address range.  
2. Generate BD HDL wrapper.  
3. Synth / impl at ~83.333 MHz; confirm timing.  
4. Bitstream → XSA → first Vitis program.  
5. Load golden 3072-byte image, START, compare FPGA vs Python/RTL.  

Do **not** add AXI BRAM Controller / DMA in this bring-up path yet.
