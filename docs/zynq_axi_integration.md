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

AXI4-Lite control remains a **separate** IP:

```text
Zynq PS
   |
   | AXI4-Lite
   v
cnn_axi_ctrl
   |
   | ordinary RTL signals
   v
cnn_accelerator_bd_wrapper
```

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

## Control / result wiring (actual port names)

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

CNN `rst` is **active-high**. If `proc_sys_reset` / AXI reset is active-low,
invert once in the Block Design (`rst = ~peripheral_aresetn`).

Leave `input_write_*` tied off until the AXI BRAM / Activation RAM A milestone.

## Windows: add the Module Reference

1. `git pull`
2. Open the existing Vivado CNN project.
3. Confirm `rtl/cnn_accelerator_bd_wrapper.v` appears in Design Sources.
4. If not: **Add Sources → Add or Create Design Sources → Add Files** and select
   `rtl/cnn_accelerator_bd_wrapper.v`. Refresh / Update Compile Order.
5. Open `zynq_cnn_system` block design.
6. Right-click canvas → **Add Module**.
7. Select **`cnn_accelerator_bd_wrapper`** (the `.v` module).
8. Add it to the block design.
9. Do **not** select `cnn_accelerator_synth_wrapper` directly.

More detail: `docs/cnn_bd_wrapper.md`.

## Next milestones (not in this wrapper)

1. Connect `cnn_axi_ctrl` ↔ `cnn_accelerator_bd_wrapper` and clock/reset.
2. Make Activation RAM A processor-accessible (AXI BRAM Controller path).
