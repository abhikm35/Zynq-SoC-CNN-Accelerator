# Block Design Module Reference — CNN wrapper

## Why `cnn_accelerator_synth_wrapper` is “incompatible”

With **Add Module → Hide incompatible modules** checked, Vivado hides
`cnn_accelerator_synth_wrapper` mainly because that module declares many
**string file-path parameters** (weight/bias/mult/shift `.mem` paths). IP
Integrator Module Reference cannot map string generics cleanly.

Secondary factors:

- Parameter-dependent port width: `signed [LOGIT_WIDTH-1:0]`
- SystemVerilog top-level Module Reference is less reliable than plain Verilog

## Why the BD wrapper is now `.v`

An earlier SystemVerilog shell (`cnn_accelerator_bd_wrapper.sv`) still did not
appear under Add Module with hide-incompatible enabled on Windows Vivado.

The active BD-facing top is therefore **plain Verilog-2001**:

```text
rtl/cnn_accelerator_bd_wrapper.v
```

Use this module name in the Block Design:

```text
cnn_accelerator_bd_wrapper
```

Hierarchy:

```text
Block Design
  └── cnn_accelerator_bd_wrapper.v   ← Add Module (compatible)
        └── cnn_accelerator_synth_wrapper
              └── cnn_accelerator_shared_compute_top
```

The BD wrapper has **no string parameters** on its boundary. Memory paths stay
inside the synth wrapper defaults.

Do **not** Add Module `cnn_accelerator_synth_wrapper` directly.

The old `.sv` BD wrapper is archived (renamed module) at:

```text
rtl/legacy/cnn_accelerator_bd_wrapper.sv.archived
```

Do **not** add that file to Design Sources.

## External ports (BD-facing)

| Port | Dir | Width | Notes |
|------|-----|-------|-------|
| `clk` | in | 1 | ~83.3 MHz PL clock |
| `rst` | in | 1 | **Active-high** (CNN polarity) |
| `start` | in | 1 | From `cnn_axi_ctrl.cnn_start` |
| `busy` | out | 1 | To `cnn_axi_ctrl.cnn_busy` |
| `done` | out | 1 | To `cnn_axi_ctrl.cnn_done` (raw pulse) |
| `predicted_class` | out | [2:0] | Unsigned |
| `maximum_logit` | out | signed [31:0] | |
| `logit_0`..`logit_4` | out | signed [31:0] | |
| `cycle_count` | out | [63:0] | Split in AXI as LOW/HIGH |
| `input_write_enable` | in | 1 | Act RAM A load (future PS path) |
| `input_write_address` | in | [11:0] | |
| `input_write_data` | in | signed [7:0] | |

No AXI inside this wrapper. Sticky DONE / START pulse live in `cnn_axi_ctrl`.

## Expected Block Design wiring

```text
cnn_axi_ctrl.cnn_start              -> cnn_accelerator_bd_wrapper.start
cnn_accelerator_bd_wrapper.busy     -> cnn_axi_ctrl.cnn_busy
cnn_accelerator_bd_wrapper.done     -> cnn_axi_ctrl.cnn_done
cnn_accelerator_bd_wrapper.predicted_class -> cnn_axi_ctrl.cnn_predicted_class
cnn_accelerator_bd_wrapper.maximum_logit   -> cnn_axi_ctrl.cnn_maximum_logit
cnn_accelerator_bd_wrapper.logit_0..4      -> cnn_axi_ctrl.cnn_logit_0..4
cnn_accelerator_bd_wrapper.cycle_count     -> cnn_axi_ctrl.cnn_cycle_count
```

PL clock → `clk`. CNN `rst` is **active-high**; if the PS/proc_sys_reset net is
active-low, invert once in the BD.

Leave `input_write_*` unconnected or tied off until the Activation RAM milestone.

## Windows steps after `git pull`

1. `git pull` on Windows.
2. Open the existing main Vivado project (or recreate from repo scripts if sources are stale).
3. Confirm **`cnn_accelerator_bd_wrapper.v`** is under Design Sources
   (`scripts/vivado_sources.tcl` lists it).
4. Confirm the archived
   `rtl/legacy/cnn_accelerator_bd_wrapper.sv.archived`
   is **not** in Design Sources (no duplicate `cnn_accelerator_bd_wrapper`).
5. **Update Compile Order** (Sources → Compile Order → Update).
6. Open `zynq_cnn_system` (or your Block Design).
7. Right-click empty canvas → **Add Module…**
8. Leave **Hide incompatible modules** **CHECKED**.
9. Verify **`cnn_accelerator_bd_wrapper`** appears.
10. Add it.
11. Do **not** add `cnn_accelerator_synth_wrapper` directly.
12. Connect `cnn_axi_ctrl` ↔ CNN wrapper as in the wiring table above.

## Related docs

- AXI register map: `docs/cnn_axi_register_map.md`
- General Windows flow: `docs/windows_vivado_workflow.md`
