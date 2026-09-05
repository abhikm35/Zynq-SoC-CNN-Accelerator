# Block Design Module Reference — CNN wrapper

## Why `cnn_accelerator_synth_wrapper` is “incompatible”

With **Add Module → Hide incompatible modules** checked, Vivado hides
`cnn_accelerator_synth_wrapper` mainly because that module declares many
**string file-path parameters** (weight/bias/mult/shift `.mem` paths). IP
Integrator Module Reference cannot map string generics cleanly.

Secondary factors:

- Parameter-dependent port width: `signed [LOGIT_WIDTH-1:0]`
- Not because of unpacked logit arrays (that wrapper never had those)

## Use this instead in the Block Design

```text
cnn_accelerator_bd_wrapper
```

File: `rtl/cnn_accelerator_bd_wrapper.sv`

Hierarchy:

```text
Block Design
  └── cnn_accelerator_bd_wrapper     ← Add Module (compatible)
        └── cnn_accelerator_synth_wrapper
              └── cnn_accelerator_shared_compute_top
```

The BD wrapper has **no string parameters** on its boundary. Memory paths stay
inside the synth wrapper defaults.

Do **not** Add Module `cnn_accelerator_synth_wrapper` directly.

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

## Windows steps after `git pull`

1. Open the main Vivado project (recreate if sources are stale).
2. Confirm `cnn_accelerator_bd_wrapper.sv` is under Design Sources
   (`scripts/vivado_sources.tcl` lists it).
3. Open the Block Design (e.g. `zynq_cnn_system`).
4. Right-click canvas → **Add Module…**
5. Keep **Hide incompatible modules** **CHECKED**.
6. Select **`cnn_accelerator_bd_wrapper`** (not the synth wrapper).
7. Connect:
   - `cnn_axi_ctrl.cnn_start` → `bd_wrapper.start`
   - `bd_wrapper.busy/done/predicted_class/maximum_logit/logit_*/cycle_count`
     → matching `cnn_axi_ctrl` CNN-side inputs
   - PL clock/reset → `clk` / `rst` (remember CNN `rst` is active-high;
     if PS reset is active-low, invert once in the BD)
8. Leave `input_write_*` unconnected or tied off until the Activation RAM
   milestone (or drive from a test source).

## Related docs

- AXI register map: `docs/cnn_axi_register_map.md`
- General Windows flow: `docs/windows_vivado_workflow.md`
