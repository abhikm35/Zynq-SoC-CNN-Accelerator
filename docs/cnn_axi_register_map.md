# CNN Accelerator AXI4-Lite Register Map (`cnn_axi_ctrl`)

Software-visible control/status interface for the timing-closed CNN accelerator
(`cnn_accelerator_shared_compute_top` / synth wrapper).

Generated IP path: `ip_repo/cnn_axi_ctrl_1_0/`

HDL:

- Top: `ip_repo/cnn_axi_ctrl_1_0/hdl/cnn_axi_ctrl.v`
- Slave: `ip_repo/cnn_axi_ctrl_1_0/hdl/cnn_axi_ctrl_slave_lite_v1_0_S00_AXI.v`

Base address is assigned later by Vivado Address Editor. This document uses
offsets only.

## Assumptions

- AXI `s00_axi_aclk` and CNN `clk` are the **same PL clock domain** (~83 MHz /
  12 ns provisional constraint) until the Zynq block design proves otherwise.
- No CDC logic is present in this peripheral.
- AXI reset `s00_axi_aresetn` is **active-low**.
- CNN core reset `rst` is **active-high**. At BD integration use
  `cnn_rst = ~s00_axi_aresetn` (or an explicit reset block)—do not invert
  inside the CNN core.
- Activation RAM / DMA / Ethernet are **out of scope** for this interface.

## Register map

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| `0x00` | CONTROL | W (R returns 0) | Bit 0 = START command |
| `0x04` | STATUS | RO | Bit 0 = BUSY, bit 1 = DONE (sticky) |
| `0x08` | PREDICTED_CLASS | RO | Zero-extended 3-bit class (0..4) |
| `0x0C` | MAX_LOGIT | RO | Signed INT32 maximum logit |
| `0x10` | LOGIT_0 | RO | Signed INT32 |
| `0x14` | LOGIT_1 | RO | Signed INT32 |
| `0x18` | LOGIT_2 | RO | Signed INT32 |
| `0x1C` | LOGIT_3 | RO | Signed INT32 |
| `0x20` | LOGIT_4 | RO | Signed INT32 |
| `0x24` | CYCLE_COUNT_LOW | RO | `cycle_count[31:0]` |
| `0x28` | CYCLE_COUNT_HIGH | RO | `cycle_count[63:32]` |
| `0x2C` | RESERVED | RO | Reads as 0; writes ignored |

## CONTROL (`0x00`)

- Software writes `1` to bit 0 to request inference start.
- **START is not a persistent configuration bit.** Reading CONTROL always
  returns `0`.
- A valid AXI write with `WSTRB[0]=1` and `WDATA[0]=1` generates
  **`cnn_start` for exactly one clock cycle**, then automatically clears.
- If `cnn_busy=1`, a new START is **ignored** (no restart, no pulse).
- Writes that do not assert byte-lane 0 strobe do not trigger START.

## STATUS (`0x04`)

| Bit | Name | Meaning |
|-----|------|---------|
| 0 | BUSY | Live `cnn_busy` |
| 1 | DONE | Software-visible sticky completion |
| 31:2 | — | 0 |

### Sticky DONE

CNN top `done` is a **one-cycle pulse** (`cnn_top_controller`). Software may
poll slowly, so the AXI layer latches:

1. Reset → sticky DONE = 0  
2. Accepted START → sticky DONE = 0  
3. `cnn_done=1` → sticky DONE = 1 until next accepted START or reset  

STATUS[1] reflects sticky DONE, not the raw pulse.

## Result registers

CNN interface (from `cnn_accelerator_shared_compute_top`):

| Signal | Width | Signedness |
|--------|-------|------------|
| `predicted_class` | 3 | unsigned |
| `maximum_logit` | 32 (`LOGIT_WIDTH`) | signed |
| `logit_0`..`logit_4` | 32 | signed |
| `cycle_count` | 64 | unsigned |
| `busy` | 1 | level while inference runs |
| `done` | 1 | one-cycle pulse |
| `start` | 1 | one-cycle accept pulse |
| `clk` / `rst` | 1 | rst active-high |

Signed logits are **sign-extended** to 32-bit AXI `RDATA` (identity when
width is already 32). Predicted class is **zero-extended**.

## CNN-side ports on the IP

```text
output cnn_start
input  cnn_busy
input  cnn_done
input  [2:0] cnn_predicted_class
input  signed [LOGIT_WIDTH-1:0] cnn_maximum_logit
input  signed [LOGIT_WIDTH-1:0] cnn_logit_0 .. cnn_logit_4
input  [63:0] cnn_cycle_count
```

The CNN RTL is **not** duplicated inside `ip_repo/`. Connect these ports to
the existing accelerator at block-design / wrapper time.

## C definitions

See `software/driver/cnn_axi_regs.h`.

## Vivado IP Packager steps (Windows, manual)

After pulling these HDL changes:

1. Open `ip_repo/edit_cnn_axi_ctrl_v1_0.xpr` **or** Tools → Create and Package
   New IP → Package a specified directory → `ip_repo/cnn_axi_ctrl_1_0`.
2. **Refresh** / re-add HDL sources if needed.
3. Review **Ports and Interfaces**: confirm new `cnn_*` ports appear as
   discrete ports (not part of AXI).
4. Re-package / update the IP catalog.
5. Only then add the IP to the main CNN Vivado project and create the Zynq BD.

Do not hand-edit `component.xml` unless the packager requires it.
