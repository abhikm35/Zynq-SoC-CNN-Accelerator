# CNN Accelerator AXI4-Lite Register Map (`cnn_axi_ctrl`)

Software-visible control/status interface for the timing-closed CNN accelerator
(`cnn_accelerator_shared_compute_top` / synth / BD wrapper).

Generated IP path: `ip_repo/cnn_axi_ctrl_1_0/`

HDL:

- Top: `ip_repo/cnn_axi_ctrl_1_0/hdl/cnn_axi_ctrl.v`
- Slave: `ip_repo/cnn_axi_ctrl_1_0/hdl/cnn_axi_ctrl_slave_lite_v1_0_S00_AXI.v`

Base address is assigned later by Vivado Address Editor. This document uses
offsets only.

AXI window: **6-bit byte address**, **32-bit data** → offsets `0x00`..`0x3C`
(64 bytes). Word select = `addr[5:2]`.

## Assumptions

- AXI `s00_axi_aclk` and CNN `clk` are the **same PL clock domain** (~83 MHz /
  12 ns provisional constraint) until the Zynq block design proves otherwise.
- No CDC logic is present in this peripheral.
- AXI reset `s00_axi_aresetn` is **active-low**.
- CNN core reset `rst` is **active-high**. At BD integration use
  `cnn_rst = ~s00_axi_aresetn` (or an explicit reset block)—do not invert
  inside the CNN core.
- AXI BRAM Controller / DMA / Ethernet are **out of scope**. Input loading for
  first bring-up uses this AXI4-Lite INPUT_* register path into the existing
  CNN `input_write_*` ports.

## Register map

| Offset | Sel | Name | Access | Description |
|--------|-----|------|--------|-------------|
| `0x00` | 0 | CONTROL | W (R returns 0) | Bit 0 = START command |
| `0x04` | 1 | STATUS | RO | Bit 0 = BUSY, bit 1 = DONE (sticky) |
| `0x08` | 2 | PREDICTED_CLASS | RO | Zero-extended 3-bit class (0..4) |
| `0x0C` | 3 | MAX_LOGIT | RO | Signed INT32 maximum logit |
| `0x10` | 4 | LOGIT_0 | RO | Signed INT32 |
| `0x14` | 5 | LOGIT_1 | RO | Signed INT32 |
| `0x18` | 6 | LOGIT_2 | RO | Signed INT32 |
| `0x1C` | 7 | LOGIT_3 | RO | Signed INT32 |
| `0x20` | 8 | LOGIT_4 | RO | Signed INT32 |
| `0x24` | 9 | CYCLE_COUNT_LOW | RO | `cycle_count[31:0]` |
| `0x28` | A | CYCLE_COUNT_HIGH | RO | `cycle_count[63:32]` |
| `0x2C` | B | RESERVED | RO | Reads as 0; writes ignored |
| `0x30` | C | INPUT_ADDRESS | R/W | Holding `[11:0]` RAM A write address |
| `0x34` | D | INPUT_DATA | R/W | Holding `[7:0]` raw INT8 bits |
| `0x38` | E | INPUT_COMMAND | W (R returns 0) | Bit 0 = WRITE pulse command |

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

## INPUT_ADDRESS (`0x30`)

- Software-writable holding register; only `[11:0]` are stored.
- Valid CNN input addresses: **0 .. 3071** (`0x000` .. `0xBFF`).
- Continuously drives `cnn_input_write_address[11:0]`.
- `WSTRB[0]` updates bits `[7:0]`; `WSTRB[1]` updates bits `[11:8]`.
- Readable for debug.

## INPUT_DATA (`0x34`)

- Software-writable holding register; only `[7:0]` are stored (raw INT8 bits).
- Example: write `0x000000FE` → `cnn_input_write_data = 8'hFE` (CNN sees −2).
- No numeric conversion in the AXI peripheral.
- Requires `WSTRB[0]=1` to update.
- Readable for debug.

## INPUT_COMMAND (`0x38`)

- Bit 0 = **WRITE** command (not sticky).
- Sequence:

```text
write INPUT_ADDRESS
write INPUT_DATA
write INPUT_COMMAND = 1
```

- Accepted WRITE produces **exactly one** cycle of `cnn_input_write_enable=1`
  with the currently held address/data on the CNN ports, then WE returns to 0.
- Ignored when `cnn_busy=1`.
- Ignored when held address `> 3071` (no wrap; no write).
- Holding address/data registers are **not** cleared by an ignored WRITE.
- Reading INPUT_COMMAND always returns `0`.
- Independent of CONTROL.START (load never starts inference; START never writes RAM).

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
output cnn_input_write_enable
output [11:0] cnn_input_write_address
output [7:0]  cnn_input_write_data
```

Connect to `cnn_accelerator_bd_wrapper` in the Block Design. The CNN RTL is
**not** duplicated inside `ip_repo/`.

Input tensor layout (unchanged): CHW `3×32×32` = 3072 INT8

```text
0..1023     R
1024..2047  G
2048..3071  B
```

## C definitions / helpers

- Offsets: `software/driver/cnn_axi_regs.h`
- Loader helpers: `software/driver/cnn_axi_loader.h`
- Golden image: `software/driver/test_image_int8.h` (from
  `vectors/end_to_end/input_image.mem`)

## First software inference sequence

1. Confirm STATUS.BUSY = 0  
2. `cnn_load_input(base, cnn_test_image, 3072)`  
3. Write CONTROL.START  
4. Poll STATUS.DONE  
5. Read logits / predicted class / cycle count  
6. Compare against INT8 Python / RTL golden  

## Vivado IP Packager steps (Windows, manual)

After pulling these HDL changes:

1. Open `ip_repo/edit_cnn_axi_ctrl_v1_0.xpr` **or** Tools → Create and Package
   New IP → Package a specified directory → `ip_repo/cnn_axi_ctrl_1_0`.
2. **Refresh** / re-add HDL sources if needed.
3. Review **Ports and Interfaces**: confirm new `cnn_input_write_*` ports appear
   as discrete ports (not part of AXI).
4. Re-package / update the IP catalog.
5. In `zynq_cnn_system`, refresh `cnn_axi_ctrl_0` if marked stale.
6. Connect:

```text
cnn_axi_ctrl_0.cnn_input_write_enable  -> bd_wrapper.input_write_enable
cnn_axi_ctrl_0.cnn_input_write_address -> bd_wrapper.input_write_address
cnn_axi_ctrl_0.cnn_input_write_data    -> bd_wrapper.input_write_data
```

7. Validate Design (unconnected `input_write_*` warnings should clear).

Do not hand-edit `component.xml` unless the packager requires it.
