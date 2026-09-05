# CNN Accelerator AXI4-Lite Register Map (`cnn_axi_ctrl`)

Software-visible control/status/input-load interface for the timing-closed CNN
accelerator (`cnn_accelerator_shared_compute_top` / synth / BD wrapper).

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
  12 ns provisional constraint).
- No CDC logic is present in this peripheral.
- AXI reset `s00_axi_aresetn` is **active-low**.
- CNN core reset `rst` is **active-high**. At BD integration use
  `cnn_rst = ~s00_axi_aresetn`.
- AXI BRAM Controller / DMA / Ethernet are **out of scope**. First bring-up
  loads the 3072 INT8 tensor via packed **INPUT_WRITE** at `0x2C`.

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
| `0x2C` | B | INPUT_WRITE | WO (R returns 0) | Packed address+data write command |

## CONTROL (`0x00`)

- Write bit 0 = 1 → one-cycle `cnn_start` (ignored if `cnn_busy`).
- Reading CONTROL always returns `0`. Requires `WSTRB[0]`.

## STATUS (`0x04`)

| Bit | Name | Meaning |
|-----|------|---------|
| 0 | BUSY | Live `cnn_busy` |
| 1 | DONE | Sticky completion (cleared on accepted START) |

## INPUT_WRITE (`0x2C`)

Packed write-only command (one AXI write = one Activation RAM A store):

```text
31                              20 19             12 11              0
+--------------------------------+-----------------+------------------+
|            reserved            |   INT8 DATA     | INPUT ADDRESS    |
+--------------------------------+-----------------+------------------+
```

| Field | Bits | Meaning |
|-------|------|---------|
| address | `[11:0]` | Activation RAM A write address |
| data | `[19:12]` | Raw INT8 bits (no conversion) |
| reserved | `[31:20]` | Ignored |

### Acceptance rules

An INPUT_WRITE is accepted only when **all** of:

1. AXI write commits to selector `4'hB` (`0x2C`)
2. `WSTRB[2:0] == 3'b111` (bits through 19 need lanes 0..2)
3. `cnn_busy == 0`
4. `WDATA[11:0] < 3072`

On accept (registered, same cycle):

- `cnn_input_write_address <= WDATA[11:0]`
- `cnn_input_write_data <= WDATA[19:12]`
- `cnn_input_write_enable <= 1` for **exactly one** clock, then 0

Ignored writes do not pulse enable. Reading `0x2C` always returns `0`.

Example: address `137`, data `0xF4` → write `(0xF4 << 12) | 137` to `BASE+0x2C`.

## CNN-side ports

```text
output cnn_start
output cnn_input_write_enable
output [11:0] cnn_input_write_address
output [7:0]  cnn_input_write_data

input  cnn_busy
input  cnn_done
input  [2:0] cnn_predicted_class
input  signed [LOGIT_WIDTH-1:0] cnn_maximum_logit / logit_0..4
input  [63:0] cnn_cycle_count
```

Connect to `cnn_accelerator_bd_wrapper` in the Block Design.

Input layout (unchanged CHW): R `0..1023`, G `1024..2047`, B `2048..3071`.

## C definitions / helpers

- `software/driver/cnn_axi_regs.h`
- `software/driver/cnn_axi_loader.h` (`cnn_write_input`, `cnn_load_input`)
- `software/driver/test_image_int8.h` (golden from `vectors/end_to_end/input_image.mem`)

## First software inference sequence

1. Confirm STATUS.BUSY = 0  
2. For `i = 0..3071`: `cnn_write_input(base, i, image[i])`  
3. Write CONTROL.START  
4. Poll STATUS.DONE  
5. Read logits / class / cycle count  
6. Compare vs INT8 Python / RTL  

## Vivado IP Packager steps (Windows)

1. `git pull`
2. Open/edit packaged `cnn_axi_ctrl` IP project.
3. Refresh HDL; confirm `cnn_input_write_*` ports.
4. Re-package IP (do not hand-edit `component.xml` unless required).
5. In `zynq_cnn_system`, refresh `cnn_axi_ctrl_0`.
6. Connect:

```text
cnn_axi_ctrl_0.cnn_input_write_enable  -> bd_wrapper.input_write_enable
cnn_axi_ctrl_0.cnn_input_write_address -> bd_wrapper.input_write_address
cnn_axi_ctrl_0.cnn_input_write_data    -> bd_wrapper.input_write_data
```

7. Validate Design.
