# MaxPool2 Architecture

MaxPool2 reduces the verified Conv2 / ReLU2 activation map with a
**2 × 2, stride-2** signed INT8 maximum. There is **no** multiply, accumulate,
requantization, or ReLU in this stage.

> Trained model note: Conv2 is **32** channels (`16→32`), so Pool2 is
> **32 × 8 × 8 = 2048** outputs. Earlier design sketches with 16 channels /
> 1024 outputs are obsolete.

---

## Purpose

- Spatial downsampling of Conv2 for global average pooling
- Preserve the strongest activation in each 2 × 2 neighborhood
- Keep the same signed INT8 representation (bit-identical to Python)

## Tensor shapes

| Tensor | Shape | Count | Datatype |
| --- | --- | --- | --- |
| Conv2 input (to pool) | `32 × 16 × 16` | 8192 | signed INT8 |
| Pool2 output | `32 × 8 × 8` | 2048 | signed INT8 |

Channels are pooled **independently**.

## Address equations

```text
conv2_read_address =
    channel * 256
  + input_row * 16
  + input_column

pool2_write_address =
    channel * 64
  + pool_row * 8
  + pool_column
```

## Signed comparison

Reuses `max4_int8` (same as MaxPool1). Comparisons are **signed**.
MaxPool2 does **not** requantize and does **not** apply another ReLU.

## Synchronous memory timing

Same 1-cycle sync ROM latency as MaxPool1:

```text
ISSUE → WAIT → CAPTURE   (×4 window positions)
COMPARE → WRITE → ADVANCE → (next | DONE)
```

## Modules

| Module | Role |
| --- | --- |
| `maxpool2_address_generator` | Combinational addresses |
| `max4_int8` | Signed max-of-four (**reused**) |
| `maxpool2_controller` | FSM + counters (MaxPool1 pattern) |
| `maxpool2_top` | Conv2 ROM + controller + Pool2 RAM |
| `int8_sync_rom` / `int8_sync_ram` | Memory models |

MaxPool1 RTL is left unchanged. Address equations differ (spatial sizes and
channel counts), so MaxPool2 uses dedicated address/controller modules while
reusing the verified signed maximum and memory primitives.

## Counts

```text
Conv2 input values:     32 x 16 x 16 = 8192
Pool2 output values:    32 x  8 x  8 = 2048
Reads per output:       4
Total Conv2 reads:      2048 x 4 = 8192
```

## Regenerate vectors

```bash
python -m tools.export_pool2_vectors
# or: make export-pool2
```

## Run tests

```bash
make test-rtl-maxpool2
make test-rtl-conv1-full   # full regression
```

## Next milestone (not this task)

Implement global average pooling by reading the complete **32 × 8 × 8** Pool2
tensor (done — see `docs/global_average_pooling_architecture.md` /
`make test-rtl-gap`).

