# Global Average Pooling Architecture

GAP reduces each Pool2 channel to one signed INT8 feature used by the
classifier. Hardware matches the Python integer golden model bit for bit.

> Trained model note: Pool2 is **32 × 8 × 8 = 2048**, so GAP produces
> **32** outputs. Earlier design sketches with 16 channels are obsolete.

---

## Purpose

- Spatial collapse of each Pool2 channel (`8 × 8 → 1`)
- Preserve the calibrated GAP / flatten activation scale for Linear `32 → 5`
- Keep signed INT8 throughout (no floating-point in RTL)

## Tensor shapes

| Tensor | Shape | Count | Datatype |
| --- | --- | --- | --- |
| Pool2 input | `32 × 8 × 8` | 2048 | signed INT8 |
| Raw channel average (pool2 scale) | `32` | 32 | signed INT8 |
| GAP / flatten output | `32` | 32 | signed INT8 |

Channels are averaged **independently**.

## Exact Python equation

From `integer_inference.py` / `fixed_point.py` with `zero_point == 0`:

```text
sum[channel] = Σ_{i=0..63} int8(pool2[channel][i])     // INT32 accumulator

averaged[channel] = round_divide_int(sum[channel], 64)
                  = rounding_right_shift(sum, 6)        // ties away from zero
averaged[channel] = saturate_int8(averaged[channel])    // still pool2 scale

gap[channel] = requantize_int32(
                   averaged[channel],
                   gap_multiplier = 1759306569,
                   gap_shift = 29,
                   output_zero_point = 0
               )                                        // INT8 flatten scale
```

### Rounding (ties away from zero)

```text
half = 32
if sum >= 0:
    avg = (sum + 32) >>> 6
else:
    avg = -(((-sum) + 32) >>> 6)
```

Examples:

| sum | result |
| --- | --- |
| 32 | +1 |
| −32 | −1 |
| 31 | 0 |
| −31 | 0 |
| −65 | −1 |

Pool2 zero point is **0**, so no centering is applied.

### Quantization metadata

| Property | Value |
| --- | --- |
| Input datatype | signed INT8 |
| Input scale | pool2 / conv2 scale (~0.1749) |
| Input zero point | 0 |
| Accumulator | signed INT32 (math minimum 14 bits) |
| Division | exact integer `/64` via rounding ASR by 6 |
| Rounding | ties **away from zero** |
| Average saturation | `[-128, 127]` |
| GAP multiplier / shift | shared per-tensor `1759306569` / `29` |
| Output datatype | signed INT8 |
| Output scale | flatten / GAP scale (~0.0534) |
| Output zero point | 0 |

## Address equations

```text
pool2_read_address =
    channel * 64
  + element_index          // element_index = row * 8 + column

gap_write_address = channel
```

## Synchronous memory timing

1-cycle sync ROM latency:

```text
ISSUE → WAIT → ACCUMULATE   (×64 per channel)
CLEAR_SUM → … → AVERAGE → WRITE → ADVANCE → (next | DONE)
```

## Controller FSM

```text
IDLE
CLEAR_SUM
ISSUE
WAIT
ACCUMULATE
AVERAGE          // divide/saturate + requantize (combinational)
WRITE
ADVANCE_CHANNEL
DONE
```

## Counts

```text
Pool2 input values:     32 x 8 x 8 = 2048
Reads per channel:      64
Total Pool2 reads:      32 x 64 = 2048
GAP outputs / writes:   32
```

## Modules

| Module | Role |
| --- | --- |
| `gap_average` | Ties-away `/64` + saturate |
| `requantize` | Shared GAP mult/shift (**reused**) |
| `saturate_int8` | INT8 clip (**reused**) |
| `global_average_pool_controller` | FSM + INT32 accumulator |
| `gap_output_storage` | 32-entry INT8 RAM wrapper |
| `global_average_pool_top` | Pool2 ROM + controller + GAP storage |
| `int8_sync_rom` / `int8_sync_ram` | Memory models (**reused**) |

## Regenerate vectors

```bash
make export-gap
# or
PYTHONPATH=. python -m tools.export_gap_vectors
```

## Run tests

```bash
make test-rtl-gap
make test-rtl-conv1-full   # full regression
```

## Debug a mismatch

1. Check `vectors/gap/channel_traces/channel_XX.txt` for expected sum / raw avg / gap_q.
2. Confirm Pool2 input mem matches MaxPool2 expected tensor.
3. Unit-test `tb_gap_average` for divide/rounding issues on negative sums.
4. Confirm requant uses multiplier `1759306569` and shift `29`.

## Next milestone (not this task)

Implement the fully connected layer by reading the **32** verified GAP outputs
(done — see `docs/fully_connected_architecture.md` / `make test-rtl-fc`).

