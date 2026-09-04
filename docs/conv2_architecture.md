# Conv2 Architecture

Conv2 applies a 3×3 stride-1 padded convolution to the verified Pool1 tensor,
followed by per-channel requantization and ReLU.

> Trained model note: Conv2 is **16 → 32** on a **16 × 16** spatial map.
> Earlier design sketches with 8 → 16 channels / 72 MACs are obsolete.

---

## Purpose

- Expand channel capacity after spatial downsampling
- Produce the feature map consumed by MaxPool2

## Tensor shapes

| Tensor | Shape | Count | Datatype |
| --- | --- | --- | --- |
| Pool1 input | `16 × 16 × 16` | 4096 | signed INT8 |
| Conv2 weights | `32 × 16 × 3 × 3` | 4608 | signed INT8 |
| Conv2 biases | `32` | 32 | signed INT32 |
| Conv2 multipliers / shifts | `32` | 32 | signed INT32 |
| Conv2 / ReLU2 output | `32 × 16 × 16` | 8192 | signed INT8 |

Each Conv2 output feature map is produced by one learned filter of shape
**16 × 3 × 3** (one 3×3 kernel per Pool1 input channel). The 144 products from
all sixteen input channels accumulate into one Conv2 activation.

```text
MACs per output:     16 x 3 x 3 = 144
Total outputs:       32 x 16 x 16 = 8192
Total MAC ops:       8192 x 144 = 1,179,648
```

## Address equations

Pool1 (activation) — channel-first:

```text
pool1_read_address =
    input_channel * 256
  + input_row * 16
  + input_column
```

Weight:

```text
conv2_weight_address =
    output_channel * 144
  + input_channel * 9
  + kernel_row * 3
  + kernel_column
```

Bias / quant params: `address = output_channel` (per-output-channel).

Output:

```text
conv2_output_address =
    output_channel * 256
  + output_row * 16
  + output_column
```

Padding (stride 1, pad 1):

```text
input_row    = output_row + kernel_row - 1
input_column = output_column + kernel_column - 1
padding when coords are outside [0, 16)
```

No activation ROM read is issued for padded positions; the MAC uses 0.

## Quantization

- Symmetric signed INT8, all zero points = 0
- Per-output-channel multipliers and shifts (not the same as Conv1)
- `requantize`: round(acc × multiplier / 2^shift) ties away from zero
- Saturate to `[-128, 127]`, then ReLU (`max(v, 0)`)

## Synchronous memory timing

Same 1-cycle sync ROM latency as Conv1:

```text
Cycle N:   issue address + read_enable
Cycle N+1: data valid → MAC (or capture bias)
```

Single-output FSM:

```text
IDLE → ISSUE_BIAS → WAIT_BIAS → LOAD_BIAS
    → (ISSUE_OP → WAIT_MAC) × 144
    → REQUANTIZE → DONE
```

Layer controller:

```text
IDLE → START_OUTPUT → WAIT_OUTPUT → WRITE_OUTPUT → ADVANCE
     → (next | CONV2_DONE)
```

Column increments fastest, then row, then channel.

## Modules

| Module | Role |
| --- | --- |
| `conv2_address_generator` | Combinational addresses |
| `conv2_memory_single_output` | 144-MAC engine (reuses `int8_mac`, `requantize`, `relu_int8`) |
| `conv2_layer_controller` | Sweep all 8192 outputs |
| `conv2_layer_top` | ROMs + engine + controller + 8192-entry output RAM |
| `int8_sync_rom` / `int32_sync_rom` / `int8_sync_ram` | Memory models |

Conv1 RTL is left unchanged. Arithmetic modules are reused, not duplicated.

## Regenerate vectors

```bash
source ~/.venvs/zynq-edge-ai-classifier/bin/activate
cd /path/to/zynq-edge-ai-classifier
export PYTHONPATH=.
python -m tools.export_conv2_vectors
# or: make export-conv2
```

Outputs under `vectors/conv2/`.

## Run tests

```bash
make test-rtl-conv2          # Conv2 suite only
make test-rtl-conv1-full     # Full regression (includes Conv2)
```

## Debugging a mismatch

1. Decode failing address: `oc = addr/256`, `row = (addr%256)/16`, `col = addr%16`.
2. Re-run the matching selected-output case under `vectors/conv2/selected_output_traces/`.
3. Compare the 144 Pool1/weight addresses, products, and running accumulator.
4. Confirm bias / multiplier / shift for that output channel.
5. Confirm ReLU placement after saturation (no float in RTL).

## Next milestone (not this task)

Implement MaxPool2 by reading the complete **32 × 16 × 16** Conv2 tensor,
performing signed 2×2 max pooling with stride 2 independently on each channel,
writing the **32 × 8 × 8** pooled tensor, and comparing all **2048** outputs
against Python.
