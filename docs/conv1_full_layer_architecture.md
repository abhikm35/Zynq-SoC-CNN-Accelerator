# Complete Conv1 Layer Architecture

Full Conv1 feature extraction using one sequential MAC datapath.

## Shapes (trained model)

| Tensor | Shape |
| --- | --- |
| Input | `3 × 32 × 32` |
| Conv1 output | **`16 × 32 × 32`** |
| Total activations | **16384** |
| MACs / output | 27 |
| Total MACs | `16384 × 27 = 442368` |

> Prompt sketches that mention **8** output channels / **8192** activations are
> obsolete. The trained and quantized checkpoint is Conv1 **3→16**. Hardware
> matches the trained model.

## Input vs output channels

Each output activation in any output channel combines all three RGB input
channels (9+9+9 MACs into one INT32 accumulator). RGB planes are **not**
separate final outputs.

One convolution engine is reused for all 16 output channels (not 16 copies).

## Iteration order

```text
for output_channel = 0 .. 15:
  for output_row = 0 .. 31:
    for output_column = 0 .. 31:
      run single-output engine (27 MACs)
      write post-ReLU INT8 to output RAM
```

Column increments fastest.

## Addresses

```text
weight_address =
    oc * 27 + ic * 9 + kr * 3 + kc

bias_address = oc
quant_address = oc          # per-channel multiplier and shift

output_write_address =
    oc * 1024 + row * 32 + col
```

Output RAM regions: channel `c` occupies `[c*1024 .. c*1024+1023]`.

## Controller FSM

```text
IDLE → START_OUTPUT → WAIT_OUTPUT → WRITE_OUTPUT → ADVANCE
     → (more ? START_OUTPUT : CONV1_DONE) → IDLE
```

`conv1_done` is a one-cycle pulse after address 16383 is written.

## Modules

| Module | Role |
| --- | --- |
| `conv1_memory_single_output` | One pixel datapath + memories |
| `conv1_layer_controller` | oc/row/col sweep + writes |
| `int8_sync_ram` | Parameterized output RAM (16384×INT8) |
| `conv1_layer_top` | Integration wrapper |

One-channel controller (`conv1_channel_*`) remains for the prior milestone.

## Vectors / tests

```bash
PYTHONPATH=. python -m tools.export_conv1_full_vectors
make test-rtl-conv1-full
```

Expected map: `vectors/conv1_full/conv1_expected.mem` (16384 lines).

## Cycle accounting

Measured by `tb_conv1_full_layer` (start → `conv1_done`):

```text
total_cycles        = 1032194
cycles_per_output   = 63
cycles_per_channel  = 64512
output RAM writes   = 16384
checksum            = 69013 (sample_000)
```

Mathematical MAC count (`16384 × 27 = 442368`) is not equal to controller cycle count.

## Debugging

On mismatch the TB prints address / oc / row / col / expected / actual hex.
Selected traces: `vectors/conv1_full/selected_output_traces/`.
