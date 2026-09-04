# MaxPool1 Architecture

MaxPool1 reduces the verified Conv1 / ReLU1 activation map with a
**2 × 2, stride-2** signed INT8 maximum. There is **no** multiply, accumulate,
requantization, or ReLU in this stage.

> Trained model note: Conv1 is **16** channels (`3→16`), so Pool1 is
> **16 × 16 × 16 = 4096** outputs. Earlier design sketches with 8 channels /
> 2048 outputs are obsolete.

---

## Purpose

- Spatial downsampling of Conv1 for Conv2
- Preserve the strongest activation in each 2 × 2 neighborhood
- Keep the same signed INT8 representation (bit-identical to Python)

## Tensor shapes

| Tensor | Shape | Count | Datatype |
| --- | --- | --- | --- |
| Conv1 input (to pool) | `16 × 32 × 32` | 16384 | signed INT8 |
| Pool1 output | `16 × 16 × 16` | 4096 | signed INT8 |

Channels are pooled **independently**. A pooling window never mixes channels.

## Window and stride

```text
pool1[ch][pr][pc] = max(
    conv1[ch][2*pr    ][2*pc    ],
    conv1[ch][2*pr    ][2*pc + 1],
    conv1[ch][2*pr + 1][2*pc    ],
    conv1[ch][2*pr + 1][2*pc + 1]
)
```

Stride 2 ⇒ non-overlapping windows; output spatial size is exactly half.

## Address equations

Conv1 (input) address — channel-first:

```text
conv1_read_address =
    channel * 1024
  + input_row * 32
  + input_column
```

Pool1 (output) address — channel-first:

```text
pool1_write_address =
    channel * 256
  + pool_row * 16
  + pool_column
```

Examples:

| Location | Address |
| --- | --- |
| ch0, pr0, pc0 | 0 |
| ch0, pr15, pc15 | 255 |
| ch1, pr0, pc0 | 256 |
| ch15, pr15, pc15 | 4095 |

Conv1 memory regions (per channel × 1024):

```text
Channel 0:     0 .. 1023
Channel 15: 15360 .. 16383
```

Pool1 memory regions (per channel × 256):

```text
Channel 0:    0 .. 255
Channel 15: 3840 .. 4095
```

## Signed comparison

Comparisons are **signed**. Example: `-1 > -20`. Treating INT8 as unsigned
would invert many negative results (e.g. `0xFF` falsely winning over small
positives).

RTL: `rtl/max4_int8.sv` — pairwise `>` on `logic signed [7:0]`.

MaxPool1 does **not** requantize and does **not** apply another ReLU.

## Synchronous memory timing

Conv1 source is `int8_sync_rom` (or equivalent sync RAM) with **1-cycle**
read latency:

```text
Cycle N:   present read_enable + address
Cycle N+1: read_data updates
Cycle N+2: controller captures value (NBA-safe)
```

The controller never consumes ROM data in the same cycle the address is
issued.

## Four-read sequence

For every pooled output the controller:

1. Issues top-left Conv1 address → wait → capture A  
2. Issues top-right → wait → capture B  
3. Issues bottom-left → wait → capture C  
4. Issues bottom-right → wait → capture D  
5. Computes signed max  
6. Writes once to Pool1 RAM  
7. Advances column (fastest), then row, then channel  

## Controller FSM

```text
IDLE
  → ISSUE → WAIT → CAPTURE   (window_index 0..3)
  → COMPARE → WRITE → ADVANCE
  → (next ISSUE | DONE)
```

`pool1_done` asserts only after address **4095** has been written and
`output_count == 4096`.

## Modules

| Module | Role |
| --- | --- |
| `maxpool2x2_address_generator` | Combinational addresses / coordinates |
| `max4_int8` | Signed max of four |
| `maxpool1_controller` | FSM + counters |
| `maxpool1_top` | Conv1 ROM + controller + Pool1 RAM |
| `int8_sync_rom` | Conv1 tensor source (reuse) |
| `int8_sync_ram` | Pool1 sink, depth 4096, addr 12 bits |

## Counts

```text
Conv1 input values:     16 x 32 x 32 = 16384
Pool1 output values:    16 x 16 x 16 = 4096
Reads per output:       4
Total Conv1 reads:      4096 x 4 = 16384
Total Pool1 writes:     4096
```

Each output also spends a few cycles on compare / write / advance. The full
testbench reports total cycles and average cycles per pooled output.

## Regenerate vectors

```bash
source ~/.venvs/zynq-edge-ai-classifier/bin/activate
cd /path/to/zynq-edge-ai-classifier
export PYTHONPATH=.
python -m tools.export_pool1_vectors
```

Outputs under `vectors/pool1/`:

- `pool1_expected.mem` / `.json` / `_summary.txt`
- `conv1_input_for_pool.mem`
- `channel_summaries/`
- `selected_window_traces/`

## Run tests

```bash
make test-rtl-maxpool1          # Pool1 suite (+ prior vectors as needed)
make test-rtl-conv1-full        # Full regression including MaxPool1
# or
python -m tools.run_rtl_tests --only pool
python -m tools.run_rtl_tests   # all
```

## Debugging a mismatch

1. Note failing Pool1 address → decode `ch = addr/256`, `pr = (addr%256)/16`, `pc = addr%16`.  
2. Open `vectors/pool1/selected_window_traces/` if the site is a traced window.  
3. Confirm the four Conv1 addresses and signed values.  
4. Check `max4_int8` vs Python `max(...)`.  
5. Confirm write happened once and after four captures (no stale ROM data).  
6. Re-export vectors if the golden sample input changed.

## Next milestone (not this task)

Generalize the verified convolution architecture for **Conv2**, using the
**16 × 16 × 16** Pool1 tensor as input, **32** output channels, 3 × 3 kernels,
**72** MAC operations per output, Conv2-specific weights / biases /
quantization parameters, and compare all **8192** Conv2 outputs against Python.
