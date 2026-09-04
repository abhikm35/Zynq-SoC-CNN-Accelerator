# Conv1 One-Channel Architecture

This milestone expands the verified memory-driven **single-output** Conv1 engine
into a controller that computes **all 32×32 activations of output channel 0**.

## Input vs output channels

```text
The design calculates one output feature-map channel, not one RGB input channel.

Every output activation in output channel 0 combines:
- 9 values from input channel 0
- 9 values from input channel 1
- 9 values from input channel 2

These 27 products are accumulated into one result.
```

* **Input channels (3):** RGB planes in the `3×32×32` activation ROM.
* **Output channel (this phase: 0 only):** one `32×32` feature map in the output RAM.

Trained model Conv1 is **3→16**. Prompt sketches that mention 8 output channels
are obsolete for width; this phase still only implements **output channel 0**.

## Sequential processing (unchanged datapath)

One MAC unit; input channels are sequential:

```text
ic=0: 9 MACs → ic=1: 9 MACs → ic=2: 9 MACs  (total 27)
```

No parallel RGB MACs, no line buffers.

## Modules

| Module | Role |
| --- | --- |
| `conv1_memory_single_output` | One pixel: memories + 27 MACs + requant + ReLU |
| `conv1_channel_controller` | Sweep `(row,col)`, handshake, write RAM |
| `int8_sync_ram` | 1024×INT8 output feature map |
| `conv1_channel_top` | Integrates controller + engine + output RAM |

## Single-output handshake (reused)

| Signal | Behavior |
| --- | --- |
| `start` | 1-cycle request; coords sampled in IDLE |
| `busy` | High while computing |
| `done` | **1-cycle pulse** in DONE; then returns IDLE |
| `relu_output` | Captured in REQUANTIZE; stable when `done` asserts |
| Cycles / pixel | ~60 from start sample to done (see prior milestone) |

New `start` is accepted only after the engine returns to IDLE (controller waits
WRITE+ADVANCE after `done`).

## Controller FSM

```text
IDLE
  -> START_OUTPUT     // engine_start=1, oc=0, current row/col
  -> WAIT_OUTPUT      // wait engine_done; latch relu_output
  -> WRITE_OUTPUT     // one-cycle write_enable to output RAM
  -> ADVANCE          // next col (then row); or CHANNEL_DONE if (31,31)
  -> CHANNEL_DONE     // channel_done pulse; output_count==1024
  -> IDLE
```

Column increments fastest: `(0,0), (0,1), … (0,31), (1,0), … (31,31)`.

## Output RAM

* Depth 1024, width INT8, sync write + sync read (1-cycle read latency)
* Address (channel-local for oc=0):

```text
output_address = output_row * 32 + output_column
```

Future full Conv1:

```text
full_output_address =
    output_channel * 1024
    + output_row * 32
    + output_column
```

## Vectors

```bash
PYTHONPATH=. python -m tools.export_conv1_channel_vectors
```

Writes `vectors/conv1_channel/conv1_channel0_expected.mem` (1024 values) and
selected pixel traces under `selected_pixel_traces/`.

Also requires existing `vectors/conv1_memory/` (input/weights/biases).

## Tests

```bash
make test-rtl-conv1-channel
# or
PYTHONPATH=. python -m tools.run_rtl_tests
```

Runs prior arithmetic + memory-driven tests, then `tb_conv1_channel`, then pytest.

## Inspecting a mismatch

TB prints address, row, col, expected/actual decimal and hex. For arithmetic
detail open:

```text
vectors/conv1_channel/selected_pixel_traces/<name>_trace.txt
```

## Cycle accounting

Reported by the testbench after `channel_done` (measured):

```text
total_cycles        ≈ 64514   (start → channel_done)
cycles_per_output   ≈ 63
output RAM writes   = 1024
checksum            = 744 (sample_000; matches Python)
```

Breakdown per pixel ≈ engine (~60) + controller WRITE/ADVANCE/START overhead (~3).
