# Conv1 Memory Addressing

Bit-exact memory layout and memory-driven single-output engine for Conv1.

**Authoritative channel count:** Conv1 is **3 → 16** (trained model). Prompt
sketches that mention 8 output channels are obsolete; weights/biases/quant
tables are length **16**, and weight memory holds **432** INT8 values.

---

## Activation memory layout

Flattened NCHW / C-order planar:

```text
activation_address =
    channel * height * width
    + row * width
    + column
```

For Conv1 input `[3, 32, 32]`:

```text
activation_address = input_channel * 1024 + input_row * 32 + input_column
```

Range: `0 .. 3071`. File: `vectors/conv1_memory/input_image.mem` (3072 × INT8 hex).

## Weight memory layout

OIHW flatten:

```text
weight_address =
    output_channel * input_channels * kernel_height * kernel_width
    + input_channel * kernel_height * kernel_width
    + kernel_row * kernel_width
    + kernel_column
```

For Conv1 (`I=3`, `kH=kW=3`):

```text
weight_address = output_channel * 27 + input_channel * 9 + kernel_row * 3 + kernel_column
```

Range: `0 .. 431` (16×27). File: `vectors/conv1_memory/conv1_weights.mem`.

## Bias / quant memory

```text
bias_address = output_channel          # 0 .. 15
multiplier_address = output_channel
shift_address = output_channel
```

Files:

* `conv1_biases.mem` — INT32 hex (8 digits)
* `conv1_multipliers.mem` — INT32
* `conv1_shifts.mem` — INT32 (used as shift amount, low bits)

JSON summary: `conv1_quant_params.json`.

## Padding

Stride 1, pad 1:

```text
input_row    = output_row + kernel_row - 1
input_column = output_column + kernel_column - 1
```

Valid iff `0 <= input_row < 32` and `0 <= input_column < 32`.

If padded:

* `padding = 1`
* activation value forced to **0** (input zero point)
* **no** activation ROM read is issued
* weight is still read; MAC still executes; product is 0; counters advance

## Counter ordering

Must match Python golden / exporter:

```text
for input_channel = 0 .. 2:
  for kernel_row = 0 .. 2:
    for kernel_column = 0 .. 2:
      MAC
```

`kernel_column` fastest. Exactly **27** MACs.

## Synchronous memory latency

`int8_sync_rom` / `int32_sync_rom`:

```text
Cycle N:   read_enable + address presented
Cycle N+1: read_data valid after the posedge (held until next enabled read)
```

## FSM timing (`conv1_memory_single_output`)

```text
IDLE
  -> ISSUE_BIAS     // bias/mult/shift addresses + read_enable
  -> WAIT_BIAS      // 1-cycle ROM latency
  -> LOAD_BIAS      // capture params; load bias into MAC (ROM data wired live)
  -> ISSUE_OP       // act/wgt addresses; act_re = ~padding
  -> WAIT_MAC       // operands valid; MAC enable; advance counters
       (repeat ISSUE_OP/WAIT_MAC until 27 MACs)
  -> REQUANTIZE     // capture final_acc, requantized, ReLU
  -> DONE           // done pulse; return IDLE
```

### Cycle-level example (one MAC)

```text
Cycle N:   ISSUE_OP — present activation (if valid) and weight addresses
Cycle N+1: WAIT_MAC — ROM data valid; assert mac_enable; perform MAC
Cycle N+2: ISSUE_OP for next (ic, kr, kc)
```

### Measured cycles for one output

From the cycle `start` is sampled through the `done` pulse, the testbench
reports:

```text
cycles_from_start = 60
```

Breakdown:

```text
IDLE samples start
ISSUE_BIAS + WAIT_BIAS + LOAD_BIAS     = 3
27 × (ISSUE_OP + WAIT_MAC)             = 54
REQUANTIZE + DONE                      = 2
(+ reset/setup overhead in TB local_c) ≈ 60 reported
```

## Regenerating vectors

```bash
source ~/.venvs/zynq-edge-ai-classifier/bin/activate
cd /path/to/zynq-edge-ai-classifier
export PYTHONPATH=.
python -m tools.export_conv1_memory_vectors
```

Selected cases:

| Name | oc | row | col |
| --- | --- | --- | --- |
| center | 0 | 10 | 10 |
| top_left | 0 | 0 | 0 |
| bottom_right | 0 | 31 | 31 |
| channel5 | 5 | 12 | 17 |

Traces: `vectors/conv1_memory/selected_output_traces/*_trace.txt`.

## Running tests

```bash
make test-rtl-memory-output
# or
PYTHONPATH=. python -m tools.run_rtl_tests
```

This regenerates vectors, runs prior arithmetic RTL tests, address-generator
tests, ROM latency tests, memory-driven output tests, and Python pytest.

## Inspecting a failed MAC

On mismatch the TB prints:

* test phase / cycle / MAC index
* output (channel, row, column)
* input channel / kernel row / column
* expected vs actual addresses, padding, operands, product, accumulator

Also open the matching `*_trace.txt` from the exporter for the Python golden
step-by-step MAC log.
