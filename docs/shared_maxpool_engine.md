# Shared Max-Pooling Engine

One reusable signed INT8 2×2 stride-2 max-pool executes Pool1 and later Pool2
under top-level control. Shared convolution, ping-pong RAMs, GAP, FC, and
Argmax are unchanged.

> Prompt sketches that list 8/16 pool channels are obsolete. This repository
> uses the trained channel counts below.

## Why Pool1 and Pool2 can share hardware

Both layers are identical algorithms:

```text
for each output location
  read 4 window values (1-cycle sync RAM each)
  signed max-of-four
  write one INT8
```

They never run simultaneously (Conv1 → Pool1 → Conv2 → Pool2), so one physical
engine is time-multiplexed.

## Architecture timeline

```text
RAM A
  |
  | Shared Conv - Conv1 mode
  v
RAM B
  |
  | Shared Pool - Pool1 mode
  v
RAM A
  |
  | Shared Conv - Conv2 mode
  v
RAM B
  |
  | Shared Pool - Pool2 mode
  v
RAM A
  |
  | GAP
  v
GAP registers
  |
  | FC
  v
Logits
  |
  | Argmax
  v
Class
```

## Trained configuration

| Parameter | Pool1 | Pool2 |
| --- | ---: | ---: |
| channels | 16 | 32 |
| input size | 32×32 | 16×16 |
| output size | 16×16 | 8×8 |
| kernel / stride | 2×2 / 2 | 2×2 / 2 |
| input channel stride | 1024 | 256 |
| output channel stride | 256 | 64 |
| total outputs | 4096 | 2048 |
| total input reads | 16384 | 8192 |

## Address equations

```text
input_address =
  channel * input_channel_stride
  + input_row * input_width
  + input_column

output_address =
  channel * output_channel_stride
  + output_row * output_width
  + output_column
```

`input_row/col = 2*pool_row/col + window offset` for window indices 0..3.

## Signed INT8 comparison

Uses verified `max4_int8` (signed `>` only). The four window values are
captured first, then compared together — **never** initialize the running max
to `0` (all-negative windows would be wrong).

## Ping-pong mapping (unchanged)

```text
Pool1 / Pool2: always read RAM B, write RAM A
```

## Modules

| Module | Role |
| --- | --- |
| `shared_maxpool_address_generator` | Muxes verified Pool1/Pool2 address generators |
| `shared_maxpool_engine` | Unified FSM + `max4_int8` |
| `cnn_accelerator_shared_compute_top` | Shared conv + shared pool + ping-pong |

Old `maxpool1_*` / `maxpool2_*` sources and prior tops remain for regression.

## Logical duplication removed (new top only)

Previous shared-conv top instantiated separate `maxpool1_controller` and
`maxpool2_controller`. The shared-compute top instantiates **one**
`shared_maxpool_engine`. Exact LUT savings require Vivado synthesis.

## Test command

```bash
export PATH=~/tools/verilator-5.028/bin:$PATH
export VERILATOR_ROOT=~/tools/verilator-5.028
make test-rtl-shared-pool
```
