# CNN Ping-Pong Activation Memory

Optimized end-to-end top that replaces stage-specific activation RAMs with
exactly two reusable buffers.

> Prompt sketches that list 8/16 channels are obsolete. This repository uses
> the trained channel counts below.

## Previous verification architecture

Separate activation memories (still used by `cnn_accelerator_top`):

| Memory | Depth (bytes) | Writer | Reader |
| --- | ---: | --- | --- |
| Input RAM | 3072 | TB / host | Conv1 |
| Conv1 RAM | 16384 | Conv1 | MaxPool1 |
| Pool1 RAM | 4096 | MaxPool1 | Conv2 |
| Conv2 RAM | 8192 | Conv2 | MaxPool2 |
| Pool2 RAM | 2048 | MaxPool2 | GAP |
| GAP storage | 32 | GAP | FC |
| Logit storage | 5 | FC | Argmax |

Logical large activation storage: **3072 + 16384 + 4096 + 8192 + 2048 = 33792 bytes**.

## New optimized activation architecture

| Memory | Depth | Role |
| --- | ---: | --- |
| Activation RAM A | 16384 x INT8 | Input load, Pool1 write, Pool2 write, Conv1/Conv2/GAP read |
| Activation RAM B | 16384 x INT8 | Conv1 write, Conv2 write, Pool1/Pool2 read |
| GAP storage | 32 | unchanged |
| Logit storage | 5 | unchanged |

Logical ping-pong activation storage: **16384 + 16384 = 32768 bytes**
(~1 KB logical reduction vs separate stage RAMs; physical BRAM mapping may differ).

Physical Vivado BRAM inference may differ from logical byte counts; do not claim
exact BRAM savings until synthesis is run.

## Ownership timeline

```text
Time 0 (after input load):
  RAM A = input image (0..3071 valid)
  RAM B = don't care

After Conv1:
  RAM A = old input (stale OK)
  RAM B = Conv1 output (0..16383)

After Pool1:
  RAM A = Pool1 output (0..4095 valid; 4096..16383 may be stale)
  RAM B = old Conv1 output (stale OK until Conv2)

After Conv2:
  RAM A = old Pool1 output (stale OK until Pool2)
  RAM B = Conv2 output (0..8191 valid)

After Pool2:
  RAM A = Pool2 output (0..2047 valid)
  RAM B = old Conv2 output (still readable for debug)

After GAP:
  RAM A still contains Pool2 output
  GAP registers/storage contain 32 INT8 values
```

Stale values outside the current valid tensor range are ignored because each
stage only addresses its logical range starting at address 0. RAMs are **not**
cleared between layers.

## Ping-pong mapping

```text
INPUT LOAD:  write RAM A
CONV1:       read RAM A, write RAM B
POOL1:       read RAM B, write RAM A
CONV2:       read RAM A, write RAM B
POOL2:       read RAM B, write RAM A
GAP:         read RAM A
FC / Argmax: use GAP / logit storage (not activation RAMs)
```

## Modules

| Module | Role |
| --- | --- |
| `activation_ram` | 16384 x INT8 wrapper around `int8_sync_ram` (1-cycle sync read) |
| `cnn_accelerator_pingpong_top` | Two RAMs + ownership mux + reused stage engines |
| `cnn_top_controller` | Unchanged stage FSM |
| `cnn_accelerator_top` | Original separate-memory reference (still tested) |

## Addressing (unchanged logical formulas)

Each stage still uses address 0 as the start of its tensor. No base offsets
are added inside the ping-pong buffers.

## Tests

```bash
make test-rtl-pingpong
```

Runs vector export, compiles the ping-pong top, verifies every intermediate
tensor via write capture (Conv1→RAM B, Pool1→RAM A, Conv2→RAM B, Pool2→RAM A),
checks GAP/FC/argmax, repeated inference, mid-reset recovery, and multi-image
regression.

```bash
make test-rtl-end-to-end   # original separate-memory top
make test-rtl-conv1-full   # full RTL + ping-pong + pytest
```

## Out of scope (this phase)

Reusable Conv1/Conv2 engine consolidation, reusable Pool1/Pool2 consolidation,
AXI, DMA, Ethernet, Zynq PS, Vivado IP packaging, arithmetic parallelism changes.
