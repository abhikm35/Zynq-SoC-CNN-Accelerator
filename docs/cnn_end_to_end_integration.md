# CNN End-to-End Integration

Connects every verified stage into one accelerator inference:

```text
Input 3x32x32
  -> Conv1/ReLU 16x32x32
  -> MaxPool1 16x16x16
  -> Conv2/ReLU 32x16x16
  -> MaxPool2 32x8x8
  -> GAP 32 INT8
  -> FC 5 INT32 logits
  -> Argmax predicted_class[2:0]
```

> Prompt sketches that list 8/16 channels are obsolete. This repository uses the
> trained channel counts above.

## Memory ownership (this phase)

Stage-specific activation RAMs (no ping-pong yet):

| Memory | Depth | Writer | Reader |
| --- | ---: | --- | --- |
| Input RAM | 3072 | TB / host | Conv1 (`EXTERNAL_ACT=1`) |
| Conv1 RAM | 16384 | Conv1 | MaxPool1 |
| Pool1 RAM | 4096 | MaxPool1 | Conv2 (`EXTERNAL_ACT=1`) |
| Conv2 RAM | 8192 | Conv2 | MaxPool2 |
| Pool2 RAM | 2048 | MaxPool2 | GAP |
| GAP storage | 32 | GAP | FC |
| Logit storage | 5 | FC | Argmax / TB |

Weight, bias, and quantization ROMs remain inside the Conv/FC engines.

## Top-level modules

| Module | Role |
| --- | --- |
| `cnn_top_controller` | Stage FSM + cycle counters |
| `cnn_accelerator_top` | Shared RAMs + verified controllers/engines |

### FSM

```text
IDLE
START_CONV1 -> WAIT_CONV1
START_POOL1 -> WAIT_POOL1
START_CONV2 -> WAIT_CONV2
START_POOL2 -> WAIT_POOL2
START_GAP   -> WAIT_GAP
START_FC    -> WAIT_FC
START_ARGMAX-> WAIT_ARGMAX
DONE
```

Each stage `*_start` is a **1-cycle pulse**. The next stage starts only after the previous `*_done` pulse. Start-while-busy is ignored (accepted only in `IDLE`).

### Done vs final write

All reused stage controllers assert `done` **after** the final memory write of that stage (same convention as standalone tops). The top controller therefore only advances once the output tensor is fully valid.

### Cycle counting

`cycle_count` clears on accepted top-level start, increments while `busy`, and stops in `DONE`. Stage counters measure cycles spent in each stage's START/WAIT pair. Transition overhead is `cycle_count - sum(stage_cycles)`.

At 100 MHz, latency_ns = `cycle_count * 10`.

## EXTERNAL_ACT

`conv1_memory_single_output` and `conv2_memory_single_output` accept `EXTERNAL_ACT=1` so activation data comes from shared RAMs with the same 1-cycle sync latency as the baked ROMs used by standalone tests.

Standalone stage tops still pass `external_act_data=0` with `EXTERNAL_ACT=0`.

## Vectors

```bash
make export-end-to-end
# or: PYTHONPATH=. python -m tools.export_end_to_end_vectors
```

Primary package: `vectors/end_to_end/`  
Multi-image (one per class): `vectors/end_to_end/multi_image/sample_*/`

## Tests

```bash
make test-rtl-end-to-end
```

Runs:

1. Vector export
2. `tb_cnn_end_to_end_single_image` (full intermediates, repeat, mid-reset recovery)
3. `tb_cnn_end_to_end_multi_image` (5 classes)

Full regression (all stages + E2E + pytest):

```bash
make test-rtl-conv1-full
```

## Debugging a mismatch

1. Confirm vectors regenerated from the same sample.
2. Check stage `*_done` counts (must be 1 each).
3. Compare the failing stage's linear address / channel / row / column.
4. Re-run the standalone stage TB for that layer.
5. Confirm `EXTERNAL_ACT` read address/latency for Conv1/Conv2.

## Out of scope (this phase)

AXI, DMA, Ethernet, Zynq PS, Vivado IP packaging, engine consolidation.

Ping-pong memories are implemented in `cnn_accelerator_pingpong_top`
(see `docs/cnn_pingpong_memory.md`). Shared Conv1/Conv2 compute is in
`cnn_accelerator_shared_conv_top` (see `docs/shared_convolution_engine.md`).
Shared Conv + shared MaxPool is in `cnn_accelerator_shared_compute_top`
(see `docs/shared_maxpool_engine.md`). This document covers the original
separate-memory verification top.
