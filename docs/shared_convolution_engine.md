# Shared Convolution Engine

One reusable configurable convolution compute engine executes Conv1 and Conv2
at different times under top-level control. Pool1/Pool2, GAP, FC, and Argmax
remain separate. Ping-pong activation RAM mapping is unchanged.

> Prompt sketches that list 8/16 channels are obsolete. This repository uses
> the trained channel counts below.

## Why Conv1 and Conv2 can share hardware

Both layers are the same algorithm:

```text
bias init -> signed INT8×INT8 MAC loop -> requantize -> ReLU -> INT8 write
```

They never run simultaneously in the CNN schedule (Conv1 → Pool1 → Conv2), so
one physical datapath and controller can be time-multiplexed.

## Why they do not execute simultaneously

The top FSM (`cnn_top_controller`) sequences stages. The shared engine is
started once for Conv1 (`layer_is_conv2=0`) and later once for Conv2
(`layer_is_conv2=1`). Assertions reject `start` while busy and reject
`layer_is_conv2` changes while busy.

## Conceptual timeline

```text
CNN start

shared conv engine
configured as Conv1
        |
        v
Conv1 completes

Pool1 runs

shared conv engine
configured as Conv2
        |
        v
Conv2 completes

Pool2 -> GAP -> FC -> Argmax
```

## Trained configuration (source of truth)

| Parameter | Conv1 | Conv2 |
| --- | ---: | ---: |
| input channels | 3 | 16 |
| output channels | 16 | 32 |
| spatial size | 32×32 | 16×16 |
| kernel / pad / stride | 3×3 / 1 / 1 | 3×3 / 1 / 1 |
| MACs per output | 27 | 144 |
| total outputs | 16384 | 8192 |
| total MACs | 442368 | 1179648 |
| weights | 432 | 4608 |
| input channel stride | 1024 | 256 |
| output channel stride | 1024 | 256 |
| weights per output channel | 27 | 144 |

## Address equations

Activation (when not padded):

```text
addr = input_channel * input_channel_stride
     + input_row * input_width
     + input_column
```

Output:

```text
addr = output_channel * output_channel_stride
     + output_row * output_width
     + output_column
```

Weight (PyTorch `[oc][ic][kr][kc]` flatten):

```text
addr = output_channel * weights_per_output_channel
     + input_channel * 9
     + kernel_row * 3
     + kernel_column
```

Padding uses `input_row/col = output + kernel - 1`; out-of-bounds yields
activation `0` and no RAM read.

## Parameter / quantization selection

Weight, bias, multiplier, and shift ROMs stay **separate per layer**.
`layer_is_conv2` muxes which bank feeds the shared MAC/requant path.
Quantization parameters are never unified or approximated.

## Ping-pong interaction

Same verified mapping as `cnn_accelerator_pingpong_top`:

```text
Conv1: read RAM A, write RAM B
Pool1: read RAM B, write RAM A
Conv2: read RAM A, write RAM B
```

## Modules

| Module | Role |
| --- | --- |
| `shared_conv_address_generator` | Muxes verified Conv1/Conv2 address generators |
| `shared_conv_single_output` | One pixel engine + both parameter ROM banks |
| `shared_conv_layer_controller` | Full-layer iterate / write / done |
| `shared_conv_engine` | Controller + single-output wrapper |
| `cnn_accelerator_shared_conv_top` | Ping-pong top with one shared engine |

Old `conv1_*` / `conv2_*` modules and `cnn_accelerator_pingpong_top` remain
for regression.

## Logical duplication removed (new top only)

Previous ping-pong top instantiated two full compute stacks (Conv1 MAC path +
controller and Conv2 MAC path + controller). The shared-conv top instantiates
**one** MAC datapath, accumulator, requant/ReLU path, and convolution
controller. Parameter ROMs remain duplicated. Exact LUT/DSP savings require
Vivado synthesis.

## Test command

```bash
export PATH=~/tools/verilator-5.028/bin:$PATH
export VERILATOR_ROOT=~/tools/verilator-5.028
make test-rtl-shared-conv
```

Runs old Conv1/Conv2 full-layer regressions, shared-engine Conv1/Conv2
standalone tests, and shared-conv ping-pong end-to-end (including repeated
inference and multi-image).
