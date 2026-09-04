# Project Overview

`zynq-edge-ai-classifier` is an in-progress Zynq edge-AI traffic-sign classifier.

## Frozen floating-point architecture

The trained `TinyCNN` architecture is:

```text
Input:                 [batch, 3, 32, 32]
Conv1 (3->16, 3x3, s1, p1, bias=False): [batch, 16, 32, 32]
BatchNorm1:            [batch, 16, 32, 32]
ReLU1:                 [batch, 16, 32, 32]
MaxPool1 (2x2, s2):    [batch, 16, 16, 16]
Conv2 (16->32, 3x3, s1, p1, bias=False): [batch, 32, 16, 16]
BatchNorm2:            [batch, 32, 16, 16]
ReLU2:                 [batch, 32, 16, 16]
MaxPool2 (2x2, s2):    [batch, 32, 8, 8]
GlobalAvgPool:         [batch, 32, 1, 1]
Flatten:               [batch, 32]
Linear classifier:     [batch, 5]
```

Trainable parameter count: **5,301**.

## Classes

```text
0 -> stop
1 -> yield
2 -> no_entry
3 -> speed_limit_30
4 -> keep_right
```

## Current phase

Freeze the trained floating-point model, export parameters and intermediate
activations, and reproduce inference with an independent NumPy model.

## Future phases

INT8 quantization, FPGA SystemVerilog acceleration, AXI/DMA integration,
Ethernet intake, and later HDMI video support.
