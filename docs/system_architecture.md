# System Architecture

## Software path (current)

```text
GTSRB filtered dataset
  -> evaluation preprocessing (RGB, resize 32x32, ToTensor, normalize)
  -> TinyCNN floating-point PyTorch model
  -> class scores [batch, 5]
```

## Frozen TinyCNN

| Stage | Configuration | Output shape |
| --- | --- | --- |
| Input | RGB NCHW | `[N, 3, 32, 32]` |
| Conv1 | 3->16, k3, s1, p1, no bias | `[N, 16, 32, 32]` |
| BN1 | BatchNorm2d(16) | `[N, 16, 32, 32]` |
| ReLU1 | ReLU | `[N, 16, 32, 32]` |
| Pool1 | MaxPool 2x2 s2 | `[N, 16, 16, 16]` |
| Conv2 | 16->32, k3, s1, p1, no bias | `[N, 32, 16, 16]` |
| BN2 | BatchNorm2d(32) | `[N, 32, 16, 16]` |
| ReLU2 | ReLU | `[N, 32, 16, 16]` |
| Pool2 | MaxPool 2x2 s2 | `[N, 32, 8, 8]` |
| GAP | AdaptiveAvgPool2d(1,1) | `[N, 32, 1, 1]` |
| Flatten | start_dim=1 | `[N, 32]` |
| Linear | 32->5 | `[N, 5]` |

### Parameter shapes

| Tensor | Shape |
| --- | --- |
| `conv1.weight` | `[16, 3, 3, 3]` |
| `bn1.weight` / `bn1.bias` | `[16]` |
| `conv2.weight` | `[32, 16, 3, 3]` |
| `bn2.weight` / `bn2.bias` | `[32]` |
| `classifier.weight` | `[5, 32]` |
| `classifier.bias` | `[5]` |

BatchNorm can later be folded into convolution weights and biases for FPGA
inference. The independent NumPy golden model currently executes BatchNorm
explicitly so it matches PyTorch layer by layer.

## Preprocessing

1. Convert to RGB
2. Resize to 32 x 32 with torchvision bilinear resize
3. Convert to float32 tensor in `[0, 1]`
4. Normalize with training-set statistics:
   - mean `[0.401230, 0.358534, 0.368098]`
   - std `[0.291221, 0.272794, 0.274302]`

## Planned hardware path

```text
ARM PS
  -> DDR buffers / DMA
  -> AXI-Stream image input
  -> FPGA CNN accelerator
  -> AXI-Lite status / results
  -> Ethernet or HDMI overlay output
```
