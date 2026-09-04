# FPGA INT8 parameter and test-vector HEX export

## Representation

* Encoding: two's-complement hexadecimal
* One value per line
* No memory addresses
* Flattening: C-style row-major (`numpy` default / C order)

## Widths and signedness

| File | Width | Signedness | Source tensor shape |
| --- | --- | --- | --- |
| `conv1_weights.hex` | 8 | signed | `[16, 3, 3, 3]` |
| `conv1_bias.hex` | 32 | signed | `[16]` |
| `conv2_weights.hex` | 8 | signed | `[32, 16, 3, 3]` |
| `conv2_bias.hex` | 32 | signed | `[32]` |
| `classifier_weights.hex` | 8 | signed | `[5, 32]` |
| `classifier_bias.hex` | 32 | signed | `[5]` |
| `sample_000_input.hex` | 8 | signed | NCHW input |
| `sample_000_conv1_expected.hex` | 8 | signed | post-requantization, **pre-ReLU** |
| `sample_000_pool1_expected.hex` | 8 | signed | post-ReLU max-pool |
| `sample_000_conv2_expected.hex` | 8 | signed | post-requantization, **pre-ReLU** |
| `sample_000_pool2_expected.hex` | 8 | signed | post-ReLU max-pool |
| `sample_000_scores_expected.hex` | 32 | signed | shared-scale class scores |

## Reconstruction

Read each line as a hex integer, interpret as two's complement of the declared
width, then reshape using the table above with C-order.

Convolution weight loop order:

```text
for out_channel
  for in_channel
    for kernel_row
      for kernel_col
```

Classifier weight loop order:

```text
for out_class
  for in_feature
```

Activation loop order:

```text
for batch
  for channel
    for row
      for column
```

## Zero points and padding

All activation and weight zero points are `0` (symmetric signed scheme).
Convolution padding inserts the input activation zero point (`0`).

## Multipliers and shifts

Per-output-channel requantization uses:

```text
y = saturate(round(accumulator * multiplier / 2^shift))
```

Multiplier and shift arrays are stored under `../requantization/`.

## Class index order

[
  "stop",
  "yield",
  "no_entry",
  "speed_limit_30",
  "keep_right"
]
