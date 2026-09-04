# Fixed-Point / INT8 Specification

This document is the arithmetic contract for the integer golden model and future RTL.

## Quantization scheme

* Scheme: symmetric signed integer affine quantization (`integer_affine`)
* Weights: `int8`, symmetric, per-output-channel, narrow range `[-127, 127]`, zero-point `0`
* Activations (including network input): `int8`, symmetric, per-tensor, range `[-128, 127]`, zero-point `0`
* Biases: `int32` in accumulator scale
* Accumulators: signed `int32`
* Classifier scores: requantized to a shared per-tensor output scale, stored as `int32`

Normalized float inputs can be negative, so the project uses one consistent signed
symmetric activation convention for input and intermediate activations.

## Architecture (authoritative)

```text
Input [N,3,32,32]
→ Conv1 3→16 (3×3,s1,p1) folded BN → ReLU → MaxPool(2)
→ Conv2 16→32 (3×3,s1,p1) folded BN → ReLU → MaxPool(2)
→ GAP → Flatten[N,32] → Linear 32→5
```

BatchNorm is folded into convolution weights/biases before INT8 quantization.
Hardware/INT8 `conv*` tensors correspond to floating-point `bn*` outputs.

## Tensor dtypes

| Tensor | dtype | signedness | range |
| --- | --- | --- | --- |
| input | int8 | signed | [-128, 127] |
| conv weights | int8 | signed | [-127, 127] |
| conv bias | int32 | signed | int32 |
| conv accumulator | int32 | signed | int32 |
| conv/relu/pool activations | int8 | signed | [-128, 127] |
| GAP / flatten | int8 | signed | [-128, 127] |
| classifier weights | int8 | signed | [-127, 127] |
| classifier bias | int32 | signed | int32 |
| classifier accumulator | int32 | signed | int32 |
| scores | int32 | signed | shared score scale |

## Scales and zero points

* Weight scales: shape `[out_channels]` for convolutions, `[num_classes]` for classifier
* Activation scales: one scalar per tensor stage
* All zero points are `0`
* After each convolution requantization, ReLU and pooling preserve the convolution output scale
* GAP / flatten are requantized from the pool2 scale into a calibrated
  flatten/GAP activation scale (shared multiplier/shift)

Exact numeric scales are recorded in:

`software/exported_model/int8/metadata/quantization_manifest.json`

## Weight quantization

```text
scale_w[oc] = max(abs(weight[oc])) / 127
q_w[oc] = clip(round(weight[oc] / scale_w[oc]), -127, 127)
```

Rounding: round-to-nearest, ties away from zero.

## Activation quantization

```text
scale_a = max(abs(values)) / 127   # calibration uses max_abs / 127
q_a = clip(round(values / scale_a), -128, 127)
```

## Bias quantization

```text
bias_scale[oc] = input_activation_scale * weight_scale[oc]
bias_int32[oc] = round(float_bias[oc] / bias_scale[oc])
```

## Convolution arithmetic

```text
accumulator = bias_int32[oc]
for each MAC term:
    accumulator += int32(input - zp_in) * int32(weight - zp_w)
```

Padding inserts the input zero point (`0`). Products are widened before multiply.
Accumulators are checked against the signed int32 range.

## Requantization

For each output channel:

```text
real_multiplier = (input_scale * weight_scale[oc]) / output_scale
(integer_multiplier, shift) = frexp-normalized fixed-point encoding (31-bit significand)
y = saturate(round(accumulator * integer_multiplier / 2^shift) + output_zp)
```

* Temporary product uses `int64` in the NumPy golden model
* RTL must reproduce the same round-to-nearest, ties-away-from-zero right shift
* Multiplier width: 31-bit positive significand; shift is non-negative

## Rounding mode

```text
round to nearest, ties away from zero
```

Examples: `2.5→3`, `-2.5→-3`. Do not use banker's rounding.

## Saturation

* Activations: clip to `[-128, 127]` before casting to int8
* Narrow weights: clip to `[-127, 127]`
* Never rely on silent wraparound casts

## ReLU ordering

```text
INT32 accumulator
→ requantization + rounding
→ add output zero point (0)
→ saturate to int8
→ ReLU: max(value, 0)
```

## Max pooling

Operates directly on quantized integers. Scale and zero point are unchanged.
Comparing integers is equivalent to comparing real values under a shared scale/zp.

## Global average pooling

For each channel, sum `H*W` int8 values in a wide integer accumulator, then:

```text
averaged = round_divide_int(sum, H*W)
```

For the frozen model, pool2 is `8×8`, so `H*W = 64` (power-of-two shift).
The averaged tensor is then requantized from the pool2 scale into the calibrated
GAP/flatten activation scale with a shared integer multiplier/shift so classifier
inputs use the full intended dynamic range.

## Classifier score scaling

Classifier weights use per-class scales. Raw accumulators are therefore not directly
comparable. All class outputs are requantized to one shared `classifier_output`
scale before `argmax`.

## Tensor ordering

* Convolution weights: `[out_channels, in_channels, kH, kW]`
* Classifier weights: `[out_features, in_features]`
* Activations: `NCHW`
* Flattening for HEX: C-style row-major

## FPGA HEX encoding

Under `software/exported_model/int8/fpga/`:

* Two's-complement hexadecimal
* Fixed width (8 for int8, 32 for int32)
* One value per line
* No addresses
* Round-trip verified against `.npy` arrays

See `software/exported_model/int8/fpga/README.md`.

## Overflow analysis

Theoretical and empirical accumulator bounds are reported in:

`results/comparisons/int8_overflow_analysis.json`

Analysis includes MAC-term theoretical maxima, reference-set observations, test-set
observations, required signed bit width, and INT32 overflow flags.
