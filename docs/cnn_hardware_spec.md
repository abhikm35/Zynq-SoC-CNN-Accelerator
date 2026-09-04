# CNN Hardware Arithmetic Specification

This document freezes the **bit-exact** arithmetic contract between the Python
integer golden model (`software/inference/integer_inference.py`,
`software/quantization/fixed_point.py`) and the initial SystemVerilog modules
under `rtl/`.

**Authoritative software sources (do not contradict):**

* Trained model: `software/models/tiny_cnn.py` — Conv1 **3→16**, Conv2 **16→32**
* INT8 parameters: `software/exported_model/int8/`
* Manifest: `software/exported_model/int8/metadata/quantization_manifest.json`
* Golden model: `software/inference/integer_inference.py`
* Requantization: `software/quantization/fixed_point.py::requantize_int32`

> Note: An earlier design sketch mentioned 8/16 channels. The **trained and
> quantized** model uses **16 / 32** channels. Hardware must match the trained
> model, not the sketch.

---

## 1. Quantization convention (derived from code)

| Property | Value |
| --- | --- |
| Scheme | Symmetric signed integer affine |
| Activation dtype | `int8`, range `[-128, 127]` |
| Weight dtype | `int8`, narrow range `[-127, 127]` |
| Bias dtype | `int32` (accumulator scale) |
| Accumulator dtype | `int32` |
| Input / weight / output zero points | **all `0`** |
| Activation scales | **per-tensor** |
| Weight scales | **per-output-channel** |
| Conv requant multipliers / shifts | **per-output-channel** |

Dequantization identity (software documentation only; **RTL does not dequantize**):

```text
real ≈ scale * (q - zero_point)
     = scale * q          # because zero_point == 0
```

---

## 2. Conv1 geometry

| Parameter | Value |
| --- | --- |
| Input shape (NCHW, N=1) | `[1, 3, 32, 32]` |
| Output shape | `[1, 16, 32, 32]` |
| Kernel | `3 × 3` |
| Stride | `1` |
| Padding | `1` (constant = input zero point = `0`) |
| Input channels | `3` |
| Output channels | `16` |
| MACs per output pixel | `3 × 3 × 3 = 27` |

### Datatypes

| Signal | dtype |
| --- | --- |
| Input activation | signed `int8` |
| Weight | signed `int8` |
| Bias | signed `int32` |
| Product (optional debug) | signed `int16` (fits `int8×int8`) |
| Accumulator | signed `int32` |
| Requantized / ReLU output | signed `int8` |

### Scales (runtime values live in export; shapes are fixed)

| Parameter | Scope | Shape |
| --- | --- | --- |
| `input_scale` | per-tensor | scalar |
| `conv1_weight_scales` | per-output-channel | `[16]` |
| `conv1_output_scale` | per-tensor | scalar |
| `conv1_multipliers` | per-output-channel | `[16]` int32 |
| `conv1_shifts` | per-output-channel | `[16]` int32 |

Zero points: `input_zp = weight_zp = output_zp = 0`.

---

## 3. Exact Conv1 integer equations (from Python)

Software uses `input_zero_point = 0` and `weight_zero_point = 0`, so centering
is a no-op. The conceptual nested loop that RTL must reproduce:

```text
# Padding: out-of-image samples are the input zero point (0).
acc = bias_int32[oc]                                 # int32

for ic in 0 .. 2:
  for kr in 0 .. 2:
    for kc in 0 .. 2:
      in_r = out_r + kr - padding
      in_c = out_c + kc - padding
      if in_r,in_c outside [0,31]:
        x = 0                                        # pad with zp
      else:
        x = input[ic, in_r, in_c]                    # int8
      w = weight[oc, ic, kr, kc]                     # int8
      # zp subtraction is present in code but zp==0:
      x_c = int32(x) - int32(0)
      w_c = int32(w) - int32(0)
      acc += int32(x_c) * int32(w_c)                 # widen before mul

# Requantization (per output channel oc):
wide   = int64(acc) * int64(multiplier[oc])
scaled = rounding_right_shift(wide, shift[oc])       # ties away from 0
with_zp = scaled + output_zero_point                 # +0
clamped = clip(with_zp, -128, 127)                   # int8  ("conv1")
relu_out = max(clamped, 0)                            # int8  ("relu1")
```

### Ordering (must match software)

```text
INT32 accumulator
  -> multiply by integer multiplier
  -> rounding right shift
  -> add output zero point (0)
  -> saturate to [-128, 127]
  -> ReLU: max(q, 0)
```

ReLU is **after** saturation, not fused inside the accumulator.

---

## 4. Rounding rule (exact)

`rounding_right_shift(value, shift)` from `fixed_point.py`:

```text
if shift == 0:
    return value
half = 1 << (shift - 1)
if value >= 0:
    return (value + half) >> shift
else:
    return -(((-value) + half) >> shift)
```

This is **round-to-nearest, ties away from zero**. It is **not** banker's
rounding and **not** a plain arithmetic right shift.

Equivalent statement used in comments:

```text
requantized ≈ round(acc * multiplier / 2^shift)
```

with the rounding defined above on the integer product.

---

## 5. Saturation

```text
if v >  127: v =  127
if v < -128: v = -128
```

Casts to `int8` occur only after clipping (no silent wraparound).

---

## 6. Accumulator width analysis

Conservative bound (symmetric zp=0):

```text
|acc| ≤ N_mac * 128 * 127 + |bias|_max
```

| Layer | N_mac | Bound (order) | Observed (test) | Fits int32? |
| --- | --- | --- | --- | --- |
| Conv1 | 27 | ~453973 + \|bias\| | ≤ 103360 | yes |
| Conv2 | 144 | ~2.35e6 + \|bias\| | ≤ 79522 | yes |

RTL uses signed **int32** accumulators matching software.

---

## 7. Memory layouts

### Activations (NCHW, C-order flatten)

```text
addr = channel * H * W + row * W + column
```

For `[3,32,32]`:

```text
addr = channel * 1024 + row * 32 + column
```

### Convolution weights `[O, I, kH, kW]`

```text
addr = oc * (I*kH*kW) + ic * (kH*kW) + kr * kW + kc
```

For Conv1 (`I=3`, `kH=kW=3`):

```text
addr = oc * 27 + ic * 9 + kr * 3 + kc
```

### Encoding

* Two's-complement
* One value per line in `.mem` / `.hex` for `$readmemh`
* INT8: 2 hex digits; INT32: 8 hex digits
* Little-endian host when packing multi-byte in NumPy; HEX files are
  **digit strings of the two's-complement bit pattern**, not host byte dumps

---

## 8. Conv2 / classifier (summary only)

Same arithmetic family as Conv1. Full channel engines are **out of scope** for
the single-output milestone. See `docs/fixed_point_specification.md`.

---

## 9. Files for the streamed single-output milestone

| Artifact | Path |
| --- | --- |
| Vector export | `tools/export_rtl_vectors.py` |
| Vectors | `vectors/conv1_single_output/` |
| MAC | `rtl/int8_mac.sv` |
| Requant | `rtl/requantize.sv` |
| ReLU | `rtl/relu_int8.sv` |
| Saturate | `rtl/saturate_int8.sv` |
| One-output engine (streamed operands) | `rtl/conv_single_output.sv` |
| Testbenches | `tb/` |
| Runner | `tools/run_rtl_tests.py` / `make test-rtl-single-output` |

## 10. Memory-driven single-output milestone

Replaces the pre-extracted 27-element patch with full activation/weight/bias
memories plus an address generator. Still **one** output coordinate only.

| Artifact | Path |
| --- | --- |
| Memory vector export | `tools/export_conv1_memory_vectors.py` |
| Memories / traces | `vectors/conv1_memory/` |
| Address generator | `rtl/conv_address_generator.sv` |
| Sync ROM models | `rtl/int8_sync_rom.sv`, `rtl/int32_sync_rom.sv` |
| Memory-driven engine | `rtl/conv1_memory_single_output.sv` |
| Addressing doc | `docs/conv1_memory_addressing.md` |
| Runner | `make test-rtl-memory-output` |

See `docs/conv1_memory_addressing.md` for address equations, padding, FSM
timing, and how to debug a failed MAC.

## 11. One-channel controller milestone

Sweeps all `32×32` spatial outputs of **output channel 0** into a 1024-entry
INT8 RAM. See `docs/conv1_channel_architecture.md`.

| Artifact | Path |
| --- | --- |
| Channel vector export | `tools/export_conv1_channel_vectors.py` |
| Expected channel map | `vectors/conv1_channel/` |
| Output RAM | `rtl/int8_sync_ram.sv` |
| Controller | `rtl/conv1_channel_controller.sv` |
| Top wrapper | `rtl/conv1_channel_top.sv` |
| Testbench | `tb/tb_conv1_channel.sv` |
| Runner | `make test-rtl-conv1-channel` |

## 12. Full Conv1 layer milestone

Sweeps **all 16** trained output channels (`16×32×32 = 16384` activations).
See `docs/conv1_full_layer_architecture.md`.

| Artifact | Path |
| --- | --- |
| Full tensor export | `tools/export_conv1_full_vectors.py` |
| Expected tensor | `vectors/conv1_full/` |
| Layer controller | `rtl/conv1_layer_controller.sv` |
| Layer top | `rtl/conv1_layer_top.sv` |
| Testbench | `tb/tb_conv1_full_layer.sv` |
| Runner | `make test-rtl-conv1-full` |

## 13. MaxPool1 milestone

2 × 2 stride-2 signed INT8 max pool over the full Conv1 tensor
(`16 × 32 × 32 → 16 × 16 × 16 = 4096`). No requantization and no extra ReLU.
See `docs/maxpool1_architecture.md`.

| Artifact | Path |
| --- | --- |
| Pool1 vector export | `tools/export_pool1_vectors.py` |
| Expected tensor | `vectors/pool1/` |
| Address generator | `rtl/maxpool2x2_address_generator.sv` |
| Signed max-of-four | `rtl/max4_int8.sv` |
| Controller | `rtl/maxpool1_controller.sv` |
| Top wrapper | `rtl/maxpool1_top.sv` |
| Testbenches | `tb/tb_maxpool*.sv` |
| Runner | `make test-rtl-maxpool1` |

## 14. Conv2 layer milestone

Memory-driven Conv2 over Pool1 (`16 × 16 × 16 → 32 × 16 × 16 = 8192`).
144 MACs per output. See `docs/conv2_architecture.md`.

| Artifact | Path |
| --- | --- |
| Vector export | `tools/export_conv2_vectors.py` |
| Expected tensor | `vectors/conv2/` |
| Address generator | `rtl/conv2_address_generator.sv` |
| Single-output engine | `rtl/conv2_memory_single_output.sv` |
| Layer controller | `rtl/conv2_layer_controller.sv` |
| Layer top | `rtl/conv2_layer_top.sv` |
| Testbenches | `tb/tb_conv2_*.sv` |
| Runner | `make test-rtl-conv2` |

## 15. MaxPool2 milestone

2 × 2 stride-2 signed INT8 max pool over Conv2
(`32 × 16 × 16 → 32 × 8 × 8 = 2048`). No requantization and no extra ReLU.
See `docs/maxpool2_architecture.md`.

| Artifact | Path |
| --- | --- |
| Pool2 vector export | `tools/export_pool2_vectors.py` |
| Expected tensor | `vectors/pool2/` |
| Address generator | `rtl/maxpool2_address_generator.sv` |
| Signed max-of-four | `rtl/max4_int8.sv` (reused) |
| Controller | `rtl/maxpool2_controller.sv` |
| Top wrapper | `rtl/maxpool2_top.sv` |
| Testbenches | `tb/tb_maxpool2_*.sv` |
| Runner | `make test-rtl-maxpool2` |

## 16. Global Average Pooling milestone

Per-channel average of Pool2 (`32 × 8 × 8 → 32`). Exact Python path:

```text
sum = Σ 64 INT8 values
avg = saturate(round_divide_int(sum, 64))   # ties away from zero
gap = requantize_int32(avg, mult=1759306569, shift=29)  # INT8 flatten scale
```

See `docs/global_average_pooling_architecture.md`.

| Artifact | Path |
| --- | --- |
| GAP vector export | `tools/export_gap_vectors.py` |
| Expected vectors | `vectors/gap/` |
| Average arithmetic | `rtl/gap_average.sv` |
| Requantization | `rtl/requantize.sv` (reused) |
| Controller | `rtl/global_average_pool_controller.sv` |
| Output storage | `rtl/gap_output_storage.sv` |
| Top wrapper | `rtl/global_average_pool_top.sv` |
| Testbenches | `tb/tb_gap_*.sv`, `tb/tb_global_average_pool_full.sv` |
| Runner | `make test-rtl-gap` |

## 17. Fully Connected classifier milestone

Linear **32 → 5** over verified GAP / flatten features.

```text
acc[c] = bias[c] + Σ gap[i] * weight[c][i]     # INT32, ZP=0
logit[c] = requantize_int32(acc[c], mult[c], shift[c])  # INT32 scores, no ReLU
```

See `docs/fully_connected_architecture.md`.

| Artifact | Path |
| --- | --- |
| FC vector export | `tools/export_fc_vectors.py` |
| Expected vectors | `vectors/fc/` |
| Address generator | `rtl/fc_address_generator.sv` |
| MAC | `rtl/int8_mac.sv` (reused) |
| Output postprocess | `rtl/fc_output_postprocess.sv` |
| Class engine | `rtl/fully_connected_class_engine.sv` |
| Layer controller | `rtl/fully_connected_layer_controller.sv` |
| Logit storage | `rtl/logit_storage.sv` |
| Top wrapper | `rtl/fully_connected_top.sv` |
| Testbenches | `tb/tb_fc_*.sv`, `tb/tb_fully_connected_*.sv` |
| Runner | `make test-rtl-fc` |

## 18. Signed argmax milestone

Signed argmax over five INT32 FC logits → `predicted_class[2:0]`.
Tie-breaking: lowest index (`strict >`), matching `numpy.argmax`.
See `docs/argmax_architecture.md`.

| Artifact | Path |
| --- | --- |
| Argmax vector export | `tools/export_argmax_vectors.py` |
| Expected vectors | `vectors/argmax/` |
| Combinational core | `rtl/signed_argmax5.sv` |
| Sequential controller | `rtl/signed_argmax5_controller.sv` |
| Top wrapper | `rtl/argmax_top.sv` |
| Testbenches | `tb/tb_signed_argmax5.sv`, `tb/tb_argmax_*.sv` |
| Runner | `make test-rtl-argmax` |


