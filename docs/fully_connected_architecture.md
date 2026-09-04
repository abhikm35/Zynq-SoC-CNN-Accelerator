# Fully Connected Classifier Architecture

The FC layer maps the verified GAP / flatten vector to five class logits.
Hardware matches the Python integer golden model bit for bit.

> Trained model note: Linear is **32 → 5** (GAP length 32). Earlier design
> sketches with 16 inputs / 80 weights are obsolete.

---

## Purpose

- Produce five comparable class scores from 32 GAP features
- Preserve signed INT32 logits for the future argmax stage
- No floating-point arithmetic in RTL

## Tensor shapes

| Tensor | Shape | Count | Datatype |
| --- | --- | --- | --- |
| GAP / flatten input | `[32]` | 32 | signed INT8 |
| Weights | `[5, 32]` | 160 | signed INT8 |
| Biases | `[5]` | 5 | signed INT32 |
| Accumulators | `[5]` | 5 | signed INT32 |
| Logits / scores | `[5]` | 5 | signed INT32 |

## Exact Python equation

From `integer_linear` + `requantize_int32` (`zero_point == 0` for input and weights):

```text
accumulator[class] =
    bias[class]
  + Σ_{i=0..31} gap[i] * weight[class][i]

logit[class] =
    saturate_int32(
      round(accumulator[class] * multiplier[class] / 2^shift[class])
    )
```

Rounding is **ties away from zero** (`rounding_right_shift`).
Saturation limits are the full INT32 range (not INT8).
**No ReLU** is applied to logits (negative scores are valid).

### Quantization metadata

| Property | Value |
| --- | --- |
| GAP datatype / width | signed INT8 / 8 |
| GAP scale | flatten / GAP scale (~0.0534) |
| GAP zero point | 0 |
| Weight datatype / width | signed INT8 / 8 |
| Weight scales | per-class `[5]` |
| Weight zero point | 0 |
| Bias datatype / width | signed INT32 / 32 (already in accumulator units) |
| Accumulator width | signed INT32 |
| MACs per class | 32 |
| Total MACs | 160 |
| Logit datatype / width | signed INT32 / 32 |
| Logit scale | shared `classifier_output` (~0.1922) |
| Logit zero point | 0 |
| Requantization | per-class multiplier / shift (exported) |
| ReLU on logits | **no** |

Worst-case product: `±128 * ±127` fits INT16.
Worst-case 32-term sum + bias fits INT32 for the trained model (Python asserts no INT32 overflow).

## Address equations

```text
gap_read_address    = input_index

fc_weight_address   = class_index * 32 + input_index

fc_bias_address     = class_index

logit_write_address = class_index
```

## Synchronous memory timing

All ROMs / RAMs use **1-cycle** sync read latency:

```text
ISSUE_BIAS → WAIT_BIAS → LOAD_BIAS
ISSUE_OP   → WAIT_MAC     (×32)
POST_PROCESS → DONE
```

## Controller FSM

Layer:

```text
IDLE → START_CLASS → WAIT_CLASS → WRITE_LOGIT → ADVANCE → (next | DONE)
```

Single-class engine:

```text
IDLE → ISSUE_BIAS → WAIT_BIAS → LOAD_BIAS
    → ISSUE_OP → WAIT_MAC (×32)
    → POST_PROCESS → DONE
```

## Counts

```text
FC inputs:     32 GAP values
FC outputs:    5 class logits
Weights:       5 x 32 = 160
Biases:        5
MACs per class: 32
Total MACs:    160
```

## Modules

| Module | Role |
| --- | --- |
| `fc_address_generator` | Combinational addresses |
| `int8_mac` | Signed MAC (**reused**) |
| `fc_output_postprocess` | INT32-saturating requant |
| `fully_connected_class_engine` | One class |
| `fully_connected_layer_controller` | Five classes |
| `logit_storage` | 5×INT32 storage |
| `fully_connected_top` | ROMs + engine + controller |
| `int8_sync_rom` / `int32_sync_rom` | Parameter / feature memories |

## Regenerate vectors

```bash
make export-fc
# or
PYTHONPATH=. python -m tools.export_fc_vectors
```

## Run tests

```bash
make test-rtl-fc
make test-rtl-conv1-full   # full regression
```

## Debug a mismatch

1. Compare `vectors/fc/class_traces/class_N.txt` running accumulators.
2. Confirm GAP input mem matches `vectors/gap/gap_expected.mem`.
3. Check per-class multiplier / shift in `fc_quant_params.json`.
4. Verify no INT8 saturation was applied to logits.

## Next milestone (not this task)

Implement signed argmax across the five verified class logits
(done — see `docs/argmax_architecture.md` / `make test-rtl-argmax`).

