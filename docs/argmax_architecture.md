# Signed Argmax Architecture

Argmax selects the winning traffic-sign class from the five verified FC logits.

> Logits are signed **INT32** FC scores (post-requant). Class IDs are
> `0..4`: stop, yield, no_entry, speed_limit_30, keep_right.

---

## Purpose

- Produce `predicted_class[2:0]` for software / future system integration
- Expose `maximum_logit` for debug
- Preserve all five original logits unchanged for software access

## Inputs / outputs

| Signal | Datatype | Notes |
| --- | --- | --- |
| `logit[0..4]` | signed INT32 | From FC logit storage |
| `predicted_class` | unsigned 3-bit | Range 0..4 |
| `maximum_logit` | signed INT32 | Equals one of the five inputs |

## Exact algorithm

```text
maximum_logit = logit[0]
maximum_index = 0

for index = 1 through 4:
    if logit[index] > maximum_logit:   # strict greater
        maximum_logit = logit[index]
        maximum_index = index
```

### Tie-breaking

Matches `numpy.argmax` / `IntegerTinyCNN.predict`:

```text
The maximum is updated only for a strictly greater value.
Equal values do not replace the previously selected class.
=> lowest class index wins among equal maxima.
```

Example: `[100, 500, 20, 500, -10]` → class **1**.  
Example: `[7, 7, 7, 7, 7]` → class **0**.

## What is not done

- No softmax
- No requantization of logits
- No ReLU
- No floating-point

All comparisons are **signed**.

## Architecture choice

FC stores logits in `logit_storage` / `int32_sync_ram` with **1-cycle sync read**.

| Module | Role |
| --- | --- |
| `signed_argmax5` | Combinational core (directed / random unit tests) |
| `signed_argmax5_controller` | Sequential reader for sync storage |
| `argmax_top` | ROM of verified logits + sequential controller |

**Sequential** path is used for repository-matched storage latency.  
**Combinational** path is used for fast directed and randomized self-checks.

### Sequential FSM

```text
IDLE → ISSUE_READ → WAIT_READ → CAPTURE_AND_COMPARE → ADVANCE → DONE
```

Exactly **5** logit reads. `argmax_done` after logit 4 is compared.  
Typical cycle count from start to done: **17** (5 × ISSUE/WAIT/CAPTURE/ADVANCE + DONE overhead).

### Start / busy / done

- `start` while `busy` does not restart
- `predicted_class` / `maximum_logit` held until next start/reset
- Original logits remain readable when not busy

## Regenerate vectors

```bash
make export-argmax
# or
PYTHONPATH=. python -m tools.export_argmax_vectors
```

## Run tests

```bash
make test-rtl-argmax
make test-rtl-conv1-full
```

## Debug a mismatch

1. Compare against `vectors/argmax/argmax_expected.json` real case.
2. Confirm comparisons use signed `>` (not `>=`).
3. For sequential path, confirm 1-cycle wait before capture.
4. Confirm logits ROM matches `vectors/fc/fc_logits_expected.mem`.

## Next milestone (not this task)

Build and verify the complete end-to-end CNN integration top connecting
Conv1 → MaxPool1 → Conv2 → MaxPool2 → GAP → FC → argmax under one
controller, run one inference from a single start pulse, and keep
existing stage-level wrappers for regression.
