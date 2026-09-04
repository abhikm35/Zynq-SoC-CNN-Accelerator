# Verification Plan

## Golden-reference chain

```text
Float32 PyTorch
  -> Float32 NumPy
  -> INT8 NumPy
  -> exported integer vectors
  -> future SystemVerilog unit tests
  -> future complete accelerator test
```

## Floating-point checks

1. Checkpoint loads into a fresh `TinyCNN` with exact state-dict shapes.
2. Official filtered GTSRB test evaluation records accuracy and confusion matrix.
3. A deterministic 20-image reference set is exported.
4. Every intermediate tensor is saved for each reference image.
5. NumPy operators are unit-tested with hand-calculated examples.
6. NumPy TinyCNN matches PyTorch intermediates within `rtol=1e-5`, `atol=1e-5`.

## INT8 checks

1. Calibration uses a deterministic validation subset (never the official test split).
2. Quantization primitives enforce positive finite scales and explicit saturation.
3. Rounding is round-to-nearest, ties away from zero (hand-tested).
4. Integer operators match hand-calculated convolution, pool, GAP, and linear examples.
5. Repeated integer inference is bitwise deterministic.
6. Exported INT8 test vectors match a fresh integer forward pass exactly.
7. FPGA HEX files round-trip to the original arrays.
8. Float32 versus INT8 reference comparison records dequantization error and class agreement.
9. Full filtered test-set INT8 accuracy is measured and compared to float32.
10. Theoretical and empirical accumulator overflow analysis is recorded.

## Future hardware checks

* RTL unit tests versus exported INT8 vectors
* AXI-Stream and AXI-Lite protocol assertions
* End-to-end FPGA versus software comparison
