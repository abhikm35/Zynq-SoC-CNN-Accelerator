# Software: training, float32 export, and INT8 golden model

This directory contains the PyTorch software path and NumPy golden models for the
frozen `TinyCNN`.

## Frozen architecture

```text
Input [N,3,32,32]
-> Conv1 3->16 (3x3,s1,p1,bias=False) -> BN1 -> ReLU -> MaxPool
-> Conv2 16->32 (3x3,s1,p1,bias=False) -> BN2 -> ReLU -> MaxPool
-> GlobalAvgPool -> Flatten[N,32] -> Linear 32->5
```

Parameter count: **5,301** (BatchNorm folded for INT8/FPGA export).

Class order:

```text
0 stop
1 yield
2 no_entry
3 speed_limit_30
4 keep_right
```

## Inspect / evaluate float32

```bash
python -m software.models.inspect_trained_model
python -m software.training.evaluate --mode float32
```

## Export float32 weights and reference activations

```bash
python -m software.export.export_test_vectors --force
python -m software.inference.compare_inference --mode float32
```

## INT8 calibration, quantization, and evaluation

```bash
python -m software.quantization.calibrate
python -m software.quantization.quantize --evaluate
python -m software.inference.compare_inference --mode int8
python -m software.training.evaluate --mode int8
python -m software.quantization.overflow_analysis
```

Useful options:

```bash
python -m software.quantization.quantize --force --calibration-samples 500
python -m software.quantization.quantize --evaluate --checkpoint software/checkpoints/tiny_cnn_best.pth
```

INT8 export root:

```text
software/exported_model/int8/
  weights/ biases/ scales/ requantization/ metadata/ test_vectors/ fpga/
```

## Offline tests

```bash
pytest software/tests/test_quantization.py -v
pytest software/tests/test_integer_inference.py -v
pytest software/tests/test_fpga_export.py -v
pytest software/tests/test_quantized_regression.py -v
pytest software/tests -v
```

## Notes

* INT8 inference is integer-only after input quantization.
* NumPy timing is a software reference only; it is not FPGA performance.
* SystemVerilog / Zynq integration is intentionally deferred.
* Official GTSRB test split is evaluation-only and is never used for calibration.
