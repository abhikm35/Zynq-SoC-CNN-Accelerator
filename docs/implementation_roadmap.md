# Implementation Roadmap

## Completed

* TinyCNN architecture freeze (16/32 channels + BatchNorm)
* Tensor-shape verification
* Floating-point model training
* Best-checkpoint saving
* Class-weighted loss, augmentation, normalization, LR scheduling
* Architecture freeze and trained-model inspection
* Floating-point parameter export
* Deterministic reference test vectors
* Independent NumPy floating-point inference
* PyTorch-versus-NumPy layer comparison
* INT8 calibration
* INT8 quantization and integer golden model
* INT8 accuracy comparison
* Overflow analysis
* FPGA HEX / integer test-vector export

## Current

* Ready for RTL planning against the INT8 arithmetic contract

## Deferred

* RTL MAC unit
* RTL convolution
* RTL pooling
* RTL classifier
* AXI integration
* Zynq deployment
* Ethernet
* HDMI
* Quantization-aware training (not required for current PTQ accuracy)
