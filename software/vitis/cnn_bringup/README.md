# Vitis `cnn_bringup` application sources (Git source of truth)

This directory is the **Git-controlled** source for the Vitis application
component `cnn_bringup` targeting:

```text
Platform:  zynq_cnn_platform
Domain:    standalone_ps7_cortexa9_0
OS:        standalone
Processor: ps7_cortexa9_0
```

Do **not** edit a separate copied `main.c` under `vitis_workspace/`.

## Workflow

```text
Mac / Cursor
  edit software/vitis/cnn_bringup/*
  git add / commit / push

Windows
  git pull
  Vitis references these files (no copy)
  Build → Run on Arty Z7-10
```

## Files

| File | Role |
|------|------|
| `main.c` | UART bring-up: load golden image, START, poll DONE, compare |
| `cnn_accelerator.c` / `.h` | AXI4-Lite driver (`cnn_axi_ctrl`) |
| `known_test_image.h` | Exact 3072 INT8 CHW vector + expected logits/class |

Golden input source: `vectors/end_to_end/input_image.mem`  
Expected outputs: `vectors/end_to_end/inference_expected.json`

## Windows Vitis: import / retarget sources

1. Open the **cnn_bringup** application component.
2. Remove the old workspace-local UART-only `main.c` from the application
   source list (do not keep two `main.c` files).
3. Add these files from the repo clone, e.g.:

```text
C:\Users\Abhik\Documents\Zynq-SoC-CNN-Accelerator\software\vitis\cnn_bringup\

  main.c
  cnn_accelerator.c
  cnn_accelerator.h
  known_test_image.h
```

4. When Vitis asks **Copy sources to component** → **UNCHECK** it so Vitis
   references the repository files directly.
5. Build the application and run on the board (UART console).

## Base address

`main.c` defaults to:

```text
CNN_BASE_ADDRESS = 0x43C00000
```

matching the Vivado Address Editor assignment for `cnn_axi_ctrl_0/S00_AXI`.
After the first Windows build, prefer the generated `xparameters.h` macro once
its exact name is confirmed; override with `-DCNN_BASE_ADDRESS=...` if needed.

## What this test does

1. Load 3072 INT8 activations via packed `INPUT_WRITE` @ `0x2C`
2. Write `CONTROL.START`
3. Poll sticky `STATUS.DONE` with timeout
4. Read logits / class / cycle count
5. Compare to Python/RTL golden → print `OVERALL: PASS/FAIL`

No RTL changes. No BSP / workspace metadata lives in this directory.
