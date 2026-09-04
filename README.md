# zynq-edge-ai-classifier

Zynq edge-AI INT8 CNN accelerator for a 5-class traffic-sign classifier.

## Synthesis top (current)

```text
rtl/cnn_accelerator_shared_compute_top.sv
```

Contains: ping-pong activation RAMs, shared Conv engine, shared MaxPool engine,
GAP, FC, Argmax.

## Development split

| Machine | Role |
| --- | --- |
| Mac / Linux + Cursor | Edit RTL, run Verilator / Python regressions, `git push` |
| Windows + Vivado | `git pull`, recreate project, synth / impl / timing |

See **[docs/windows_vivado_workflow.md](docs/windows_vivado_workflow.md)**.

Quick Windows:

```text
scripts\windows\create_vivado_project.bat
scripts\windows\run_vivado_analysis.bat
```

## RTL regression (Linux)

```bash
export PATH=~/tools/verilator-5.028/bin:$PATH
export VERILATOR_ROOT=~/tools/verilator-5.028
make test-rtl-shared-pool
```
