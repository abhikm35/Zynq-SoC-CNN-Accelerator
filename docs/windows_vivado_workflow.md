# Windows Vivado Workflow (Git as source of truth)

This repository is edited primarily in Cursor on macOS / Linux. Synthesis and
implementation run on a separate Windows PC with Vivado. **Git is authoritative.**
Do not maintain a second manually edited RTL copy inside a Vivado project folder.

```text
                  Git Remote
                /            \
               /              \
          push/pull         push/pull
             /                  \
            v                    v

    Mac / Linux              Windows PC
       Cursor                   Vivado
          |                       |
          |                       |
          v                       v

       rtl/*.sv  <same Git>   rtl/*.sv
                                    |
                                    v
                              Vivado project
                                    |
                          +---------+---------+
                          |                   |
                          v                   v
                      synthesis        implementation
                          |                   |
                          +---------+---------+
                                    |
                                    v
                              timing/reports
                                    |
                                    v
                                bitstream
```

## Current synthesis top

```text
cnn_accelerator_synth_wrapper
```

This is a **tiny pinout wrapper** around the verified
`cnn_accelerator_shared_compute_top`. The full top exposes hundreds of
TB/debug ports and cannot place on xc7z010 (~230 user IOs). The wrapper
keeps `clk/rst/start/busy/done/predicted_class/maximum_logit` only.

Architecture underneath (verified in RTL sim):

```text
2 ping-pong activation RAMs
1 shared convolution engine  (Conv1 + Conv2)
1 shared max-pool engine     (Pool1 + Pool2)
GAP -> FC -> Argmax
```

FPGA part: **xc7z010clg400-1** (Arty Z7-10 / Zynq-7000).

Project scripts recreate Vivado under:

```text
vivado_build/cnn_accelerator/
```

Reports go to:

```text
vivado_output/reports/
```

Those directories are **local build outputs** (gitignored). Dispose and recreate anytime.

---

## First-time Windows setup

1. Install Git for Windows.
2. Install Vivado (with Zynq-7000 device support).
3. Clone **the same repository** used by Cursor (once a remote exists):

```bash
git clone <repository-url>
cd zynq-edge-ai-classifier
```

4. Open a terminal where `vivado` is on `PATH` (Vivado Command Prompt / Tcl Shell).
5. Create the project:

```text
scripts\windows\create_vivado_project.bat
```

6. Open the GUI project:

```text
scripts\windows\open_vivado_project.bat
```

7. Or run synthesis / implementation / reports in batch:

```text
scripts\windows\run_vivado_analysis.bat
```

### Note if this tree is not yet a Git repo

On the Cursor/Linux side, initialize and push once before relying on the
Mac↔Windows sync story:

```bash
cd /path/to/zynq-edge-ai-classifier
git init
git add -A
git status   # review; do not add vivado_build/ or secrets
git commit -m "Initial CNN accelerator with Vivado workflow scripts"
# create remote on GitHub/GitLab, then:
# git remote add origin <url>
# git push -u origin HEAD
```

This documentation does not create remotes or push for you.

---

## Normal Mac / Linux development workflow

```bash
git pull
```

Edit RTL / tests / scripts in Cursor.

Run regressions (example):

```bash
export PATH=~/tools/verilator-5.028/bin:$PATH
export VERILATOR_ROOT=~/tools/verilator-5.028
make test-rtl-shared-pool
```

Then:

```bash
git status
git add <specific files>
git commit -m "Describe the change"
git push
```

---

## Normal Windows Vivado workflow

**Always pull before synthesizing:**

```bash
git pull
```

Then either:

```text
scripts\windows\open_vivado_project.bat
```

and run Synthesis / Implementation in the GUI, or:

```text
scripts\windows\run_vivado_analysis.bat
```

If the project is stale or missing after a major RTL reshape:

```text
scripts\windows\create_vivado_project.bat
```

(recreates `vivado_build/cnn_accelerator` from current repo files)

---

## Windows-side source changes

If you change tracked files on Windows (XDC, Tcl, RTL):

```bash
git status
git add <specific files>
git commit -m "Update Vivado constraints"
git push
```

Then on Mac/Linux:

```bash
git pull
```

Never commit:

```text
vivado_build/
vivado_output/
.Xil/
*.jou / *.log / *.runs / *.cache / ...
```

---

## Protect against stale Windows RTL

Failure mode:

```text
Mac Cursor:  new code committed
Windows:     old clone
Vivado:      synthesizes old code
```

Prevention:

```text
Always git pull on Windows before synthesis.
Always git status before commit/push on either machine.
```

Verify both machines share the same tip:

```bash
git status
git log -1 --oneline
```

Example match:

```text
Mac/Linux:  abc1234 Shared pool engine
Windows:    abc1234 Shared pool engine
```

---

## What Vivado synthesizes vs simulation-only

### Included (synthesis)

- Explicit list in `scripts/vivado_sources.tcl`
- Top: `cnn_accelerator_shared_compute_top`
- Model ROM `.mem` files (weights / biases / quant params)
- `rtl/constraints/zynq_edge_ai.xdc`

### Excluded (intentionally)

- All `tb/*.sv` testbenches
- Legacy tops: `cnn_accelerator_top`, `cnn_accelerator_pingpong_top`,
  `cnn_accelerator_shared_conv_top`
- Separate Conv1/Conv2 / MaxPool1/MaxPool2 controller modules (kept for
  regression; shared engines replace them in the synth top)
- Simulation golden tensors such as `*_expected.mem`, end-to-end images

### Model parameters (tracked; needed for FPGA)

```text
vectors/conv1_memory/conv1_weights.mem
vectors/conv1_memory/conv1_biases.mem
vectors/conv1_memory/conv1_multipliers.mem
vectors/conv1_memory/conv1_shifts.mem
vectors/conv2/conv2_weights.mem
vectors/conv2/conv2_biases.mem
vectors/conv2/conv2_multipliers.mem
vectors/conv2/conv2_shifts.mem
vectors/fc/fc_weights.mem
vectors/fc/fc_biases.mem
vectors/fc/fc_multipliers.mem
vectors/fc/fc_shifts.mem
```

Loaded via `$readmemh` in ROM modules. The create-project script also sets
absolute string generics so paths resolve on any clone location.

### Simulation golden data (not for synthesis)

```text
vectors/end_to_end/*_expected.mem
vectors/*/pool*_expected.mem
... other expected / trace files
```

---

## Clock constraint status

Port `clk` has a **provisional** `create_clock -period 10.000` (100 MHz) in
`rtl/constraints/zynq_edge_ai.xdc` so standalone timing reports are possible.

This does **not** mean the board PL oscillator is wired, and it is **not** a
Zynq PS FCLK constraint. Revise before claiming timing closure for the
integrated PS design.

---

## Repository-relative Tcl

All Vivado Tcl scripts resolve:

```tcl
set script_dir [file normalize [file dirname [info script]]]
set repo_root  [file normalize [file join $script_dir ".."]]
```

No `/Users/...`, `/nethome/...`, or `C:\Users\...` hardcoded paths.

---

## Vivado execution note

These scripts are **statically reviewed** on the Linux/Cursor side. Actual
synthesis / implementation / timing must be run on the Windows machine with
Vivado installed. Do not treat this documentation as proof of timing closure.

---

## Next milestone (after Windows synth works)

Characterize the clean RTL accelerator:

- LUT / FF / BRAM / DSP
- Whether both ping-pong RAMs infer BRAM
- Hierarchical utilization
- WNS / TNS / critical path
- Whether the target clock closes timing

Only then begin AXI / Zynq processor integration.
