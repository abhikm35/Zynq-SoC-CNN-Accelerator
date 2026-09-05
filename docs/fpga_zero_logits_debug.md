# FPGA bring-up: all logits zero, cycle_count exact

## Observed symptom

```text
DONE works
STATUS works
cycle_count == 3613312   (exact RTL golden)
all five logits == 0
maximum_logit == 0
predicted_class == 0     (NOT meaningful — argmax of all-zeros)
```

## Root cause (repository evidence)

**Most likely: parameter ROM `$readmemh` failed during Vivado BD synthesis.**

Evidence:

1. `int8_sync_rom` / `int32_sync_rom` load via `$readmemh(MEM_FILE, mem)`.
   If the file is not found, the `initial` block does nothing → arrays stay
   at default **0** in synthesis.

2. Standalone project (`scripts/create_vivado_project.tcl`) sets **absolute**
   `generic` paths on `cnn_accelerator_synth_wrapper`.

3. Zynq Block Design uses **Module Reference** of `cnn_accelerator_bd_wrapper`,
   which previously instantiated the synth wrapper with **relative** defaults
   such as `vectors/conv1_memory/conv1_weights.mem`. Vivado's synth CWD is
   not the repo root → `$readmemh` fails.

4. FC biases in-repo are **non-zero**. If only the input image were wrong but
   weights/biases were loaded, logits would generally **not** all be exactly 0.
   All-zero logits strongly indicate **zero parameter ROMs** (or a disconnected
   logit bus — less likely given cycle_count AXI path works).

5. Exact cycle_count proves the **control FSM** ran the full schedule. It does
   **not** prove weights or input pixels are correct.

## Decision tree

```text
Board symptoms:
DONE works
cycle count exact
all logits zero

Step 1:
Are parameter .mem files found during BD / Module Reference synthesis?
    (search synth log for readmemh / cannot open / conv1_weights.mem)
    NO  -> run scripts/apply_bd_mem_generics.tcl
           ensure 12 .mem files in project
           rebuild bitstream + export XSA + refresh Vitis
    YES -> Step 2

Step 2:
Does RAM A contain known input after AXI load?
    (optional future AXI readback; not required if Step 1 was broken)
    NO  -> debug AXI INPUT_WRITE / input_write_* / ownership mux
    YES -> Step 3

Step 3:
Does selected Conv1 output match golden?
    NO  -> debug Conv1 / remaining parameter memories
    YES -> Step 4

Step 4:
Check later stage checkpoints (Pool / Conv2 / GAP / FC)

Step 5:
Check final logit -> AXI readback path
```

## The 12 required model `.mem` files

```text
vectors/conv1_memory/conv1_weights.mem      (432)
vectors/conv1_memory/conv1_biases.mem       (16)
vectors/conv1_memory/conv1_multipliers.mem  (16)
vectors/conv1_memory/conv1_shifts.mem       (16)
vectors/conv2/conv2_weights.mem             (4608)
vectors/conv2/conv2_biases.mem              (32)
vectors/conv2/conv2_multipliers.mem         (32)
vectors/conv2/conv2_shifts.mem              (32)
vectors/fc/fc_weights.mem                   (160)
vectors/fc/fc_biases.mem                    (5)
vectors/fc/fc_multipliers.mem               (5)
vectors/fc/fc_shifts.mem                    (5)
```

(Trained 16/32 channel sizes — not the obsolete 8/16 sketch.)

Audit locally:

```bash
python3 tools/audit_vivado_memory_files.py
```

## AXI input path (secondary — currently looks correct in RTL)

```text
ARM Xil_Out32(BASE+0x2C, packed)
  -> cnn_axi_ctrl INPUT_WRITE
  -> cnn_input_write_{enable,address,data}
  -> bd_wrapper.input_write_*
  -> synth_wrapper
  -> shared_compute_top: when ~busy && input_write_enable
       write Activation RAM A[address]
```

- Addresses 0..3071 accepted; busy ignores WRITE.
- `activation_ram` has **no** array reset — START does not clear RAM A.
- Preload then START is the correct software order.

## Windows Vivado inspection

1. Open the Zynq Vivado project.
2. Open the synthesis log for the Module Reference / BD synth run.
3. Search for:

```text
readmemh
conv1_weights.mem
conv2_weights.mem
fc_weights.mem
cannot open
failed
warning
memory initialization
```

4. Sources → confirm all 12 model `.mem` files are listed.
5. Apply absolute generics to the Module Reference:

```tcl
cd <repo_root>
source scripts/apply_bd_mem_generics.tcl
cnn_apply_bd_mem_generics
validate_bd_design
save_bd_design
```

6. Rebuild: synth → impl → timing (~83.333 MHz) → bitstream → **export new XSA**.
7. Vitis: refresh/update platform from new XSA → rebuild `cnn_bringup` → run.

## First board test after fix

Same `cnn_bringup` golden test. Expect:

```text
logits: 25, -24, -18, -21, -57
max_logit: 25
class: 0
cycle_count: 3613312
OVERALL: PASS
```

## What not to do yet

- Do not retrain / requantize
- Do not add AXI BRAM / DMA
- Do not add RAM readback until Step 1 (MEM init) is proven fixed
