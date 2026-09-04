# Vivado directory (legacy placeholders)

Active, portable Vivado automation lives under:

```text
scripts/create_vivado_project.tcl
scripts/run_vivado_analysis.tcl
scripts/vivado_sources.tcl
scripts/windows/*.bat
```

See `docs/windows_vivado_workflow.md`.

Generated projects belong in gitignored `vivado_build/` (not here).
The scripts in `vivado/scripts/` are empty legacy stubs retained so existing
paths do not break; prefer the `scripts/` entry points above.
