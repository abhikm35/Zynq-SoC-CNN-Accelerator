#!/usr/bin/env python3
"""Audit CNN parameter .mem files referenced by synthesis RTL.

Does not mutate files. Exit 0 if all expected files exist and are non-empty.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Actual trained sizes (16/32 channels) — not the obsolete 8/16 sketch.
EXPECTED = {
    "vectors/conv1_memory/conv1_weights.mem": 432,
    "vectors/conv1_memory/conv1_biases.mem": 16,
    "vectors/conv1_memory/conv1_multipliers.mem": 16,
    "vectors/conv1_memory/conv1_shifts.mem": 16,
    "vectors/conv2/conv2_weights.mem": 4608,
    "vectors/conv2/conv2_biases.mem": 32,
    "vectors/conv2/conv2_multipliers.mem": 32,
    "vectors/conv2/conv2_shifts.mem": 32,
    "vectors/fc/fc_weights.mem": 160,
    "vectors/fc/fc_biases.mem": 5,
    "vectors/fc/fc_multipliers.mem": 5,
    "vectors/fc/fc_shifts.mem": 5,
}

PARAM_RE = re.compile(
    r'parameter\s+\w*(?:WGT|BIAS|MULT|SHIFT)_MEM\s*=\s*"([^"]+)"'
)
READMEM_RE = re.compile(r"\$readmemh?\s*\(\s*([^,\)]+)")


def collect_rtl_references() -> set[str]:
    refs: set[str] = set()
    for path in (ROOT / "rtl").rglob("*"):
        if path.suffix not in {".sv", ".v"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in PARAM_RE.finditer(text):
            refs.add(m.group(1))
        for m in READMEM_RE.finditer(text):
            raw = m.group(1).strip()
            if raw.startswith('"') and raw.endswith('"'):
                refs.add(raw[1:-1])
    return refs


def line_count(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def main() -> int:
    print(f"repo_root: {ROOT}")
    print()
    print("=== Expected synthesis parameter .mem files ===")
    missing = []
    bad_count = []
    empty = []
    for rel, expect in EXPECTED.items():
        path = ROOT / rel
        if not path.is_file():
            print(f"MISSING  {rel}")
            missing.append(rel)
            continue
        n = line_count(path)
        size = path.stat().st_size
        status = "OK"
        if size == 0:
            status = "EMPTY"
            empty.append(rel)
        elif n != expect:
            status = f"COUNT_MISMATCH got={n} expected={expect}"
            bad_count.append(rel)
        print(f"{status:24s} lines={n:5d} bytes={size:6d}  {rel}")

    print()
    print("=== RTL-referenced *.mem string defaults (synth tree) ===")
    refs = sorted(collect_rtl_references())
    for r in refs:
        if "vectors/" not in r and not r.endswith(".mem"):
            continue
        # Skip obvious TB-only golden patterns when under expected/
        exists = (ROOT / r).is_file() if not Path(r).is_absolute() else Path(r).is_file()
        print(f"{'FOUND' if exists else 'MISSING':8s}  {r}")

    print()
    print("=== Vivado project script coverage ===")
    sources = (ROOT / "scripts/vivado_sources.tcl").read_text()
    for rel in EXPECTED:
        base = Path(rel).name
        listed = base in sources or rel.replace("\\", "/") in sources
        print(f"{'LISTED' if listed else 'NOT_IN_TCL':10s}  {rel}")

    print()
    print("=== Risk note ===")
    print(
        "cnn_accelerator_bd_wrapper defaults use relative vectors/... paths.\n"
        "Standalone create_vivado_project.tcl overrides synth_wrapper with\n"
        "absolute generics. Block Design Module Reference does NOT inherit\n"
        "that override unless scripts/apply_bd_mem_generics.tcl is run.\n"
        "Missing $readmemh init => ROM arrays stay 0 => all logits 0 with\n"
        "correct cycle_count."
    )

    print()
    if missing or empty:
        print("RESULT: FAIL")
        return 1
    if bad_count:
        print("RESULT: WARN (counts differ from expected trained sizes)")
        return 0
    print("RESULT: PASS (all 12 parameter .mem files present and non-empty)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
