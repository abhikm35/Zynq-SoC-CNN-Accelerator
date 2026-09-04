# RTL — Conv1 single-output bit-exact milestone

Working modules for this phase (flat paths required by the milestone):

| Module | File |
| --- | --- |
| INT8 MAC | `rtl/int8_mac.sv` |
| Requantizer | `rtl/requantize.sv` |
| ReLU | `rtl/relu_int8.sv` |
| Saturate | `rtl/saturate_int8.sv` |
| One Conv1 pixel engine | `rtl/conv_single_output.sv` |

Empty placeholder files under `rtl/convolution/`, `rtl/activation/`, etc. are
reserved for later full-accelerator work and are intentionally unused here.

## Run tests

```bash
make test-rtl-single-output
# or
PYTHONPATH=. python -m tools.run_rtl_tests
```

Requires Verilator 4.x+ on `PATH`.
