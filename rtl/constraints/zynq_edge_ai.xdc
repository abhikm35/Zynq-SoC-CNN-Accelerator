# Vivado / Arty Z7-10 constraints for the standalone CNN accelerator top.
#
# Synthesis top: cnn_accelerator_synth_wrapper
#   (wraps cnn_accelerator_shared_compute_top with a tiny pinout)
# Part:          xc7z010clg400-1
#
# IMPORTANT — clock source is not yet finalized for Zynq PS integration:
#   - Board PL oscillator (100 MHz) vs
#   - Zynq PS FCLK (programmable)
# are different architectures. Do not confuse them.
#
# The create_clock below is a PROVISIONAL characterization constraint on the
# RTL `clk` port (~83.3 MHz / 12 ns). After pipelining requantize, the design
# was ~1 ns short of 100 MHz (WNS ≈ -0.95 ns at 10 ns); 12 ns closes timing
# with margin for standalone reports. When connected to a PS FCLK (or another
# PL clock), revise or replace this to match the real source and period.
#
# Pins are intentionally unconstrained (no PACKAGE_PIN). Vivado will
# auto-assign IOs for resource/timing characterization only. Add real
# board pin LOC constraints only when integrating to the Arty Z7 / PS design.

## -----------------------------------------------------------------------------
## Provisional PL clock (standalone characterization only)
## -----------------------------------------------------------------------------
create_clock -period 12.000 -name cnn_clk -waveform {0.000 6.000} [get_ports clk]

## -----------------------------------------------------------------------------
## TODO (Zynq PS integration — not done in this milestone)
## -----------------------------------------------------------------------------
## When wrapping this RTL under AXI / PS:
##   1. Identify the real clock net (e.g. FCLK_CLK0).
##   2. Replace or retarget create_clock above.
##   3. Add false paths / CDC constraints for PS↔PL if needed.
##   4. Constrain AXI and host ports once they exist.
