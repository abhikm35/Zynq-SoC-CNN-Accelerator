# Vivado / Arty Z7-10 constraints for the standalone CNN accelerator top.
#
# Synthesis top: cnn_accelerator_shared_compute_top
# Part:          xc7z010clg400-1
#
# IMPORTANT — clock source is not yet finalized for Zynq PS integration:
#   - Board PL oscillator (100 MHz) vs
#   - Zynq PS FCLK (programmable)
# are different architectures. Do not confuse them.
#
# The create_clock below is a PROVISIONAL 100 MHz constraint on the RTL `clk`
# port so standalone synthesis/implementation can produce timing reports.
# When the accelerator is connected to a PS FCLK (or another PL clock),
# revise or replace this constraint to match the actual clock source and period.

## -----------------------------------------------------------------------------
## Provisional PL clock (standalone characterization only)
## -----------------------------------------------------------------------------
create_clock -period 10.000 -name cnn_clk -waveform {0.000 5.000} [get_ports clk]

## -----------------------------------------------------------------------------
## TODO (Zynq PS integration — not done in this milestone)
## -----------------------------------------------------------------------------
## When wrapping this RTL under AXI / PS:
##   1. Identify the real clock net (e.g. FCLK_CLK0).
##   2. Replace or retarget create_clock above.
##   3. Add false paths / CDC constraints for PS↔PL if needed.
##   4. Constrain AXI and host ports once they exist.
##
## Host / debug ports (start, busy, done, predicted_class, …) are currently
## unconstrained beyond the clock. That is intentional until the I/O wrapper
## and board pinout are defined.
