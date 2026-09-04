# create_vivado_project.tcl
# Recreate the Vivado project from repository RTL (no duplicate RTL copy).
#
# Usage (from a shell where Vivado is on PATH):
#   vivado -mode batch -source scripts/create_vivado_project.tcl
#
# Or from Vivado Tcl console after cd to repo root:
#   source scripts/create_vivado_project.tcl

set script_dir [file normalize [file dirname [info script]]]
set repo_root  [file normalize [file join $script_dir ".."]]

source [file join $script_dir vivado_sources.tcl]
cnn_assert_sources_exist

set proj_dir  [file normalize [file join $repo_root vivado_build cnn_accelerator]]
set proj_name "cnn_accelerator"

puts "================================================================"
puts " Creating Vivado project for Zynq CNN accelerator"
puts "================================================================"
puts " repo_root : $repo_root"
puts " proj_dir  : $proj_dir"
puts " part      : $cnn_fpga_part"
puts " top       : $cnn_synth_top"
puts "================================================================"

file mkdir [file dirname $proj_dir]
if {[file exists $proj_dir]} {
    puts "INFO: Removing existing project directory $proj_dir"
    file delete -force $proj_dir
}

create_project $proj_name $proj_dir -part $cnn_fpga_part -force

# Prefer board part if Digilent board files are installed; otherwise part-only.
# Do not fail if board catalog is missing.
if {[llength [get_board_parts -quiet {*arty*z7*10*}]] > 0} {
    set bp [lindex [get_board_parts -quiet {*arty*z7*10*}] 0]
    puts "INFO: Setting board_part $bp"
    set_property board_part $bp [current_project]
} else {
    puts "INFO: Arty Z7-10 board files not found; using part $cnn_fpga_part only"
}

# -------------------------------------------------------------------------
# RTL sources (repository files referenced in place — not copied)
# -------------------------------------------------------------------------
add_files -fileset sources_1 $cnn_rtl_sources
foreach f $cnn_rtl_sources {
    set_property file_type SystemVerilog [get_files $f]
}

# Model parameter .mem files (also kept in-repo; referenced by $readmemh)
add_files -fileset sources_1 $cnn_mem_sources

# Constraints
add_files -fileset constrs_1 $cnn_xdc_sources

set_property top $cnn_synth_top [current_fileset]
update_compile_order -fileset sources_1

# Force absolute MEM paths so $readmemh resolves during synth elaboration
# regardless of Vivado run CWD.
set_property generic $cnn_generic_list [current_fileset]

# Do not add tb/ or Python to synthesis.
# Legacy tops (pingpong / shared_conv / separate-memory) are intentionally
# omitted from sources_1.

set n_rtl  [llength $cnn_rtl_sources]
set n_mem  [llength $cnn_mem_sources]
set n_xdc  [llength $cnn_xdc_sources]
set xpr    [file join $proj_dir ${proj_name}.xpr]

puts "================================================================"
puts " Vivado project created"
puts "================================================================"
puts " project   : $xpr"
puts " part      : $cnn_fpga_part"
puts " top       : $cnn_synth_top"
puts " RTL files : $n_rtl"
puts " MEM files : $n_mem"
puts " XDC files : $n_xdc"
puts "================================================================"
puts "Next:"
puts "  scripts/windows/open_vivado_project.bat"
puts "  or"
puts "  scripts/windows/run_vivado_analysis.bat"
puts "================================================================"
