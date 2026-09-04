# run_vivado_analysis.tcl
# Open (or create) the Vivado project and run synth + impl + reports.
#
# Usage:
#   vivado -mode batch -source scripts/run_vivado_analysis.tcl

set script_dir [file normalize [file dirname [info script]]]
set repo_root  [file normalize [file join $script_dir ".."]]

source [file join $script_dir vivado_sources.tcl]

set proj_dir  [file normalize [file join $repo_root vivado_build cnn_accelerator]]
set proj_name "cnn_accelerator"
set xpr       [file join $proj_dir ${proj_name}.xpr]
set rpt_dir   [file normalize [file join $repo_root vivado_output reports]]

file mkdir $rpt_dir

if {![file exists $xpr]} {
    puts "INFO: Project not found; creating via create_vivado_project.tcl"
    source [file join $script_dir create_vivado_project.tcl]
} else {
    puts "INFO: Opening existing project $xpr"
    open_project $xpr
}

# Refresh generics in case repo moved on disk
set_property generic $cnn_generic_list [current_fileset]
update_compile_order -fileset sources_1

puts "================================================================"
puts " Running synthesis"
puts "================================================================"
reset_run synth_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1

if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    error "Synthesis failed"
}

open_run synth_1
report_utilization -file [file join $rpt_dir post_synth_utilization.rpt]
report_timing_summary -file [file join $rpt_dir post_synth_timing.rpt]
report_utilization -hierarchical -file [file join $rpt_dir hierarchical_utilization.rpt]

puts "================================================================"
puts " Running implementation (opt + place + route)"
puts "================================================================"
reset_run impl_1
launch_runs impl_1 -to_step route_design -jobs 4
wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    error "Implementation failed"
}

open_run impl_1
report_utilization -file [file join $rpt_dir post_route_utilization.rpt]
report_timing_summary -file [file join $rpt_dir post_route_timing.rpt]
report_timing -max_paths 20 -nworst 10 -sort_by group \
    -file [file join $rpt_dir worst_timing_paths.rpt]
report_utilization -hierarchical \
    -file [file join $rpt_dir hierarchical_utilization.rpt]

puts "================================================================"
puts " Analysis complete"
puts "================================================================"
puts " Reports written under:"
puts "   $rpt_dir"
puts " Files:"
puts "   post_synth_utilization.rpt"
puts "   post_synth_timing.rpt"
puts "   post_route_utilization.rpt"
puts "   post_route_timing.rpt"
puts "   hierarchical_utilization.rpt"
puts "   worst_timing_paths.rpt"
puts "================================================================"
puts "NOTE: Clock constraint is provisional 100 MHz on port clk."
puts "      Revisit before claiming timing closure for PS FCLK."
puts "================================================================"
