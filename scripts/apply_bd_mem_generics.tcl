# apply_bd_mem_generics.tcl
#
# Fix for FPGA all-zero logits with correct cycle_count:
# Block Design Module Reference of cnn_accelerator_bd_wrapper uses relative
# default .mem paths. Vivado synthesis CWD is not the repo root, so
# $readmemh fails and parameter ROMs synthesize as zeros.
#
# This script:
#   1. Resolves absolute forward-slash paths for all 12 model .mem files
#   2. Ensures they are in the project sources_1 fileset
#   3. Applies them as CONFIG.* generics on the BD Module Reference cell
#
# Usage (Vivado Tcl console, with zynq BD open):
#
#   cd <repo_root>
#   source scripts/apply_bd_mem_generics.tcl
#   # optional: set cell name if different
#   # set cnn_bd_cell cnn_accelerator_bd_w_0
#   cnn_apply_bd_mem_generics
#   validate_bd_design
#   save_bd_design
#
# Then regenerate bitstream / export XSA / refresh Vitis platform.

if {![info exists repo_root]} {
    set _script_dir [file normalize [file dirname [info script]]]
    set repo_root   [file normalize [file join $_script_dir ".."]]
}

source [file join $repo_root scripts vivado_sources.tcl]

if {![info exists cnn_bd_cell]} {
    # Vivado often shortens long module-reference instance names
    set cnn_bd_cell ""
    foreach cand {
        cnn_accelerator_bd_w_0
        cnn_accelerator_bd_wrapper_0
        cnn_accelerator_bd_0
    } {
        if {[llength [get_bd_cells -quiet $cand]] > 0} {
            set cnn_bd_cell $cand
            break
        }
    }
}

proc cnn_apply_bd_mem_generics {{cell_name ""}} {
    global repo_root cnn_mem_sources cnn_bd_cell cnn_generic_list

    cnn_assert_sources_exist

    if {$cell_name eq ""} {
        set cell_name $cnn_bd_cell
    }
    if {$cell_name eq "" || [llength [get_bd_cells -quiet $cell_name]] == 0} {
        set cells [get_bd_cells -quiet -filter {VLNV =~ "*cnn_accelerator_bd_wrapper*"}]
        if {[llength $cells] == 0} {
            set cells [get_bd_cells -quiet *]
            puts "ERROR: Could not find cnn_accelerator_bd_wrapper Module Reference."
            puts "       Existing BD cells:"
            foreach c $cells { puts "         $c" }
            error "Set cnn_bd_cell to your Module Reference instance name and retry."
        }
        set cell_name [lindex $cells 0]
    }

    puts "INFO: Applying absolute MEM paths to BD cell: $cell_name"
    puts "INFO: repo_root = $repo_root"

    # Ensure .mem files are in the project (portable; no machine-specific edit)
    foreach f $cnn_mem_sources {
        if {[llength [get_files -quiet $f]] == 0} {
            puts "INFO: add_files $f"
            add_files -fileset sources_1 $f
        } else {
            puts "INFO: already in project: $f"
        }
    }

    # Build CONFIG dict from absolute paths (same list as standalone synth generics)
    # cnn_generic_list entries look like: CONV1_WGT_MEM="/abs/path/file.mem"
    foreach entry $cnn_generic_list {
        if {![regexp {^([A-Z0-9_]+)=\"(.*)\"$} $entry -> pname pval]} {
            error "Bad generic entry: $entry"
        }
        if {![file exists $pval]} {
            error "MEM file missing for $pname: $pval"
        }
        puts "INFO: CONFIG.$pname = $pval"
        set_property CONFIG.$pname $pval [get_bd_cells $cell_name]
    }

    puts "INFO: Done. Next: validate_bd_design ; save_bd_design ; then rebuild bitstream."
}

puts "Loaded apply_bd_mem_generics.tcl — run: cnn_apply_bd_mem_generics"
