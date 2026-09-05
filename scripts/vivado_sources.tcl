# vivado_sources.tcl
# Explicit synthesizable source / parameter / constraint lists for Vivado.
#
# Sourced by create_vivado_project.tcl after repo_root is defined.
# Do NOT recursively add all rtl/*.sv — that would pull in legacy tops and
# simulation-only wrappers.

if {![info exists repo_root]} {
    error "vivado_sources.tcl requires repo_root to be set first"
}

proc cnn_repo_path {args} {
    global repo_root
    return [file normalize [file join $repo_root {*}$args]]
}

# Convert to forward-slash form for $readmemh-friendly string parameters.
proc cnn_fwd_path {p} {
    return [string map {\\ /} [file normalize $p]]
}

# ---------------------------------------------------------------------------
# Synthesis top (minimal pinout for xc7z010; full CNN sits underneath)
# ---------------------------------------------------------------------------
set cnn_synth_top "cnn_accelerator_synth_wrapper"
set cnn_fpga_part "xc7z010clg400-1"

# ---------------------------------------------------------------------------
# Synthesizable RTL (dependency cone of cnn_accelerator_shared_compute_top)
# Order is roughly bottom-up; Vivado will reorder as needed.
# ---------------------------------------------------------------------------
set cnn_rtl_sources [list \
    [cnn_repo_path rtl int8_mac.sv] \
    [cnn_repo_path rtl rounding_right_shift64.sv] \
    [cnn_repo_path rtl saturate_shifted_int8.sv] \
    [cnn_repo_path rtl saturate_shifted_int32.sv] \
    [cnn_repo_path rtl requantize_from_product.sv] \
    [cnn_repo_path rtl requantize.sv] \
    [cnn_repo_path rtl relu_int8.sv] \
    [cnn_repo_path rtl saturate_int8.sv] \
    [cnn_repo_path rtl max4_int8.sv] \
    [cnn_repo_path rtl int8_sync_ram.sv] \
    [cnn_repo_path rtl int8_sync_rom.sv] \
    [cnn_repo_path rtl int32_sync_ram.sv] \
    [cnn_repo_path rtl int32_sync_rom.sv] \
    [cnn_repo_path rtl activation_ram.sv] \
    [cnn_repo_path rtl gap_output_storage.sv] \
    [cnn_repo_path rtl logit_storage.sv] \
    [cnn_repo_path rtl conv_address_generator.sv] \
    [cnn_repo_path rtl conv2_address_generator.sv] \
    [cnn_repo_path rtl shared_conv_address_generator.sv] \
    [cnn_repo_path rtl shared_conv_single_output.sv] \
    [cnn_repo_path rtl shared_conv_layer_controller.sv] \
    [cnn_repo_path rtl shared_conv_engine.sv] \
    [cnn_repo_path rtl maxpool2x2_address_generator.sv] \
    [cnn_repo_path rtl maxpool2_address_generator.sv] \
    [cnn_repo_path rtl shared_maxpool_address_generator.sv] \
    [cnn_repo_path rtl shared_maxpool_engine.sv] \
    [cnn_repo_path rtl gap_average.sv] \
    [cnn_repo_path rtl global_average_pool_controller.sv] \
    [cnn_repo_path rtl fc_address_generator.sv] \
    [cnn_repo_path rtl fc_logit_from_product.sv] \
    [cnn_repo_path rtl fc_output_postprocess.sv] \
    [cnn_repo_path rtl fully_connected_class_engine.sv] \
    [cnn_repo_path rtl fully_connected_layer_controller.sv] \
    [cnn_repo_path rtl signed_argmax5_controller.sv] \
    [cnn_repo_path rtl cnn_top_controller.sv] \
    [cnn_repo_path rtl cnn_accelerator_shared_compute_top.sv] \
    [cnn_repo_path rtl cnn_accelerator_synth_wrapper.sv] \
    [cnn_repo_path rtl cnn_accelerator_bd_wrapper.v] \
]

# ---------------------------------------------------------------------------
# Model parameter ROM init files required by $readmemh in synthesis RTL
# (NOT simulation golden expected tensors)
# ---------------------------------------------------------------------------
set cnn_mem_sources [list \
    [cnn_repo_path vectors conv1_memory conv1_weights.mem] \
    [cnn_repo_path vectors conv1_memory conv1_biases.mem] \
    [cnn_repo_path vectors conv1_memory conv1_multipliers.mem] \
    [cnn_repo_path vectors conv1_memory conv1_shifts.mem] \
    [cnn_repo_path vectors conv2 conv2_weights.mem] \
    [cnn_repo_path vectors conv2 conv2_biases.mem] \
    [cnn_repo_path vectors conv2 conv2_multipliers.mem] \
    [cnn_repo_path vectors conv2 conv2_shifts.mem] \
    [cnn_repo_path vectors fc fc_weights.mem] \
    [cnn_repo_path vectors fc fc_biases.mem] \
    [cnn_repo_path vectors fc fc_multipliers.mem] \
    [cnn_repo_path vectors fc fc_shifts.mem] \
]

# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------
set cnn_xdc_sources [list \
    [cnn_repo_path rtl constraints zynq_edge_ai.xdc] \
]

# Absolute forward-slash paths for top-level string parameters / generics.
# Vivado elaborates $readmemh relative to a fragile CWD otherwise.
# Values are quoted string literals for SystemVerilog string parameters.
set cnn_generic_list [list \
    "CONV1_WGT_MEM=\"[cnn_fwd_path [cnn_repo_path vectors conv1_memory conv1_weights.mem]]\"" \
    "CONV1_BIAS_MEM=\"[cnn_fwd_path [cnn_repo_path vectors conv1_memory conv1_biases.mem]]\"" \
    "CONV1_MULT_MEM=\"[cnn_fwd_path [cnn_repo_path vectors conv1_memory conv1_multipliers.mem]]\"" \
    "CONV1_SHIFT_MEM=\"[cnn_fwd_path [cnn_repo_path vectors conv1_memory conv1_shifts.mem]]\"" \
    "CONV2_WGT_MEM=\"[cnn_fwd_path [cnn_repo_path vectors conv2 conv2_weights.mem]]\"" \
    "CONV2_BIAS_MEM=\"[cnn_fwd_path [cnn_repo_path vectors conv2 conv2_biases.mem]]\"" \
    "CONV2_MULT_MEM=\"[cnn_fwd_path [cnn_repo_path vectors conv2 conv2_multipliers.mem]]\"" \
    "CONV2_SHIFT_MEM=\"[cnn_fwd_path [cnn_repo_path vectors conv2 conv2_shifts.mem]]\"" \
    "FC_WGT_MEM=\"[cnn_fwd_path [cnn_repo_path vectors fc fc_weights.mem]]\"" \
    "FC_BIAS_MEM=\"[cnn_fwd_path [cnn_repo_path vectors fc fc_biases.mem]]\"" \
    "FC_MULT_MEM=\"[cnn_fwd_path [cnn_repo_path vectors fc fc_multipliers.mem]]\"" \
    "FC_SHIFT_MEM=\"[cnn_fwd_path [cnn_repo_path vectors fc fc_shifts.mem]]\"" \
]

proc cnn_assert_sources_exist {} {
    global cnn_rtl_sources cnn_mem_sources cnn_xdc_sources
    foreach f [concat $cnn_rtl_sources $cnn_mem_sources $cnn_xdc_sources] {
        if {![file exists $f]} {
            error "Required source missing: $f"
        }
    }
}
