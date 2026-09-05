"""Run Verilator unit tests for Conv1 / MaxPool / Conv2 / GAP / FC / Argmax RTL milestones.

Includes:
  * streamed single-output arithmetic
  * memory-driven address generation + single-output engine
  * one-channel (oc=0) controller
  * full Conv1 layer (all 16 trained output channels)
  * MaxPool1 (16 x 16 x 16 signed INT8 max-pool)
  * Conv2 full layer (16->32 on 16x16 Pool1, 144 MACs/output)
  * MaxPool2 (32 x 8 x 8 signed INT8 max-pool)
  * Global Average Pooling (32 x 8 x 8 -> 32)
  * Fully connected classifier (32 -> 5 INT32 logits)
  * Signed argmax (5 logits -> predicted_class[2:0])
  * End-to-end CNN integration (Conv1..Argmax)
  * Ping-pong activation-memory end-to-end top
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from software.utils.config import project_root


ROOT = project_root()
OBJ = ROOT / "build" / "rtl_obj"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)


def verilate_and_run(
    tb_name: str, sources: list[str], *, timing: bool = False
) -> None:
    if not shutil.which("verilator"):
        raise RuntimeError("verilator not found on PATH")

    out_dir = OBJ / tb_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    top_sv = f"tb/{tb_name}.sv"
    header = f"V{tb_name}.h"
    cls = f"V{tb_name}"

    cmd = [
        "verilator",
        "-Wall",
        "-Wno-fatal",
        "--cc",
        "--exe",
        "-Mdir",
        str(out_dir),
        "-CFLAGS",
        f'-DTOP_HEADER=\\"{header}\\" -DTOP_CLASS={cls}',
        str(ROOT / "tb" / "sim_main.cpp"),
        *[str(ROOT / s) if not Path(s).is_absolute() else s for s in sources],
        str(ROOT / top_sv),
        "--top-module",
        tb_name,
    ]
    if timing:
        # Event-control tasks (@(posedge clk)) need Verilator timing mode.
        cmd.insert(cmd.index("--cc"), "--timing")
    run(cmd)

    make_cmd = [
        "make",
        "-C",
        str(out_dir),
        "-f",
        f"V{tb_name}.mk",
        f"V{tb_name}",
    ]
    run(make_cmd)

    binary = out_dir / f"V{tb_name}"
    if not binary.is_file():
        raise FileNotFoundError(f"Missing Verilator binary {binary}")
    run([str(binary)], cwd=ROOT)


def run_python_tests() -> None:
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "software/tests/test_memory_layout.py",
            "software/tests/test_quantization.py",
            "software/tests/test_integer_inference.py",
            "software/tests/test_fpga_export.py",
            "software/tests/test_quantized_regression.py",
            "-q",
        ]
    )


ARITH_SOURCES = {
    "tb_int8_mac": ["rtl/int8_mac.sv"],
    "tb_relu_int8": ["rtl/relu_int8.sv"],
    "tb_requantize": ["rtl/rounding_right_shift64.sv", "rtl/saturate_shifted_int8.sv", "rtl/requantize_from_product.sv", "rtl/requantize.sv"],
    "tb_saturate_int8": ["rtl/saturate_int8.sv"],
    "tb_conv_single_output": [
        "rtl/int8_mac.sv",
        "rtl/rounding_right_shift64.sv", "rtl/saturate_shifted_int8.sv", "rtl/requantize_from_product.sv", "rtl/requantize.sv",
        "rtl/relu_int8.sv",
        "rtl/saturate_int8.sv",
        "rtl/conv_single_output.sv",
    ],
}

MEMORY_SOURCES = {
    "tb_conv_address_generator": ["rtl/conv_address_generator.sv"],
    "tb_int8_sync_rom_latency": ["rtl/int8_sync_rom.sv"],
    "tb_conv1_memory_single_output": [
        "rtl/conv_address_generator.sv",
        "rtl/int8_sync_rom.sv",
        "rtl/int32_sync_rom.sv",
        "rtl/int8_mac.sv",
        "rtl/rounding_right_shift64.sv", "rtl/saturate_shifted_int8.sv", "rtl/requantize_from_product.sv", "rtl/requantize.sv",
        "rtl/relu_int8.sv",
        "rtl/saturate_int8.sv",
        "rtl/conv1_memory_single_output.sv",
    ],
}

CHANNEL_SOURCES = {
    "tb_conv1_channel": [
        "rtl/conv_address_generator.sv",
        "rtl/int8_sync_rom.sv",
        "rtl/int32_sync_rom.sv",
        "rtl/int8_sync_ram.sv",
        "rtl/int8_mac.sv",
        "rtl/rounding_right_shift64.sv", "rtl/saturate_shifted_int8.sv", "rtl/requantize_from_product.sv", "rtl/requantize.sv",
        "rtl/relu_int8.sv",
        "rtl/saturate_int8.sv",
        "rtl/conv1_memory_single_output.sv",
        "rtl/conv1_channel_controller.sv",
        "rtl/conv1_channel_top.sv",
    ],
}

FULL_SOURCES = {
    "tb_conv1_full_layer": [
        "rtl/conv_address_generator.sv",
        "rtl/int8_sync_rom.sv",
        "rtl/int32_sync_rom.sv",
        "rtl/int8_sync_ram.sv",
        "rtl/int8_mac.sv",
        "rtl/rounding_right_shift64.sv", "rtl/saturate_shifted_int8.sv", "rtl/requantize_from_product.sv", "rtl/requantize.sv",
        "rtl/relu_int8.sv",
        "rtl/saturate_int8.sv",
        "rtl/conv1_memory_single_output.sv",
        "rtl/conv1_layer_controller.sv",
        "rtl/conv1_layer_top.sv",
    ],
}

POOL_SOURCES = {
    "tb_maxpool2x2_address_generator": ["rtl/maxpool2x2_address_generator.sv"],
    "tb_max4_int8": ["rtl/max4_int8.sv"],
    "tb_maxpool1_selected_windows": [
        "rtl/maxpool2x2_address_generator.sv",
        "rtl/max4_int8.sv",
        "rtl/maxpool1_controller.sv",
        "rtl/int8_sync_rom.sv",
        "rtl/int8_sync_ram.sv",
        "rtl/maxpool1_top.sv",
    ],
    "tb_maxpool1_full": [
        "rtl/maxpool2x2_address_generator.sv",
        "rtl/max4_int8.sv",
        "rtl/maxpool1_controller.sv",
        "rtl/int8_sync_rom.sv",
        "rtl/int8_sync_ram.sv",
        "rtl/maxpool1_top.sv",
    ],
}

CONV2_SOURCES = {
    "tb_conv2_address_generator": ["rtl/conv2_address_generator.sv"],
    "tb_conv2_selected_outputs": [
        "rtl/conv2_address_generator.sv",
        "rtl/int8_sync_rom.sv",
        "rtl/int32_sync_rom.sv",
        "rtl/int8_mac.sv",
        "rtl/rounding_right_shift64.sv", "rtl/saturate_shifted_int8.sv", "rtl/requantize_from_product.sv", "rtl/requantize.sv",
        "rtl/relu_int8.sv",
        "rtl/saturate_int8.sv",
        "rtl/conv2_memory_single_output.sv",
    ],
    "tb_conv2_full_layer": [
        "rtl/conv2_address_generator.sv",
        "rtl/int8_sync_rom.sv",
        "rtl/int32_sync_rom.sv",
        "rtl/int8_sync_ram.sv",
        "rtl/int8_mac.sv",
        "rtl/rounding_right_shift64.sv", "rtl/saturate_shifted_int8.sv", "rtl/requantize_from_product.sv", "rtl/requantize.sv",
        "rtl/relu_int8.sv",
        "rtl/saturate_int8.sv",
        "rtl/conv2_memory_single_output.sv",
        "rtl/conv2_layer_controller.sv",
        "rtl/conv2_layer_top.sv",
    ],
}

POOL2_SOURCES = {
    "tb_maxpool2_address_generator": ["rtl/maxpool2_address_generator.sv"],
    "tb_maxpool2_selected_windows": [
        "rtl/maxpool2_address_generator.sv",
        "rtl/max4_int8.sv",
        "rtl/maxpool2_controller.sv",
        "rtl/int8_sync_rom.sv",
        "rtl/int8_sync_ram.sv",
        "rtl/maxpool2_top.sv",
    ],
    "tb_maxpool2_full": [
        "rtl/maxpool2_address_generator.sv",
        "rtl/max4_int8.sv",
        "rtl/maxpool2_controller.sv",
        "rtl/int8_sync_rom.sv",
        "rtl/int8_sync_ram.sv",
        "rtl/maxpool2_top.sv",
    ],
}

GAP_SOURCES = {
    "tb_gap_average": [
        "rtl/saturate_int8.sv",
        "rtl/gap_average.sv",
    ],
    "tb_gap_selected_channels": [
        "rtl/saturate_int8.sv",
        "rtl/gap_average.sv",
        "rtl/rounding_right_shift64.sv", "rtl/saturate_shifted_int8.sv", "rtl/requantize_from_product.sv", "rtl/requantize.sv",
        "rtl/int8_sync_rom.sv",
        "rtl/int8_sync_ram.sv",
        "rtl/gap_output_storage.sv",
        "rtl/global_average_pool_controller.sv",
        "rtl/global_average_pool_top.sv",
    ],
    "tb_global_average_pool_full": [
        "rtl/saturate_int8.sv",
        "rtl/gap_average.sv",
        "rtl/rounding_right_shift64.sv", "rtl/saturate_shifted_int8.sv", "rtl/requantize_from_product.sv", "rtl/requantize.sv",
        "rtl/int8_sync_rom.sv",
        "rtl/int8_sync_ram.sv",
        "rtl/gap_output_storage.sv",
        "rtl/global_average_pool_controller.sv",
        "rtl/global_average_pool_top.sv",
    ],
}

FC_SOURCES = {
    "tb_fc_address_generator": ["rtl/fc_address_generator.sv"],
    "tb_fully_connected_selected_class": [
        "rtl/fc_address_generator.sv",
        "rtl/int8_mac.sv",
        "rtl/rounding_right_shift64.sv", "rtl/saturate_shifted_int32.sv", "rtl/fc_logit_from_product.sv", "rtl/fc_output_postprocess.sv",
        "rtl/fully_connected_class_engine.sv",
        "rtl/fully_connected_layer_controller.sv",
        "rtl/int8_sync_rom.sv",
        "rtl/int32_sync_rom.sv",
        "rtl/int32_sync_ram.sv",
        "rtl/logit_storage.sv",
        "rtl/fully_connected_top.sv",
    ],
    "tb_fully_connected_full": [
        "rtl/fc_address_generator.sv",
        "rtl/int8_mac.sv",
        "rtl/rounding_right_shift64.sv", "rtl/saturate_shifted_int32.sv", "rtl/fc_logit_from_product.sv", "rtl/fc_output_postprocess.sv",
        "rtl/fully_connected_class_engine.sv",
        "rtl/fully_connected_layer_controller.sv",
        "rtl/int8_sync_rom.sv",
        "rtl/int32_sync_rom.sv",
        "rtl/int32_sync_ram.sv",
        "rtl/logit_storage.sv",
        "rtl/fully_connected_top.sv",
    ],
}


E2E_SOURCES = {
    "tb_cnn_end_to_end_single_image": [
        "rtl/conv_address_generator.sv",
        "rtl/conv2_address_generator.sv",
        "rtl/maxpool2x2_address_generator.sv",
        "rtl/maxpool2_address_generator.sv",
        "rtl/int8_sync_rom.sv",
        "rtl/int32_sync_rom.sv",
        "rtl/int8_sync_ram.sv",
        "rtl/int32_sync_ram.sv",
        "rtl/int8_mac.sv",
        "rtl/rounding_right_shift64.sv", "rtl/saturate_shifted_int8.sv", "rtl/requantize_from_product.sv", "rtl/requantize.sv",
        "rtl/relu_int8.sv",
        "rtl/saturate_int8.sv",
        "rtl/max4_int8.sv",
        "rtl/gap_average.sv",
        "rtl/fc_address_generator.sv",
        "rtl/rounding_right_shift64.sv", "rtl/saturate_shifted_int32.sv", "rtl/fc_logit_from_product.sv", "rtl/fc_output_postprocess.sv",
        "rtl/gap_output_storage.sv",
        "rtl/logit_storage.sv",
        "rtl/conv1_memory_single_output.sv",
        "rtl/conv1_layer_controller.sv",
        "rtl/maxpool1_controller.sv",
        "rtl/conv2_memory_single_output.sv",
        "rtl/conv2_layer_controller.sv",
        "rtl/maxpool2_controller.sv",
        "rtl/global_average_pool_controller.sv",
        "rtl/fully_connected_class_engine.sv",
        "rtl/fully_connected_layer_controller.sv",
        "rtl/signed_argmax5_controller.sv",
        "rtl/cnn_top_controller.sv",
        "rtl/cnn_accelerator_top.sv",
    ],
}
E2E_SOURCES["tb_cnn_end_to_end_multi_image"] = list(
    E2E_SOURCES["tb_cnn_end_to_end_single_image"]
)

PINGPONG_SOURCES = {
    "tb_cnn_pingpong_end_to_end": [
        "rtl/conv_address_generator.sv",
        "rtl/conv2_address_generator.sv",
        "rtl/maxpool2x2_address_generator.sv",
        "rtl/maxpool2_address_generator.sv",
        "rtl/int8_sync_rom.sv",
        "rtl/int32_sync_rom.sv",
        "rtl/int8_sync_ram.sv",
        "rtl/int32_sync_ram.sv",
        "rtl/activation_ram.sv",
        "rtl/int8_mac.sv",
        "rtl/rounding_right_shift64.sv", "rtl/saturate_shifted_int8.sv", "rtl/requantize_from_product.sv", "rtl/requantize.sv",
        "rtl/relu_int8.sv",
        "rtl/saturate_int8.sv",
        "rtl/max4_int8.sv",
        "rtl/gap_average.sv",
        "rtl/fc_address_generator.sv",
        "rtl/rounding_right_shift64.sv", "rtl/saturate_shifted_int32.sv", "rtl/fc_logit_from_product.sv", "rtl/fc_output_postprocess.sv",
        "rtl/gap_output_storage.sv",
        "rtl/logit_storage.sv",
        "rtl/conv1_memory_single_output.sv",
        "rtl/conv1_layer_controller.sv",
        "rtl/maxpool1_controller.sv",
        "rtl/conv2_memory_single_output.sv",
        "rtl/conv2_layer_controller.sv",
        "rtl/maxpool2_controller.sv",
        "rtl/global_average_pool_controller.sv",
        "rtl/fully_connected_class_engine.sv",
        "rtl/fully_connected_layer_controller.sv",
        "rtl/signed_argmax5_controller.sv",
        "rtl/cnn_top_controller.sv",
        "rtl/cnn_accelerator_pingpong_top.sv",
    ],
}

SHARED_CONV_CORE = [
    "rtl/conv_address_generator.sv",
    "rtl/conv2_address_generator.sv",
    "rtl/shared_conv_address_generator.sv",
    "rtl/int8_sync_rom.sv",
    "rtl/int32_sync_rom.sv",
    "rtl/int8_sync_ram.sv",
    "rtl/int8_mac.sv",
    "rtl/rounding_right_shift64.sv", "rtl/saturate_shifted_int8.sv", "rtl/requantize_from_product.sv", "rtl/requantize.sv",
    "rtl/relu_int8.sv",
    "rtl/saturate_int8.sv",
    "rtl/shared_conv_single_output.sv",
    "rtl/shared_conv_layer_controller.sv",
    "rtl/shared_conv_engine.sv",
]

SHARED_CONV_SOURCES = {
    "tb_shared_conv_engine_conv1": list(SHARED_CONV_CORE),
    "tb_shared_conv_engine_conv2": list(SHARED_CONV_CORE),
    "tb_cnn_shared_conv_end_to_end": SHARED_CONV_CORE
    + [
        "rtl/maxpool2x2_address_generator.sv",
        "rtl/maxpool2_address_generator.sv",
        "rtl/int32_sync_ram.sv",
        "rtl/activation_ram.sv",
        "rtl/max4_int8.sv",
        "rtl/gap_average.sv",
        "rtl/fc_address_generator.sv",
        "rtl/rounding_right_shift64.sv", "rtl/saturate_shifted_int32.sv", "rtl/fc_logit_from_product.sv", "rtl/fc_output_postprocess.sv",
        "rtl/gap_output_storage.sv",
        "rtl/logit_storage.sv",
        "rtl/maxpool1_controller.sv",
        "rtl/maxpool2_controller.sv",
        "rtl/global_average_pool_controller.sv",
        "rtl/fully_connected_class_engine.sv",
        "rtl/fully_connected_layer_controller.sv",
        "rtl/signed_argmax5_controller.sv",
        "rtl/cnn_top_controller.sv",
        "rtl/cnn_accelerator_shared_conv_top.sv",
    ],
}

SHARED_POOL_CORE = [
    "rtl/maxpool2x2_address_generator.sv",
    "rtl/maxpool2_address_generator.sv",
    "rtl/shared_maxpool_address_generator.sv",
    "rtl/max4_int8.sv",
    "rtl/int8_sync_ram.sv",
    "rtl/shared_maxpool_engine.sv",
]

SHARED_POOL_SOURCES = {
    "tb_shared_maxpool_pool1": list(SHARED_POOL_CORE),
    "tb_shared_maxpool_pool2": list(SHARED_POOL_CORE),
    "tb_cnn_shared_compute_end_to_end": SHARED_CONV_CORE
    + [
        "rtl/maxpool2x2_address_generator.sv",
        "rtl/maxpool2_address_generator.sv",
        "rtl/shared_maxpool_address_generator.sv",
        "rtl/shared_maxpool_engine.sv",
        "rtl/int32_sync_ram.sv",
        "rtl/activation_ram.sv",
        "rtl/max4_int8.sv",
        "rtl/gap_average.sv",
        "rtl/fc_address_generator.sv",
        "rtl/rounding_right_shift64.sv", "rtl/saturate_shifted_int32.sv", "rtl/fc_logit_from_product.sv", "rtl/fc_output_postprocess.sv",
        "rtl/gap_output_storage.sv",
        "rtl/logit_storage.sv",
        "rtl/global_average_pool_controller.sv",
        "rtl/fully_connected_class_engine.sv",
        "rtl/fully_connected_layer_controller.sv",
        "rtl/signed_argmax5_controller.sv",
        "rtl/cnn_top_controller.sv",
        "rtl/cnn_accelerator_shared_compute_top.sv",
    ],
}

ARGMAX_SOURCES = {
    "tb_signed_argmax5": ["rtl/signed_argmax5.sv"],
    "tb_argmax_real_logits": [
        "rtl/int32_sync_rom.sv",
        "rtl/signed_argmax5_controller.sv",
        "rtl/argmax_top.sv",
    ],
    "tb_argmax_random": ["rtl/signed_argmax5.sv"],
}

AXI_CTRL_SOURCES = {
    "tb_cnn_axi_ctrl": [
        "ip_repo/cnn_axi_ctrl_1_0/hdl/cnn_axi_ctrl_slave_lite_v1_0_S00_AXI.v",
        "ip_repo/cnn_axi_ctrl_1_0/hdl/cnn_axi_ctrl.v",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument(
        "--only",
        choices=[
            "vectors",
            "mac",
            "relu",
            "requant",
            "conv",
            "address",
            "latency",
            "memory",
            "channel",
            "full",
            "pool",
            "conv2",
            "pool2",
            "gap",
            "fc",
            "argmax",
            "e2e",
            "pingpong",
            "shared_conv",
            "shared_pool",
            "axi",
            "arith",
            "all",
        ],
        default="all",
    )
    args = parser.parse_args()

    need_single = args.only in ("all", "vectors", "arith")
    need_memory = args.only in (
        "all",
        "vectors",
        "memory",
        "address",
        "latency",
        "channel",
        "full",
    )
    need_channel = args.only in ("all", "vectors", "channel", "full")
    need_full = args.only in ("all", "vectors", "full", "shared_conv")
    need_pool = args.only in ("all", "vectors", "pool", "shared_pool")
    need_conv2 = args.only in ("all", "vectors", "conv2", "shared_conv")
    need_pool2 = args.only in ("all", "vectors", "pool2", "shared_pool")
    need_gap = args.only in ("all", "vectors", "gap")
    need_fc = args.only in ("all", "vectors", "fc")
    need_argmax = args.only in ("all", "vectors", "argmax")
    need_e2e = args.only in ("all", "vectors", "e2e")
    need_pingpong = args.only in ("all", "vectors", "pingpong", "e2e")
    need_shared = args.only in ("all", "vectors", "shared_conv")
    need_shared_pool = args.only in ("all", "vectors", "shared_pool")

    if need_single:
        run([sys.executable, "-m", "tools.export_rtl_vectors"], cwd=ROOT)
    if need_memory:
        run([sys.executable, "-m", "tools.export_conv1_memory_vectors"], cwd=ROOT)
    if need_channel:
        run([sys.executable, "-m", "tools.export_conv1_channel_vectors"], cwd=ROOT)
    if need_full:
        run([sys.executable, "-m", "tools.export_conv1_full_vectors"], cwd=ROOT)
    if need_pool:
        run([sys.executable, "-m", "tools.export_pool1_vectors"], cwd=ROOT)
    if need_conv2:
        run([sys.executable, "-m", "tools.export_conv2_vectors"], cwd=ROOT)
    if need_pool2:
        run([sys.executable, "-m", "tools.export_pool2_vectors"], cwd=ROOT)
    if need_gap:
        run([sys.executable, "-m", "tools.export_gap_vectors"], cwd=ROOT)
    if need_fc:
        run([sys.executable, "-m", "tools.export_fc_vectors"], cwd=ROOT)
    if need_argmax:
        run([sys.executable, "-m", "tools.export_argmax_vectors"], cwd=ROOT)
    if need_e2e or need_pingpong or need_shared or need_shared_pool:
        run([sys.executable, "-m", "tools.export_end_to_end_vectors"], cwd=ROOT)
    if args.only == "vectors":
        print("\nVector export completed successfully.")
        return

    if args.only in ("all", "arith", "mac"):
        verilate_and_run("tb_int8_mac", ARITH_SOURCES["tb_int8_mac"])
    if args.only in ("all", "arith", "relu"):
        verilate_and_run("tb_relu_int8", ARITH_SOURCES["tb_relu_int8"])
    if args.only in ("all", "arith", "requant"):
        verilate_and_run("tb_requantize", ARITH_SOURCES["tb_requantize"])
    if args.only in ("all", "arith"):
        verilate_and_run("tb_saturate_int8", ARITH_SOURCES["tb_saturate_int8"])
    if args.only in ("all", "arith", "conv"):
        verilate_and_run(
            "tb_conv_single_output", ARITH_SOURCES["tb_conv_single_output"]
        )

    if args.only in ("all", "memory", "address", "channel", "full"):
        verilate_and_run(
            "tb_conv_address_generator", MEMORY_SOURCES["tb_conv_address_generator"]
        )
    if args.only in ("all", "memory", "latency", "channel", "full"):
        verilate_and_run(
            "tb_int8_sync_rom_latency", MEMORY_SOURCES["tb_int8_sync_rom_latency"]
        )
    if args.only in ("all", "memory", "channel", "full"):
        verilate_and_run(
            "tb_conv1_memory_single_output",
            MEMORY_SOURCES["tb_conv1_memory_single_output"],
        )

    if args.only in ("all", "channel", "full"):
        verilate_and_run("tb_conv1_channel", CHANNEL_SOURCES["tb_conv1_channel"])

    if args.only in ("all", "full"):
        verilate_and_run("tb_conv1_full_layer", FULL_SOURCES["tb_conv1_full_layer"])

    if args.only in ("all", "pool"):
        verilate_and_run(
            "tb_maxpool2x2_address_generator",
            POOL_SOURCES["tb_maxpool2x2_address_generator"],
        )
        verilate_and_run("tb_max4_int8", POOL_SOURCES["tb_max4_int8"])
        verilate_and_run(
            "tb_maxpool1_selected_windows",
            POOL_SOURCES["tb_maxpool1_selected_windows"],
        )
        verilate_and_run("tb_maxpool1_full", POOL_SOURCES["tb_maxpool1_full"])

    if args.only in ("all", "conv2"):
        verilate_and_run(
            "tb_conv2_address_generator",
            CONV2_SOURCES["tb_conv2_address_generator"],
        )
        verilate_and_run(
            "tb_conv2_selected_outputs",
            CONV2_SOURCES["tb_conv2_selected_outputs"],
        )
        verilate_and_run("tb_conv2_full_layer", CONV2_SOURCES["tb_conv2_full_layer"])

    if args.only in ("all", "pool2"):
        verilate_and_run(
            "tb_maxpool2_address_generator",
            POOL2_SOURCES["tb_maxpool2_address_generator"],
        )
        verilate_and_run(
            "tb_maxpool2_selected_windows",
            POOL2_SOURCES["tb_maxpool2_selected_windows"],
        )
        verilate_and_run("tb_maxpool2_full", POOL2_SOURCES["tb_maxpool2_full"])

    if args.only in ("all", "gap"):
        verilate_and_run("tb_gap_average", GAP_SOURCES["tb_gap_average"])
        verilate_and_run(
            "tb_gap_selected_channels",
            GAP_SOURCES["tb_gap_selected_channels"],
        )
        verilate_and_run(
            "tb_global_average_pool_full",
            GAP_SOURCES["tb_global_average_pool_full"],
        )

    if args.only in ("all", "fc"):
        verilate_and_run(
            "tb_fc_address_generator",
            FC_SOURCES["tb_fc_address_generator"],
        )
        verilate_and_run(
            "tb_fully_connected_selected_class",
            FC_SOURCES["tb_fully_connected_selected_class"],
        )
        verilate_and_run(
            "tb_fully_connected_full",
            FC_SOURCES["tb_fully_connected_full"],
        )

    if args.only in ("all", "argmax"):
        verilate_and_run("tb_signed_argmax5", ARGMAX_SOURCES["tb_signed_argmax5"])
        verilate_and_run(
            "tb_argmax_real_logits",
            ARGMAX_SOURCES["tb_argmax_real_logits"],
        )
        verilate_and_run("tb_argmax_random", ARGMAX_SOURCES["tb_argmax_random"])

    if args.only in ("all", "e2e"):
        verilate_and_run(
            "tb_cnn_end_to_end_single_image",
            E2E_SOURCES["tb_cnn_end_to_end_single_image"],
        )
        verilate_and_run(
            "tb_cnn_end_to_end_multi_image",
            E2E_SOURCES["tb_cnn_end_to_end_multi_image"],
        )

    if args.only in ("all", "e2e", "pingpong"):
        verilate_and_run(
            "tb_cnn_pingpong_end_to_end",
            PINGPONG_SOURCES["tb_cnn_pingpong_end_to_end"],
        )

    if args.only in ("all", "shared_conv"):
        # Old Conv1/Conv2 regressions first
        verilate_and_run("tb_conv1_full_layer", FULL_SOURCES["tb_conv1_full_layer"])
        verilate_and_run("tb_conv2_full_layer", CONV2_SOURCES["tb_conv2_full_layer"])
        verilate_and_run(
            "tb_shared_conv_engine_conv1",
            SHARED_CONV_SOURCES["tb_shared_conv_engine_conv1"],
        )
        verilate_and_run(
            "tb_shared_conv_engine_conv2",
            SHARED_CONV_SOURCES["tb_shared_conv_engine_conv2"],
        )
        verilate_and_run(
            "tb_cnn_shared_conv_end_to_end",
            SHARED_CONV_SOURCES["tb_cnn_shared_conv_end_to_end"],
        )

    if args.only in ("all", "shared_pool"):
        verilate_and_run("tb_maxpool1_full", POOL_SOURCES["tb_maxpool1_full"])
        verilate_and_run("tb_maxpool2_full", POOL2_SOURCES["tb_maxpool2_full"])
        verilate_and_run(
            "tb_shared_maxpool_pool1",
            SHARED_POOL_SOURCES["tb_shared_maxpool_pool1"],
        )
        verilate_and_run(
            "tb_shared_maxpool_pool2",
            SHARED_POOL_SOURCES["tb_shared_maxpool_pool2"],
        )
        verilate_and_run(
            "tb_cnn_shared_compute_end_to_end",
            SHARED_POOL_SOURCES["tb_cnn_shared_compute_end_to_end"],
        )

    if args.only in ("all", "axi"):
        verilate_and_run(
            "tb_cnn_axi_ctrl",
            AXI_CTRL_SOURCES["tb_cnn_axi_ctrl"],
            timing=True,
        )

    if args.only == "all" and not args.skip_pytest:
        run_python_tests()

    print("\nAll requested RTL checks completed successfully.")


if __name__ == "__main__":
    main()
