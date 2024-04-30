import os
import argparse
import onnx
import onnx.numpy_helper as nph
import torch
import numpy as np
from exporter.Exporter import preprocessing, postprocessing, make_input_channels_last, streamline_resnet, convert_to_hw_resnet, name_nodes
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import part_map, alveo_default_platform

from brevitas.export import export_qonnx

build_dir = os.environ['FINN_BUILD_DIR']

parser = argparse.ArgumentParser(description = 'Transform input onnx model to hw')
parser.add_argument('--onnx-model', required = True, type = str, help = 'QONNX model to transform using FINN Compiler')
parser.add_argument('--output-dir', required = False, default = '', type = None, help = 'Output directory')
parser.add_argument('--synth-clk-period-ns', type = float, default = 10.0, help = 'Target clock period in ns')
parser.add_argument('--board', default = "U250", help = "Name of target board")
parser.add_argument('--shell-flow-type', default = "vitis_alveo", choices = ["vivado_zynq", "vitis_alveo"], help = "Target shell type")
parser.add_argument('--target-fps', type = int, default = 100000, help = 'Target fps')
parser.add_argument('--dataset', default = "MNIST", choices = ["MNIST", "CIFAR10"], help = 'Dataset')


def main():
	args = parser.parse_args()
	output_dir = build_dir + "/" + args.output_dir

	cfg_build = build.DataflowBuildConfig(
		output_dir = output_dir,
		synth_clk_period_ns = args.synth_clk_period_ns,
		mvau_wwidth_max = 16,
		board = args.board,
		shell_flow_type = args.shell_flow_type,
		fpga_part = part_map[args.board],
		vitis_platform = alveo_default_platform[args.board],
		steps = [
			preprocessing,
			postprocessing,
			make_input_channels_last,
			"step_tidy_up",
			"step_qonnx_to_finn",
			"step_tidy_up",
			streamline_resnet,
			convert_to_hw_resnet,
			"step_create_dataflow_partition",
			"step_specialize_layers",
			"step_target_fps_parallelization",
			name_nodes,
			"step_apply_folding_config",
			"step_minimize_bit_width",
			"step_generate_estimate_reports",
			name_nodes,
			"step_hw_codegen",
			"step_hw_ipgen",
			"step_set_fifo_depths",
			"step_create_stitched_ip",
			"step_measure_rtlsim_performance",
			"step_out_of_context_synthesis",
			"step_synthesize_bitfile",
			"step_make_pynq_driver",
			"step_deployment_package",
		],
		generate_outputs = [
			build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
			build_cfg.DataflowOutputType.STITCHED_IP,
			build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
			build_cfg.DataflowOutputType.OOC_SYNTH,
			build_cfg.DataflowOutputType.BITFILE,
			build_cfg.DataflowOutputType.PYNQ_DRIVER,
			build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
		],
		verify_steps = [
			build_cfg.VerificationStepType.QONNX_TO_FINN_PYTHON,
			build_cfg.VerificationStepType.TIDY_UP_PYTHON,
			build_cfg.VerificationStepType.STREAMLINED_PYTHON,
			build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM,
			build_cfg.VerificationStepType.STITCHED_IP_RTLSIM
		]
	)

	build.build_dataflow_cfg(args.onnx_model, cfg_build)

if __name__ == "__main__":
	main()
