import os
import argparse
from exporter.Exporter import (preprocessing, postprocessing,
							   make_input_channels_last, streamline_resnet, 
							   convert_to_hw_resnet, name_nodes, streamline_lenet,
							   convert_to_hw_lenet)

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import part_map, alveo_default_platform

build_dir = os.environ['FINN_BUILD_DIR']

parser = argparse.ArgumentParser(description = 'Transform input onnx model to hw')
parser.add_argument('--model-name', required = True, type = str, help = 'Model name')
parser.add_argument('--onnx-model', required = True, type = str, help = 'ONNX model to transform using FINN Compiler')
parser.add_argument('--output-dir', required = False, default = '', type = None, help = 'Output directory')
parser.add_argument('--synth-clk-period-ns', type = float, default = 3.33, help = 'Target clock period in ns')
parser.add_argument('--board', default = "U250", help = "Name of target board")
parser.add_argument('--shell-flow-type', default = "vitis_alveo", choices = ["vivado_zynq", "vitis_alveo"], help = "Target shell type")
parser.add_argument('--input-file', default = 'input.npy', type = str, help = 'Input file for validation')
parser.add_argument('--expected-output-file', default = 'expected_output.npy', type = str, help = 'Output file for validation')
parser.add_argument('--folding-config-file', default = 'folding_config.json', type = str, help = 'Folding config file')

parser.add_argument('--tidy-up-verification', action = argparse.BooleanOptionalAction, help = 'Perform verification after tidy-up transformation')
parser.add_argument('--qonnx-to-finn-verification', action = argparse.BooleanOptionalAction, help = 'Perform verification after QONNXToFinn transformation')
parser.add_argument('--streamlined-python-verification', action = argparse.BooleanOptionalAction, help = 'Perform verification after streamlining')
parser.add_argument('--folded-hls-cppsim', action = argparse.BooleanOptionalAction, help = 'Perform cpp simulation after folding')
parser.add_argument('--rtlsim-performance', action=argparse.BooleanOptionalAction, help = 'Generate rtlsim performance reports')
parser.add_argument('--rtlsim-verification', action=argparse.BooleanOptionalAction, help = 'Perform rtlsim verification (not recommended for large networks)')

streamline_functions = {
	'LeNet5' : streamline_lenet,
	'resnet18' : streamline_resnet,
	'resnet34' : streamline_resnet,
	'resnet50' : streamline_resnet,
	'resnet100' : streamline_resnet,
	'resnet152' : streamline_resnet,
}

convert_to_hw_functions = {
	'LeNet5' : convert_to_hw_lenet,
	'resnet18' : convert_to_hw_resnet,
	'resnet34' : convert_to_hw_resnet,
	'resnet50' : convert_to_hw_resnet,
	'resnet100' : convert_to_hw_resnet,
	'resnet152' : convert_to_hw_resnet,
}

def main():
	args = parser.parse_args()
	#output_dir = build_dir + "/" + args.output_dir
	output_dir = args.output_dir
	streamline_function = streamline_functions[args.model_name]
	convert_to_hw_function = convert_to_hw_functions[args.model_name]

	generate_outputs = [
		build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
		build_cfg.DataflowOutputType.STITCHED_IP,
		build_cfg.DataflowOutputType.OOC_SYNTH,
		build_cfg.DataflowOutputType.BITFILE,
		build_cfg.DataflowOutputType.PYNQ_DRIVER,
		build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
	]

	if args.rtlsim_performance:
		generate_outputs.append(build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE)

	verify_steps = []
	if args.tidy_up_verification:
		verify_steps.append(build_cfg.VerificationStepType.TIDY_UP_PYTHON)

	if args.qonnx_to_finn_verification:
		verify_steps.append(build_cfg.VerificationStepType.QONNX_TO_FINN_PYTHON)
	
	if args.streamlined_python_verification:
		verify_steps.append(build_cfg.VerificationStepType.STREAMLINED_PYTHON)

	if args.folded_hls_cppsim:
		verify_steps.append(build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM)

	if args.rtlsim_verification:
		verify_steps.append(build_cfg.VerificationStepType.STITCHED_IP_RTLSIM)

	cfg_build = build.DataflowBuildConfig(
		output_dir = output_dir,
		synth_clk_period_ns = args.synth_clk_period_ns,
		mvau_wwidth_max = 1000000,
		board = args.board,
		shell_flow_type = args.shell_flow_type,
		fpga_part = part_map[args.board],
		vitis_platform = alveo_default_platform[args.board],
		split_large_fifos = True,
		folding_config_file = args.folding_config_file,
		verify_input_npy = args.input_file,
		verify_expected_output_npy = args.expected_output_file,
		verify_save_full_context = True,
		steps = [
			preprocessing,
			postprocessing,
			make_input_channels_last,
			"step_tidy_up",
			name_nodes,
			"step_qonnx_to_finn",
			name_nodes,
			streamline_function,
			convert_to_hw_function,
			"step_create_dataflow_partition",
			"step_specialize_layers",
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
		generate_outputs = generate_outputs,
		verify_steps = verify_steps
	)

	build.build_dataflow_cfg(args.onnx_model, cfg_build)

if __name__ == "__main__":
	main()
