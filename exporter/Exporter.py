from ast import mod
from operator import contains
import torch
import time
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import VerificationStepType
from finn.builder.build_dataflow_steps import verify_step

from finn.util.pytorch import ToTensor
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline import Streamline
from finn.util.basic import part_map, alveo_default_platform

from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.change_3d_tensors_to_4d import Change3DTo4DTensors
from qonnx.transformation.change_datalayout import ChangeDataLayoutQuantAvgPool2d
from qonnx.transformation.channels_last import AbsorbChanFirstIntoMatMul
from qonnx.transformation.extract_conv_bias import ExtractBiasFromConv
from qonnx.transformation.general import ConvertDivToMul, ConvertSubToAdd, GiveUniqueNodeNames, RemoveUnusedTensors
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.make_input_chanlast import MakeInputChannelsLast
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.transformation.infer_data_layouts import InferDataLayouts

from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs

from finn.transformation.move_reshape import RemoveCNVtoFCFlatten

import finn.transformation.streamline.absorb as absorb
import finn.transformation.streamline.collapse_repeated as collapse
import finn.transformation.streamline.reorder as reorder
import finn.transformation.streamline.round_thresholds as round
import finn.transformation.streamline.sign_to_thres as sign
import finn.transformation.fpgadataflow.convert_to_hw_layers as convert

import json
from samo.backend.finn import parser
from samo.backend.finn.export import export
from samo.optimiser.annealing import SimulatedAnnealing

from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.derive_characteristic import (
	DeriveCharacteristic, 
	DeriveFIFOSizes,
)
from finn.transformation.fpgadataflow.vitis_build import VitisBuild
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.util.cleanup import cleanup as qonnx_cleanup

from copy import deepcopy
from brevitas.export import export_qonnx

from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from finn.util.basic import make_build_dir

from shutil import copy
from distutils.dir_util import copy_tree

def preprocessing(model: ModelWrapper, cfg: build.DataflowBuildConfig):
	input_shape = model.get_tensor_shape(model.graph.input[0].name)
	preproc = ToTensor()
	export_qonnx(preproc, torch.randn(input_shape), "preproc.onnx", opset_version = 11)
	qonnx_cleanup("preproc.onnx", out_file = "preproc.onnx")
	preproc_model = ModelWrapper("preproc.onnx")
	preproc_model = preproc_model.transform(ConvertQONNXtoFINN())

	model = model.transform(MergeONNXModels(preproc_model))
	global_inp_name = model.graph.input[0].name
	model.set_tensor_datatype(global_inp_name, DataType["UINT8"])
	model = model.transform(InferShapes())
	model = model.transform(FoldConstants())
	model = model.transform(GiveUniqueNodeNames())
	model = model.transform(GiveReadableTensorNames())
	model = model.transform(InferDataTypes())
	model = model.transform(RemoveStaticGraphInputs())
	return model

def postprocessing(model: ModelWrapper, cfg: build.DataflowBuildConfig):
	model = model.transform(InsertTopK(k=1))
	model = model.transform(InferShapes())
	model = model.transform(FoldConstants())
	model = model.transform(GiveUniqueNodeNames())
	model = model.transform(GiveReadableTensorNames())
	model = model.transform(InferDataTypes())
	model = model.transform(RemoveStaticGraphInputs())
	return model

def streamline(model: ModelWrapper, cfg: build.DataflowBuildConfig):
	transformations = [BatchNormToAffine, ConvertBipolarMatMulToXnorPopcount, Change3DTo4DTensors, ChangeDataLayoutQuantAvgPool2d, 
					 AbsorbChanFirstIntoMatMul, ExtractBiasFromConv, ConvertDivToMul, 
					 ConvertSubToAdd, LowerConvsToMatMul,
					 FoldTransposeIntoQuantInit, RemoveIdentityOps, RemoveCNVtoFCFlatten]
	
	absorb_transformations = [getattr(absorb, transformation) for transformation in dir(absorb) if transformation.startswith('Absorb')]
	collapse_transformations = [getattr(collapse, transformation) for transformation in dir(collapse) if transformation.startswith('Collapse') and transformation != 'CollapseRepeatedOp']
	reorder_transformations = [getattr(reorder, transformation) for transformation in dir(reorder) if (transformation.startswith('Make') or transformation.startswith('Move')) \
							and transformation != 'MoveOpPastFork' and transformation != 'MoveIdenticalOpPastJoinOp']
	round_transformations = [getattr(round, transformation) for transformation in dir(round) if transformation.startswith('Round')]
	sign_transformations = [getattr(sign, transformation) for transformation in dir(sign) if transformation.startswith('Convert')]
	
	streamlining_transformations = transformations + absorb_transformations + reorder_transformations + \
			collapse_transformations + round_transformations + \
			sign_transformations
		
	model_was_changed = True
	while model_was_changed:
		prev_model = deepcopy(model)
		model_was_changed = False
		for transformation in streamlining_transformations:
			model = model.transform(transformation())
			model = model.transform(Streamline())
		
		if (prev_model.model != model.model):
			model_was_changed = True
		
		model = model.transform(InferShapes())
		model = model.transform(FoldConstants())
		model = model.transform(GiveUniqueNodeNames())
		model = model.transform(GiveReadableTensorNames())
		model = model.transform(InferDataTypes())
		model = model.transform(RemoveStaticGraphInputs())

	if VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
		verify_step(model, cfg, "streamlined_python", need_parent=False)
	
	return model

def convert_to_hw(model: ModelWrapper, cfg: build.DataflowBuildConfig):
	transformations = [BatchNormToAffine, ConvertBipolarMatMulToXnorPopcount, Change3DTo4DTensors, ChangeDataLayoutQuantAvgPool2d, 
					 AbsorbChanFirstIntoMatMul, ExtractBiasFromConv, ConvertDivToMul, 
					 ConvertSubToAdd, LowerConvsToMatMul,
					 FoldTransposeIntoQuantInit, RemoveIdentityOps, RemoveCNVtoFCFlatten]
	
	absorb_transformations = [getattr(absorb, transformation) for transformation in dir(absorb) if transformation.startswith('Absorb')]
	collapse_transformations = [getattr(collapse, transformation) for transformation in dir(collapse) if transformation.startswith('Collapse') and transformation != 'CollapseRepeatedOp']
	reorder_transformations = [getattr(reorder, transformation) for transformation in dir(reorder) if (transformation.startswith('Make') or transformation.startswith('Move')) \
							and transformation != 'MoveOpPastFork' and transformation != 'MoveIdenticalOpPastJoinOp']
	round_transformations = [getattr(round, transformation) for transformation in dir(round) if transformation.startswith('Round')]
	sign_transformations = [getattr(sign, transformation) for transformation in dir(sign) if transformation.startswith('Convert')]
	
	streamlining_transformations = transformations + absorb_transformations + reorder_transformations + \
			collapse_transformations + round_transformations + \
			sign_transformations
	
	hls_transformations = [getattr(convert, transformation) for transformation in dir(convert) if transformation.startswith('Infer')]
	
	model_was_changed = True
	while model_was_changed:
		prev_model = deepcopy(model)
		model_was_changed = False

		for transformation in hls_transformations:
			model = model.transform(transformation())
		
		for transformation in streamlining_transformations:
			model = model.transform(transformation())
			model = model.transform(Streamline())
		
		if (prev_model.model != model.model):
			model_was_changed = True
		
		model = model.transform(InferShapes())
		model = model.transform(FoldConstants())
		model = model.transform(GiveUniqueNodeNames())
		model = model.transform(GiveReadableTensorNames())
		model = model.transform(InferDataTypes())
		model = model.transform(RemoveStaticGraphInputs())
	
	return model

def insert_fifos(model: ModelWrapper, cfg: build.DataflowBuildConfig):
		model = model.transform(InsertDWC())
		model = model.transform(SpecializeLayers())
		model = model.transform(GiveUniqueNodeNames())
		model = model.transform(
			PrepareIP(cfg._resolve_fpga_part(), cfg._resolve_hls_clk_period())
		)
		model = model.transform(HLSSynthIP())
		model = model.transform(PrepareRTLSim())
		model = model.transform(AnnotateCycles())
		period = model.analysis(dataflow_performance)["max_cycles"] + 10
		model = model.transform(DeriveCharacteristic(period))
		model = model.transform(DeriveFIFOSizes())
		model = model.transform(
			InsertFIFO(
				vivado_ram_style="auto",
				max_qsrl_depth=256,
				create_shallow_fifos=True,
			)
		)
		model = model.transform(SpecializeLayers())
		model = model.transform(GiveUniqueNodeNames())
		model = model.transform(GiveReadableTensorNames())

		return model

def set_folding(model: ModelWrapper, cfg: build.DataflowBuildConfig):
	# important for samo (it needs names to label the edges)
	model = model.transform(GiveUniqueNodeNames())
	model = model.transform(GiveReadableTensorNames())

	# transform this appropriately
	platform_file = "/srv/homes/ipanagou/thesis/finn/thesis/code/samo/platforms/u250_1slr.json"
	with open(platform_file, "r") as f:
		platform = json.load(f)

	graph = parser.parse(model, platform, 1000 / cfg._resolve_hls_clk_period())
	graph.enable_reconf = False
	graph.objective = "latency"

	for partition in graph.partitions:
		partition.reset()

	opt = SimulatedAnnealing(graph)

	opt.start_time = time.time()
	can_split = True
	while can_split:
		can_split = False
		for i in range(len(opt.network.partitions)):
			valid_splits = opt.network.valid_splits(i)
			network_copy = deepcopy(opt.network)
			if valid_splits:
				can_split = True
				prev = opt.network.check_constraints()
				opt.network.split(i, valid_splits[0])
				if prev and not opt.network.check_constraints():
					can_split = False
					opt.network = network_copy

	assert opt.network.check_constraints(), "Initial design infeasible"
	
	opt.optimise()

	assert opt.network.check_constraints(), "Optimized design infeasible"

	opt.network.summary()

	model = export(opt.network, model)

	return model

def generate_hw(model: ModelWrapper, cfg: build.DataflowBuildConfig):
		fpga_part = cfg._resolve_fpga_part()
		platform = cfg._resolve_vitis_platform()
		model= model.transform(VitisBuild(fpga_part, cfg._resolve_hls_clk_period(), platform))
		model = model.transform(MakePYNQDriver("alveo"))
		
		deployment_dir = make_build_dir(prefix="pynq_deployment_")
		model.set_metadata_prop("pynq_deployment_dir", deployment_dir)

		# get and copy necessary files
		# .bit and .hwh file
		bitfile = model.get_metadata_prop("bitfile")
		hwh_file = model.get_metadata_prop("hw_handoff")
		deploy_files = [bitfile, hwh_file]

		for dfile in deploy_files:
			if dfile is not None:
				copy(dfile, deployment_dir)

		# driver.py and python libraries
		pynq_driver_dir = model.get_metadata_prop("pynq_driver_dir")
		copy_tree(pynq_driver_dir, deployment_dir)

		return model