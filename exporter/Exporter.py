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

from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.change_3d_tensors_to_4d import Change3DTo4DTensors
from qonnx.transformation.change_datalayout import ChangeDataLayoutQuantAvgPool2d
from qonnx.transformation.channels_last import AbsorbChanFirstIntoMatMul
from qonnx.transformation.extract_conv_bias import ExtractBiasFromConv
from qonnx.transformation.general import ApplyConfig, ConvertDivToMul, ConvertSubToAdd, GiveUniqueNodeNames, RemoveUnusedTensors
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit
from qonnx.transformation.remove import RemoveIdentityOps
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
from qonnx.transformation.infer_data_layouts import InferDataLayouts

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.util.cleanup import cleanup as qonnx_cleanup

from copy import deepcopy
from brevitas.export import export_qonnx


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
	
	model = model.transform(InferDataLayouts())
	model = model.transform(RemoveUnusedTensors())

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

	model = model.transform(GiveUniqueNodeNames())
	model = model.transform(GiveReadableTensorNames())
	model = model.transform(InferDataLayouts())
	
	return model

def name_nodes(model: ModelWrapper, cfg: build.DataflowBuildConfig):
	model = model.transform(GiveUniqueNodeNames())
	model = model.transform(GiveReadableTensorNames())

	return model