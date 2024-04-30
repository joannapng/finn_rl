from ast import mod
from operator import contains
import torch
import time
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import VerificationStepType
from finn.builder.build_dataflow_steps import verify_step
import onnx
import torch
import json
import numpy as np
import os
import shutil
import warnings
from copy import deepcopy
from distutils.dir_util import copy_tree
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import SortGraph
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    ApplyConfig,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.make_input_chanlast import MakeInputChannelsLast
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from qonnx.util.cleanup import cleanup_model
from qonnx.util.config import extract_model_config_to_json
from shutil import copy

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finn.transformation.streamline.absorb as absorb
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.analysis.fpgadataflow.op_and_param_counts import (
    aggregate_dict_keys,
    op_and_param_counts,
)
from finn.analysis.fpgadataflow.res_estimation import (
    res_estimation,
    res_estimation_complete,
)
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    ShellFlowType,
    VerificationStepType,
)
from finn.core.onnx_exec import execute_onnx
from finn.core.rtlsim_exec import rtlsim_exec
from finn.core.throughput_test import throughput_test_rtlsim
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.derive_characteristic import (
    DeriveCharacteristic,
    DeriveFIFOSizes,
)
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
    MinimizeWeightBitWidth,
)
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import (
    InsertAndSetFIFODepths,
    RemoveShallowFIFOs,
    SplitLargeFIFOs,
)
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext
from finn.transformation.fpgadataflow.vitis_build import VitisBuild
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.qonnx.quant_act_to_multithreshold import (
    default_filter_function_generator,
)
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC
from finn.util.basic import (
    get_rtlsim_trace_depth,
    pyverilate_get_liveness_threshold_cycles,
)
from finn.util.pyverilator import verilator_fifosim
from finn.util.test import execute_parent

from finn.util.visualization import showInNetron
from finn.util.pytorch import ToTensor
from brevitas.onnx import export_qonnx

from finn.transformation.fpgadataflow import convert_to_hw_layers as convert
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds

from qonnx.transformation.general import ConvertSubToAdd, ConvertDivToMul
import finn.transformation.streamline.collapse_repeated as collapse
import finn.transformation.streamline.reorder as reorder

def tidy_up(model):
    """Run the tidy-up step on given model. This includes shape and datatype
    inference, constant folding, and giving nodes and tensors better names.
    """

    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())

    return model

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

def make_input_channels_last(model: ModelWrapper, cfg: build.DataflowBuildConfig):
	model = model.transform(MakeInputChannelsLast())
	return model

def streamline_resnet(model: ModelWrapper, cfg: build.DataflowBuildConfig):
	model = model.transform(ConvertSubToAdd())
	model = model.transform(ConvertDivToMul())
	model = model.transform(collapse.CollapseRepeatedMul())
	model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
	model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
	model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
	model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())

	model = model.transform(reorder.MoveMulPastMaxPool())
	model = model.transform(reorder.MoveLinearPastFork())
	model = model.transform(reorder.MoveLinearPastEltwiseAdd())
	model = model.transform(reorder.MoveScalarMulPastConv())
	model = model.transform(reorder.MoveScalarMulPastMatMul())
	model = model.transform(reorder.MoveScalarLinearPastInvariants()) # for the mul before the global average pool
	model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
	model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
	model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())

	model = model.transform(RoundAndClipThresholds())
	if VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
		verify_step(model, cfg, "streamlined_python", need_parent=False)
	
	return model

def convert_to_hw_resnet(model: ModelWrapper, cfg: build.DataflowBuildConfig):
	model = model.transform(InferDataLayouts())
	model = model.transform(convert.InferGlobalAccPoolLayer())
	model = model.transform(convert.InferPool())
	model = model.transform(absorb.AbsorbTransposeIntoFlatten())
	model = model.transform(reorder.MoveScalarLinearPastInvariants())
	model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
	model = model.transform(LowerConvsToMatMul())
	model = model.transform(convert.InferChannelwiseLinearLayer())
	model = model.transform(convert.InferConvInpGen())
	model = model.transform(convert.InferQuantizedMatrixVectorActivation())

	model = model.transform(absorb.AbsorbConsecutiveTransposes())
	model = model.transform(reorder.MoveTransposePastFork())
	model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
	model = model.transform(absorb.AbsorbConsecutiveTransposes())

	model = model.transform(reorder.MoveTransposePastJoinAdd())
	model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
	model = model.transform(reorder.MoveTransposePastFork())
	model = model.transform(absorb.AbsorbConsecutiveTransposes())

	model = model.transform(reorder.MoveTransposePastJoinAdd())
	model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
	model = model.transform(reorder.MoveTransposePastFork())
	model = model.transform(absorb.AbsorbConsecutiveTransposes())

	model = model.transform(InferDataLayouts())
	model = model.transform(RoundAndClipThresholds())
	model = model.transform(convert.InferThresholdingLayer())

	model = model.transform(RemoveCNVtoFCFlatten())
	model = model.transform(InferDataLayouts())
	model = model.transform(convert.InferLabelSelectLayer())

	model = tidy_up(model)
	model = model.transform(convert.InferAddStreamsLayer())
	model = model.transform(convert.InferDuplicateStreamsLayer())

	return model
'''
def streamline(model: ModelWrapper, cfg: build.DataflowBuildConfig):
	transformations = [BatchNormToAffine, ConvertBipolarMatMulToXnorPopcount, Change3DTo4DTensors, ChangeDataLayoutQuantAvgPool2d, 
					 AbsorbChanFirstIntoMatMul, ExtractBiasFromConv, ConvertDivToMul, 
					 ConvertSubToAdd, LowerConvsToMatMul,
					 FoldTransposeIntoQuantInit, RemoveIdentityOps, RemoveCNVtoFCFlatten]
	
	absorb_transformations = [getattr(absorb, transformation) for transformation in dir(absorb) if transformation.startswith('Absorb')]
	collapse_transformations = [getattr(collapse, transformation) for transformation in dir(collapse) if transformation.startswith('Collapse') and transformation != 'CollapseRepeatedOp']
	reorder_transformations = [getattr(reorder, transformation) for transformation in dir(reorder) if (transformation.startswith('Make') or transformation.startswith('Move')) \
							and transformation != 'MoveOpPastFork' and transformation != 'MoveIdenticalOpPastJoinOp' and transformation != 'MoveScalarLinearPastInvariants']
	round_transformations = [getattr(round, transformation) for transformation in dir(round) if transformation.startswith('Round')]
	sign_transformations = [getattr(sign, transformation) for transformation in dir(sign) if transformation.startswith('Convert')]
	
	streamlining_transformations = transformations + absorb_transformations + reorder_transformations + \
			collapse_transformations + round_transformations + \
			sign_transformations
		
	model_was_changed = True
	while model_was_changed:
		model_was_changed = False
		for transformation in streamlining_transformations:
			old_model = model.model.SerializeToString()
			model = model.transform(transformation())
			new_model = model.model.SerializeToString()
			if model_was_changed == False:
				model_was_changed = old_model != new_model
			model = model.transform(RemoveIdentityOps())
			model = model.transform(GiveUniqueNodeNames())
			model = model.transform(GiveReadableTensorNames())
			model = model.transform(InferDataTypes())
	
	
	model = model.transform(InferDataLayouts())
	model = model.transform(RemoveUnusedTensors())

	#if VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
	#	verify_step(model, cfg, "streamlined_python", need_parent=False)
	
	return model


def convert_to_hw(model: ModelWrapper, cfg: build.DataflowBuildConfig):
	transformations = [BatchNormToAffine, ConvertBipolarMatMulToXnorPopcount, Change3DTo4DTensors, ChangeDataLayoutQuantAvgPool2d, 
					 AbsorbChanFirstIntoMatMul, ExtractBiasFromConv, ConvertDivToMul, 
					 ConvertSubToAdd, LowerConvsToMatMul,
					 FoldTransposeIntoQuantInit, RemoveIdentityOps, RemoveCNVtoFCFlatten]
	
	absorb_transformations = [getattr(absorb, transformation) for transformation in dir(absorb) if transformation.startswith('Absorb')]
	collapse_transformations = [getattr(collapse, transformation) for transformation in dir(collapse) if transformation.startswith('Collapse') and transformation != 'CollapseRepeatedOp']
	reorder_transformations = [getattr(reorder, transformation) for transformation in dir(reorder) if (transformation.startswith('Make') or transformation.startswith('Move')) \
							and transformation != 'MoveOpPastFork' and transformation != 'MoveIdenticalOpPastJoinOp' and transformation != 'MoveScalarLinearPastInvariants']
	round_transformations = [getattr(round, transformation) for transformation in dir(round) if transformation.startswith('Round')]
	sign_transformations = [getattr(sign, transformation) for transformation in dir(sign) if transformation.startswith('Convert')]
	
	streamlining_transformations = transformations + absorb_transformations + reorder_transformations + \
			collapse_transformations + round_transformations + \
			sign_transformations
	
	hls_transformations = [getattr(convert, transformation) for transformation in dir(convert) if transformation.startswith('Infer')]
	
	model_was_changed = True
	while model_was_changed:
		model_was_changed = False

		for transformation in hls_transformations:
			old_model = model.model.SerializeToString()
			model = model.transform(transformation())
			new_model = model.model.SerializeToString()
			if model_was_changed == False:
				model_was_changed = old_model != new_model
			model = model.transform(transformation())
		
		for transformation in streamlining_transformations:
			old_model = model.model.SerializeToString()
			model = model.transform(transformation())
			new_model = model.model.SerializeToString()
			if model_was_changed == False:
				model_was_changed = old_model != new_model
			model = model.transform(RemoveIdentityOps())
			model = model.transform(GiveUniqueNodeNames())
			model = model.transform(GiveReadableTensorNames())
			model = model.transform(InferDataTypes())
		

	model = model.transform(GiveUniqueNodeNames())
	model = model.transform(GiveReadableTensorNames())
	model = model.transform(InferDataLayouts())
	
	return model
'''

def name_nodes(model: ModelWrapper, cfg: build.DataflowBuildConfig):
	model = model.transform(GiveUniqueNodeNames())
	model = model.transform(GiveReadableTensorNames())

	return model