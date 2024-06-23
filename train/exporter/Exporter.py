from matplotlib.style import available
import torch
import torch
import numpy as np
import json
import time
from copy import deepcopy

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.core.datatype import DataType
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

import finn.transformation.streamline.absorb as absorb

from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.qonnx.quant_act_to_multithreshold import (
	default_filter_function_generator,
)
from finn.transformation.fpgadataflow.create_dataflow_partition import (
	CreateDataflowPartition,
)

from finn.util.pytorch import ToTensor
from brevitas.onnx import export_qonnx


from finn.transformation.fpgadataflow import convert_to_hw_layers as convert
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
	MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
	MinimizeWeightBitWidth,
)
from finn.analysis.fpgadataflow.res_estimation import (
	res_estimation,
	res_estimation_complete,
)
from finn.transformation.fpgadataflow.set_fifo_depths import (
	InsertAndSetFIFODepths,
	RemoveShallowFIFOs,
	SplitLargeFIFOs,
)

from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO

from qonnx.transformation.general import ConvertSubToAdd, ConvertDivToMul
import finn.transformation.streamline.collapse_repeated as collapse
import finn.transformation.streamline.reorder as reorder
from finn.analysis.fpgadataflow.op_and_param_counts import aggregate_dict_keys
from finn.builder.build_dataflow_config import LargeFIFOMemStyle
from train.exporter.utils import (
	isFeasible,
	set_defaults,
	folding,
	estimate_resources
)

from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit
from qonnx.transformation.remove import RemoveIdentityOps

from finn.transformation.qonnx.fold_quant_weights import FoldQuantWeights
from finn.transformation.qonnx.infer_quant_avg_pool_2d import (
	AvgPoolAndTruncToQuantAvgPool,
)
from finn.transformation.qonnx.quant_act_to_multithreshold import (
	ConvertQuantActToMultiThreshold,
	default_filter_function_generator,
)

platform_path = 'platforms'
platform_files = {}
platform_files['U250'] = f'{platform_path}/u250.json'

def tidy_up(model):
	model = model.transform(InferShapes())
	model = model.transform(FoldConstants())
	model = model.transform(GiveUniqueNodeNames())
	model = model.transform(GiveReadableTensorNames())
	model = model.transform(InferDataTypes())
	model = model.transform(RemoveStaticGraphInputs())
	model = model.transform(GiveUniqueNodeNames())
	model = model.transform(GiveReadableTensorNames())

	return model

def preprocessing(model):
	input_shape = model.get_tensor_shape(model.graph.input[0].name)
	preproc = ToTensor()
	export_qonnx(preproc, torch.randn(input_shape), "preproc.onnx", opset_version = 11)
	qonnx_cleanup("preproc.onnx", out_file = "preproc.onnx")
	preproc_model = ModelWrapper("preproc.onnx")
	preproc_model = preproc_model.transform(ConvertQONNXtoFINN())

	model = model.transform(MergeONNXModels(preproc_model))
	global_inp_name = model.graph.input[0].name
	model.set_tensor_datatype(global_inp_name, DataType["UINT8"])
	model = tidy_up(model)
	return model

def postprocessing(model):
	model = model.transform(InsertTopK(k=1))
	model = tidy_up(model)
	return model

def make_input_channels_last(model):
	model = model.transform(MakeInputChannelsLast())
	return model

def qonnx_to_finn(model):
	q_count = 0
	for op_type in ["BinaryQuant", "Quant", "Trunc"]:
		q_count += len(model.get_nodes_by_op_type(op_type))
	if q_count == 0:
		return model
	
	model = model.transform(GiveUniqueNodeNames())
	model = model.transform(GiveReadableTensorNames())
	model = cleanup_model(model)
	model = model.transform(
		ConvertQONNXtoFINN(
			filter_function = default_filter_function_generator(
				max_multithreshold_bit_width = 8
			)
		)
	)

	return model

def create_dataflow_partition(model):
	parent_model = model.transform(
		CreateDataflowPartition()
	)

	sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
	sdp_node = sdp_nodes[0]
	sdp_node = getCustomOp(sdp_node)
	dataflow_model_filename = sdp_node.get_nodeattr("model")
	model = ModelWrapper(dataflow_model_filename)

	return model

def specialize_layers(model, fpga_part):
	model = model.transform(SpecializeLayers(fpga_part))
	model = model.transform(InferShapes())
	model = model.transform(InferDataTypes())
	graph = model.graph
	for node in graph.node:
		if "MVAU" in node.op_type:
			print(node.op_type)
	return model

def set_folding(model, board):
	model = model.transform(GiveUniqueNodeNames())
	model = model.transform(GiveReadableTensorNames())

	model = set_defaults(model)
	f = open(platform_files[board], 'r')
	available_resources = json.load(f)['resources']
	
	model, max_cycles, avg_util, max_util, feasible = folding(model, available_resources)

	if not feasible:
		return model, 1000000, avg_util, max_util
	else:
		hw_attrs = [
		"PE",
		"SIMD",
		"parallel_window",
		"ram_style",
		"resType",
		"mem_mode",
		"runtime_writeable_weights",
		"depth_trigger_uram",
		"depth_trigger_bram",
		"inFIFODepths",
		"outFIFODepths"
		]

		extract_model_config_to_json(model, "auto_folding_config.json", hw_attrs)
		return model, max_cycles, avg_util, max_util

def apply_folding_config(model):
	model = model.transform(GiveUniqueNodeNames())
	model = model.transform(ApplyConfig("auto_folding_config.json"))

	return model

def minimize_bit_width(model):
	model = model.transform(MinimizeWeightBitWidth())
	model = model.transform(MinimizeAccumulatorWidth())
	model = model.transform(InferDataTypes())

	return model

def set_fifo_depths(model):
	extw_optypes = ["MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl"]

	model = model.transform(GiveUniqueNodeNames())
	model = model.transform(GiveReadableTensorNames())

	for node in model.graph.node:
		op_type = node.op_type

		node = getCustomOp(node)
		ifd = node.get_nodeattr("inFIFODepths")
		ofd = node.get_nodeattr("outFIFODepths")

		for i in range(len(ifd)):
			ifd[i] = np.prod(node.get_folded_input_shape(i)[:-1])
			
		for o in range(len(ofd)):
			ofd[o] = np.prod(node.get_folded_output_shape(o)[:-1])
			
		node.set_nodeattr("inFIFODepths", ifd)
		node.set_nodeattr("outFIFODepths", ofd)

	model = model.transform(InsertDWC())
	model = model.transform(InsertFIFO(create_shallow_fifos = True))
		
	model = model.transform(GiveUniqueNodeNames())
	model = model.transform(GiveReadableTensorNames())

	return model

def resource_estimates(model):
	layer_resources = model.analysis(res_estimation)
	layer_resources["total"] = aggregate_dict_keys(layer_resources)
	return layer_resources["total"]

def streamline_lenet(model):
	model = model.transform(ConvertSubToAdd())
	model = model.transform(ConvertDivToMul())

	model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
	model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
	model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
	model = model.transform(collapse.CollapseRepeatedMul())
	model = model.transform(reorder.MoveScalarMulPastConv())
	model = model.transform(reorder.MoveScalarMulPastMatMul())
	model = model.transform(collapse.CollapseRepeatedMul())
	model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
	model = model.transform(collapse.CollapseRepeatedMul())
	model = model.transform(reorder.MoveMulPastMaxPool())
	model = model.transform(reorder.MoveScalarLinearPastInvariants())
	model = model.transform(reorder.MoveScalarMulPastConv())
	model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
	model = model.transform(reorder.MoveMulPastMaxPool())
	model = model.transform(reorder.MoveScalarLinearPastInvariants())
	model = model.transform(collapse.CollapseRepeatedMul())

	for i in range(3):
		model = model.transform(reorder.MoveScalarMulPastMatMul())
		model = model.transform(absorb.AbsorbMulIntoMultiThreshold())

	model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())

	return model

def streamline_simple(model):
	model = model.transform(ConvertSubToAdd())
	model = model.transform(ConvertDivToMul())

	model = model.transform(collapse.CollapseRepeatedMul())
	model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
	model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
	model = model.transform(absorb.AbsorbAddIntoMultiThreshold())

	model = model.transform(reorder.MoveScalarMulPastConv())
	model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
	model = model.transform(collapse.CollapseRepeatedMul())
	model = model.transform(reorder.MoveMulPastMaxPool())
	model = model.transform(reorder.MoveScalarLinearPastInvariants())
	model = model.transform(reorder.MoveScalarMulPastMatMul())
	model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
	model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())

	return model

def streamline_resnet(model):
	model = model.transform(ConvertSubToAdd())
	model = model.transform(ConvertDivToMul())
	model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
	model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())

	model = model.transform(collapse.CollapseRepeatedMul())
	model = model.transform(reorder.MoveLinearPastFork())
	model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
	model = model.transform(collapse.CollapseRepeatedMul())

	for i in range(2):
		model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
		model = model.transform(reorder.MoveScalarMulPastConv())
		model = model.transform(reorder.MoveScalarMulPastMatMul())
		model = model.transform(collapse.CollapseRepeatedMul())

	model = model.transform(reorder.MoveLinearPastEltwiseAdd())
	model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
	model = model.transform(reorder.MoveLinearPastFork())

	model = model.transform(reorder.MoveScalarLinearPastInvariants())
	model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
	model = model.transform(reorder.MoveScalarMulPastMatMul())
	model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
	model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
	model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
	model = model.transform(RoundAndClipThresholds())
	model = model.transform(InferDataLayouts())
	model = model.transform(RemoveUnusedTensors())

	return model

def convert_to_hw_resnet(model):
	model = model.transform(InferDataLayouts())
	model = model.transform(convert.InferGlobalAccPoolLayer())
	model = model.transform(convert.InferPool())
	model = model.transform(absorb.AbsorbTransposeIntoFlatten())
	model = model.transform(reorder.MoveScalarLinearPastInvariants())
	model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
	model = model.transform(LowerConvsToMatMul())
	model = model.transform(convert.InferChannelwiseLinearLayer())
	model = model.transform(convert.InferConvInpGen())
	model = model.transform(convert.InferVectorVectorActivation())
	model = model.transform(convert.InferBinaryMatrixVectorActivation())
	model = model.transform(convert.InferQuantizedMatrixVectorActivation())

	model = model.transform(absorb.AbsorbConsecutiveTransposes())
	model = model.transform(reorder.MoveTransposePastFork())
	model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
	model = model.transform(absorb.AbsorbConsecutiveTransposes())

	for i in range(12):
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

def convert_to_hw_lenet(model):
	model = model.transform(InferDataLayouts())
	model = model.transform(convert.InferPool())
	model = model.transform(LowerConvsToMatMul())
	model = model.transform(convert.InferConvInpGen())
	model = model.transform(convert.InferVectorVectorActivation())
	model = model.transform(convert.InferBinaryMatrixVectorActivation())
	model = model.transform(convert.InferQuantizedMatrixVectorActivation())
	model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
	model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
	model = model.transform(absorb.AbsorbConsecutiveTransposes())

	model = model.transform(InferDataLayouts())
	model = model.transform(convert.InferThresholdingLayer())

	model = model.transform(InferDataLayouts())
	model = model.transform(convert.InferLabelSelectLayer())

	model = model.transform(InferDataLayouts())
	model = model.transform(RemoveCNVtoFCFlatten())

	model = model.transform(RoundAndClipThresholds())
	model = tidy_up(model)

	return model

def convert_to_hw_simple(model):
	model = model.transform(InferDataLayouts())
	model = model.transform(convert.InferPool())

	model = model.transform(LowerConvsToMatMul())
	model = model.transform(convert.InferConvInpGen())
	model = model.transform(convert.InferVectorVectorActivation())
	model = model.transform(convert.InferBinaryMatrixVectorActivation())
	model = model.transform(convert.InferQuantizedMatrixVectorActivation())

	model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
	model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
	model = model.transform(absorb.AbsorbConsecutiveTransposes())

	model = model.transform(InferDataLayouts())
	model = model.transform(convert.InferThresholdingLayer())

	model = model.transform(InferDataLayouts())
	model = model.transform(convert.InferLabelSelectLayer())

	model = model.transform(InferDataLayouts())
	model = model.transform(RemoveCNVtoFCFlatten())

	model = model.transform(RoundAndClipThresholds())
	model = tidy_up(model)

	return model

def name_nodes(model):
	model = model.transform(GiveUniqueNodeNames())
	model = model.transform(GiveReadableTensorNames())

	return model