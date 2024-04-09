import copy
from copy import deepcopy
from distutils.command import clean
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.change_3d_tensors_to_4d import Change3DTo4DTensors
from qonnx.transformation.change_datalayout import ChangeDataLayoutQuantAvgPool2d
from qonnx.transformation.channels_last import AbsorbChanFirstIntoMatMul
from qonnx.transformation.extract_conv_bias import ExtractBiasFromConv
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.transformation.general import ConvertDivToMul
from qonnx.transformation.general import ConvertSubToAdd
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.make_input_chanlast import MakeInputChannelsLast
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit
from qonnx.transformation.remove import RemoveIdentityOps
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten

from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline import Streamline
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
import finn.transformation.streamline.absorb as absorb
import finn.transformation.streamline.collapse_repeated as collapse
import finn.transformation.streamline.reorder as reorder
import finn.transformation.streamline.round_thresholds as round
import finn.transformation.streamline.sign_to_thres as sign
import finn.transformation.fpgadataflow.convert_to_hw_layers as convert
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferBinaryMatrixVectorActivation, InferQuantizedMatrixVectorActivation, InferThresholdingLayer, InferVectorVectorActivation

from finn.transformation.fpgadataflow.vitis_build import VitisBuild
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

from finn.util.basic import part_map, alveo_default_platform
from qonnx.custom_op.registry import getCustomOp

from samo.backend.finn.import parser
from samo.backend.finn.export import export
from samo.optimiser.annealing import SimulatedAnnealing

mem_mode_transformations = [InferBinaryMatrixVectorActivation, InferQuantizedMatrixVectorActivation, InferThresholdingLayer, InferVectorVectorActivation]

class Exporter:
	def __init__(self, model_name = None):
		# Convert Model from QONNX to FINN-ONNX (all bit widths must be under 8 bit)
		self.model_name = model_name

		if self.model_name is not None:
			self.model = ModelWrapper(self.model_name)
			self.model = cleanup_model(self.model) # VERY IMPORTANT TO CLEANUP MODEL
			print(f'\033[1;32mConverting model {self.model_name} from QONNX to FINN-ONNX\033[1;0m')
			self.model = self.model.transform(ConvertQONNXtoFINN())
			self.model.save('.'.join(self.model_name.split('.')[:-1]) + '_finn-onnx.onnx')
			print('\033[1;32mFinished converting model from QONNX to FINN-ONNX\033[1;0m')
	
	def tidy_up(self, model_name = None, store = True):
		print('\033[1;32mBeginning tidy up transformations\033[1;0m')

		if (model_name is not None):
			self.model = ModelWrapper(model_name)
		
		self.model = self.model.transform(InferShapes())
		self.model = self.model.transform(FoldConstants())
		self.model = self.model.transform(GiveUniqueNodeNames())
		self.model = self.model.transform(GiveReadableTensorNames())
		self.model = self.model.transform(InferDataTypes())
		self.model = self.model.transform(RemoveStaticGraphInputs())

		if store:
			if (model_name) is not None:
				self.model.save('.'.join(model_name.split('.')[:-1]) + '_tidy.onnx')
			else:
				self.model.save(('.'.join(self.model_name.split('.')[:-1]) + '_tidy.onnx'))
		
		print('\033[1;32mFinished tidy up transformations\033[1;0m')

	def post_processing(self, model_name = None):
		print('\033[1;32mBeginning post-processing transformations\033[1;0m')

		if (model_name is not None):
			self.model = ModelWrapper(model_name)

		self.model = self.model.transform(InsertTopK(k=1))
	
		if (model_name) is not None:
			self.model.save('.'.join(model_name.split('.')[:-1]) + '_post.onnx')
		else:
			self.model.save(('.'.join(self.model_name.split('.')[:-1]) + '_post.onnx'))
		
		print('\033[1;32mFinished post-processing transformations\033[1;0m')

	def streamline(self, model_name = None):
		print('\033[1;32mBeginning streamlining transformations\033[1;0m')

		if (model_name is not None):
			self.model = ModelWrapper(model_name)
		
		transformations = [BatchNormToAffine, ConvertBipolarMatMulToXnorPopcount, Change3DTo4DTensors, ChangeDataLayoutQuantAvgPool2d, 
					 AbsorbChanFirstIntoMatMul, ExtractBiasFromConv, GemmToMatMul, ConvertDivToMul, 
					 ConvertSubToAdd, LowerConvsToMatMul, MakeInputChannelsLast,
					 FoldTransposeIntoQuantInit, RemoveIdentityOps, RemoveCNVtoFCFlatten]

		absorb_transformations = [getattr(absorb, transformation) for transformation in dir(absorb) if transformation.startswith('Absorb')]
		collapse_transformations = [getattr(collapse, transformation) for transformation in dir(collapse) if transformation.startswith('Collapse') and transformation != 'CollapseRepeatedOp']
		reorder_transformations = [getattr(reorder, transformation) for transformation in dir(reorder) if (transformation.startswith('Make') or transformation.startswith('Move')) and transformation != 'MoveOpPastFork' and transformation != 'MoveIdenticalOpPastJoinOp']
		round_transformations = [getattr(round, transformation) for transformation in dir(round) if transformation.startswith('Round')]
		sign_transformations = [getattr(sign, transformation) for transformation in dir(sign) if transformation.startswith('Convert')]

		self.streamlining_transformations = transformations + reorder_transformations + absorb_transformations + collapse_transformations + round_transformations + sign_transformations
		
		model_was_changed = True
		while model_was_changed:
			self.prev_model = deepcopy(self.model)
			model_was_changed = False
			for transformation in self.streamlining_transformations:
				print(transformation)
				'''
				if (transformation in mem_mode_transformations):
					self.model = self.model.transform(transformation(mem_mode = "decoupled"))
				else:
					self.model = self.model.transform(transformation())
				'''
				self.model = self.model.transform(transformation())
				self.model = self.model.transform(Streamline())
			
			if (self.prev_model.model != self.model.model):
				model_was_changed = True
			
			self.tidy_up(store = False)

		if (model_name) is not None:
			self.model.save('.'.join(model_name.split('.')[:-1]) + '_streamlined.onnx')
		else:
			self.model.save(('.'.join(self.model_name.split('.')[:-1]) + '_streamlined.onnx'))
		
		print('\033[1;32mFinished streamlining transformations\033[1;0m')

	def hls_conversion(self, model_name = None):
		print('\033[1;32mBeginning HLS conversion\033[1;0m')

		if (model_name is not None):
			self.model = ModelWrapper(model_name)
		
		hls_transformations = [getattr(convert, transformation) for transformation in dir(convert) if transformation.startswith('Infer')]

		model_was_changed = True
		while model_was_changed:
			self.prev_model = deepcopy(self.model)
			model_was_changed = False
			for transformation in hls_transformations:
				print(transformation)
				self.model = self.model.transform(transformation())
			
			for transformation in self.streamlining_transformations:
				print(transformation)
				'''
				if (transformation in mem_mode_transformations):
					self.model = self.model.transform(transformation(mem_mode = "decoupled"))
				else:
					self.model = self.model.transform(transformation())
				'''
				self.model = self.model.transform(transformation())
				self.model = self.model.transform(Streamline())

			if (self.prev_model.model != self.model.model):
				model_was_changed = True
			
			self.tidy_up(store = False)


		if (model_name) is not None:
			self.model.save('.'.join(model_name.split('.')[:-1]) + '_hls.onnx')
		else:
			self.model.save(('.'.join(self.model_name.split('.')[:-1]) + '_hls.onnx'))
		
		print('\033[1;32mFinished HLS conversion\033[1;0m')

	def create_dataflow_partition(self, model_name = None):
		print('\033[1;32mCreating dataflow partition\033[1;0m')

		if (model_name is not None):
			self.model = ModelWrapper(model_name)
		
		self.parent_model = self.model.transform(CreateDataflowPartition())
		sdp_node = self.parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
		sdp_node = getCustomOp(sdp_node)
		self.dataflow_model_filename = sdp_node.get_nodeattr('model')
		self.dataflow_model = ModelWrapper(self.dataflow_model_filename)
		self.dataflow_model = self.dataflow_model.transform(SpecializeLayers())

		if (model_name) is not None:
			self.dataflow_model.save('.'.join(model_name.split('.')[:-1]) + '_dataflow.onnx')
		else:
			self.dataflow_model.save(('.'.join(self.model_name.split('.')[:-1]) + '_dataflow.onnx'))
		
		print('\033[1;32mFinished dataflow partition\033[1;0m')

	def set_folding(self, model_name = None, platform = "U250", optimizer = "annealing", period_ns = 100):
		if model_name is not None:
			self.model = ModelWrapper(model_name)
		else:
			self.model = self.dataflow_model

		graph = parser.graph(self.dataflow_model, platform, 1000 / 100)
		graph.enable_reconf = False
		graph.objective = "latency"

		for partition in graph.partitions:
			partition.reset()
		
		if optimizer == "annealing":
			opt = SimulatedAnnealing(graph)

		can_split = True
		while can_split:
			can_split = False
			for i in range(len(opt.network.partitions)):
				valid_splits = opt.network.valid_splits(i)
				network_copy = copy.deepcopy(opt.network)
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

		self.dataflow_model = export(self.model)

	
	def generate_hw(self, model_name = None, platform = "U250", period_ns = 100):
		fpga_part = part_map[platform]
		platform = alveo_default_platform[platform]
		self.dataflow_model = self.dataflow_model.transform(VitisBuild(fpga_part, period_ns, platform))
