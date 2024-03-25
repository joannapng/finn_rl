from distutils.command import clean
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from qonnx.transformation.insert_topk import InsertTopK

from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline import Streamline
import finn.transformation.streamline.absorb as absorb
import finn.transformation.streamline.collapse_repeated as collapse
import finn.transformation.streamline.reorder as reorder
import finn.transformation.streamline.round_thresholds as round
import finn.transformation.streamline.sign_to_thres as sign

class Exporter:
	def __init__(self, model_name):
		# Convert Model from QONNX to FINN-ONNX (all bit widths must be under 8 bit)
		self.model_name = model_name
		self.model = ModelWrapper(self.model_name)
		self.model = cleanup_model(self.model) # VERY IMPORTANT TO CLEANUP MODEL
		print(f'\033[1;32mConverting model {self.model_name} from QONNX to FINN-ONNX\033[1;0m')
		self.model = self.model.transform(ConvertQONNXtoFINN())
		self.model.save('.'.join(self.model_name.split('.')[:-1]) + '_finn-onnx.onnx')
		print('\033[1;32mFinished converting model from QONNX to FINN-ONNX\033[1;0m')
	
	def tidy_up(self, model_name = None):
		print('\033[1;32mBeginning tidy up transformations\033[1;0m')

		if (model_name is not None):
			self.model = ModelWrapper(model_name)
		
		self.model = self.model.transform(InferShapes())
		self.model = self.model.transform(FoldConstants())
		self.model = self.model.transform(GiveUniqueNodeNames())
		self.model = self.model.transform(GiveReadableTensorNames())
		self.model = self.model.transform(InferDataTypes())
		self.model = self.model.transform(RemoveStaticGraphInputs())

		if (model_name) is not None:
			self.model.save('.'.join(model_name.split('.')[:-1]) + '_tidy.onnx')
		else:
			self.model.save(('.'.join(self.model_name.split('.')[:-1]) + '_tidy.onnx'))
		
		print('\033[1;32mFinished tidy up transformations\033[1;0m')

	def post_processing(self, model_name = None):
		print('\033[1;32mBeginning post-processing transformations\033[1;0m')

		if (model_name is not None):
			self.model = ModelWrapper(model_name)
	

		if (model_name) is not None:
			self.model.save('.'.join(model_name.split('.')[:-1]) + '_post.onnx')
		else:
			self.model.save(('.'.join(self.model_name.split('.')[:-1]) + '_post.onnx'))
		
		print('\033[1;32mFinished post-processing transformations\033[1;0m')

	def streamline(self, model_name = None):
		print('\033[1;32mBeginning streamlining transformations\033[1;0m')

		if (model_name is not None):
			self.model = ModelWrapper(model_name)
		
		absorb_transformations = [getattr(absorb, transformation) for transformation in dir(absorb) if transformation.startswith('Absorb')]
		collapse_transformations = [getattr(collapse, transformation) for transformation in dir(collapse) if transformation.startswith('Collapse') and transformation != 'CollapseRepeatedOp']
		reorder_transformations = [getattr(reorder, transformation) for transformation in dir(reorder) if (transformation.startswith('Make') or transformation.startswith('Move')) and transformation != 'MoveOpPastFork' and transformation != 'MoveIdenticalOpPastJoinOp']
		round_transformations = [getattr(round, transformation) for transformation in dir(round) if transformation.startswith('Round')]
		sign_transformations = [getattr(sign, transformation) for transformation in dir(sign) if transformation.startswith('Convert')]

		transformations = absorb_transformations + collapse_transformations + reorder_transformations + round_transformations
		for transformation in transformations:
			print(transformation)
			self.model = self.model.transform(transformation())
			self.model = self.model.transform(Streamline())

		if (model_name) is not None:
			self.model.save('.'.join(model_name.split('.')[:-1]) + '_streamlined.onnx')
		else:
			self.model.save(('.'.join(self.model_name.split('.')[:-1]) + '_streamlined.onnx'))
		
		print('\033[1;32mFinished streamlining transformations\033[1;0m')