import argparse
import onnx
import brevitas.onnx as bo
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from qonnx.transformation.insert_topk import InsertTopK

import finn.transformation.streamline.absorb as absorb
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline.reorder          import MoveScalarLinearPastInvariants
from finn.transformation.streamline                  import Streamline
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from qonnx.transformation.infer_data_layouts          import InferDataLayouts
from qonnx.transformation.general                     import RemoveUnusedTensors

parser = argparse.ArgumentParser(description='Transform input onnx model')
parser.add_argument('--onnx-model', required = True, type = str, help = 'ONNX model to transform using FINN compiler')

def tidy_up(model):
	model = model.transform(InferShapes())
	model = model.transform(FoldConstants())
	model = model.transform(GiveUniqueNodeNames())
	model = model.transform(GiveReadableTensorNames())
	model = model.transform(InferDataTypes())
	model = model.transform(RemoveStaticGraphInputs())
	return model

def post_processing(model):
    model = model.transform(InsertTopK(k=1))
    model = tidy_up(model)
    return model

def streamline(model):
    model = model.transform(MoveScalarLinearPastInvariants())
    model = model.transform(Streamline())
    # Absorb add and mul in thresholds
    model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    # Absorb add-mul in top-k
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    model = model.transform(RoundAndClipThresholds())
    # Tidy-up
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())
    return model
  
def main():
	args = parser.parse_args()

	model = ModelWrapper(args.onnx_model)
	model = cleanup_model(model)
	model = model.transform(ConvertQONNXtoFINN())
	model = tidy_up(model)
	model = post_processing(model)
	model = streamline(model)

if __name__=="__main__":
	main()
