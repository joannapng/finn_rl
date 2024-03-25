import argparse
import onnx
import brevitas.onnx as bo
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.custom_op.registry import getCustomOp
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
from finn.transformation.qonnx.infer_quant_avg_pool_2d import AvgPoolAndTruncToQuantAvgPool

import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition

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
    model = model.transform(AvgPoolAndTruncToQuantAvgPool())
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

# hls_conversion
def hls_conversion(model, mem_mode = "const"):
    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(to_hls.InferBinaryMatrixVectorActivation(mem_mode))
    model = model.transform(to_hls.InferQuantizedMatrixVectorActivation(mem_mode))
    model = model.transform(to_hls.InferVectorVectorActivation())
    #model = model.transform(to_hls.InferThresholdingLayer())
    #model = model.transform(to_hls.InferStreamingMaxPool())
    #model = model.transform(to_hls.InferPool_Batch())
    #model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(InferDataLayouts())
    model = tidy_up(model)

    return model

def create_dataflow_partition(model_name):
    model = ModelWrapper(model_name)
    parent_model = model.transform(CreateDataflowPartition())
    node = parent_model.get_nodes_by_op_type("StreamingFCLayer_Batch")[0]
    node = getCustomOp(node)
    name = node.get_nodeattr("model")
    dataflow_model = ModelWrapper(name)
    return dataflow_model

def folding(model):
    fc_layers = model.get_nodes_by_op_type("StreamingFCLayer_Batch")
    # (PE, SIMD, in_fifo_depth, out_fifo_depth, ramstyle) for each layer
    # Test Divided by two the PE and in_fifo_depth
    config = [
        (8, 8, 8, 8, "block"),
        (8, 8, 8, 8, "auto"),
        (8, 8, 8, 8, "auto"),
        (8, 8, 8, 8, "distributed"),
    ]
    for fcl, (pe, simd, ififo, ofifo, ramstyle) in zip(fc_layers, config):
        fcl_inst = getCustomOp(fcl)
        fcl_inst.set_nodeattr("PE", pe)
        fcl_inst.set_nodeattr("SIMD", simd)
        fcl_inst.set_nodeattr("inFIFODepth", ififo)
        fcl_inst.set_nodeattr("outFIFODepth", ofifo)
        fcl_inst.set_nodeattr("ram_style", ramstyle)
        print("MW="+str(fcl_inst.get_nodeattr("MW")))
        print("SIMD="+str(fcl_inst.get_nodeattr("SIMD")))
        print("---")
    return model

target_clk_ns = 10

from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild

def hardware_generation(model, platform = "ZCU102", target_clk_ns = 10):
    model = model.transform(ZynqBuild(platform = platform, period_ns = target_clk_ns))
    return model

def main():
    args = parser.parse_args()

    model = ModelWrapper(args.onnx_model)
    model = cleanup_model(model)
    model = model.transform(ConvertQONNXtoFINN())
    model = tidy_up(model)
    model = post_processing(model)
    model = streamline(model)
    model = hls_conversion(model)
    model.save('.'.join(args.onnx_model.split('.')[:-1]) + '_hls.onnx')
    model = create_dataflow_partition('.'.join(args.onnx_model.split('.')[:-1]) + '_hls.onnx')
    model = folding(model)
    model = hardware_generation(model)

if __name__=="__main__":
    main()
