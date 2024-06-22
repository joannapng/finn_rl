from brevitas.export import export_qonnx
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat, Int16Bias
from brevitas.nn import QuantConv1d
import torch
from urllib3 import disable_warnings 
float_inp = torch.randn(1, 2, 5)

quant_conv_4b8b = QuantConv1d(
    2, 4, 3, bias=True, weight_bit_width=4,
    input_quant=Int8ActPerTensorFloat,
    output_quant=Int8ActPerTensorFloat,
    bias_quant=Int16Bias)

output_path = 'brevitas_onnx_conv4b8b.onnx'
export_qonnx(quant_conv_4b8b, input_t=float_inp, export_path=output_path, disable_warnings=False)
