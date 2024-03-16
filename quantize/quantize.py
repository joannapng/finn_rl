import torch.nn as nn
import brevitas
import brevitas.nn as qnn
import operator

from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
from brevitas.quant.scaled_int import Int16Bias, Int32Bias, Int8ActPerTensorFloat, Uint8ActPerTensorFloat, Int8WeightPerTensorFloat
from brevitas.inject.enum import RestrictValueType
from brevitas.core.function_wrapper.ops_ste import CeilSte
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.nn.quant_mha import QuantMultiheadAttention
from brevitas import config
from brevitas.graph.utils import get_module
from brevitas.graph.base import InsertModuleCallAfter
import torch
from brevitas.graph.base import ModuleToModuleByInstance

ADD_FNS = [torch.add, operator.add, operator.iadd]

ADD_METHODS = ['add', 'add_']
CAT = brevitas.original_cat

PRECISION_PRESERVING_MODULES = (
    nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)

SIGN_PRESERVING_MODULES = (
    nn.Dropout,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d)

LAYER_MAP = {
    nn.AvgPool2d:
        None,
    nn.MultiheadAttention: (
        qnn.QuantMultiheadAttention,
        {
            'in_proj_input_quant': Int8ActPerTensorFloat,
            'in_proj_weight_quant': Int8WeightPerTensorFloat,
            'in_proj_bias_quant': Int32Bias,
            'attn_output_weights_quant': Uint8ActPerTensorFloat,
            'q_scaled_quant': Int8ActPerTensorFloat,
            'k_transposed_quant': Int8ActPerTensorFloat,
            'v_quant': Int8ActPerTensorFloat,
            'out_proj_input_quant': Int8ActPerTensorFloat,
            'out_proj_weight_quant': Int8WeightPerTensorFloat,
            'out_proj_bias_quant': Int32Bias,
            'return_quant_tensor': False}),
    nn.LSTM: (
        qnn.QuantLSTM,
        {
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'io_quant': Int8ActPerTensorFloat,
            'gate_acc_quant': Int8ActPerTensorFloat,
            'sigmoid_quant': Uint8ActPerTensorFloat,
            'tanh_quant': Int8ActPerTensorFloat,
            'cell_state_quant': Int8ActPerTensorFloat,
            'return_quant_tensor': False}),
    nn.RNN: (
        qnn.QuantRNN,
        {
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'io_quant': Int8ActPerTensorFloat,
            'gate_acc_quant': Int8ActPerTensorFloat,
            'return_quant_tensor': False}),
    nn.Conv1d: (
        qnn.QuantConv1d,
        {
            'input_quant': Int8ActPerTensorFloat,
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': False}),
    nn.Conv2d: (
        qnn.QuantConv2d,
        {
            'input_quant': Int8ActPerTensorFloat,
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': False}),
    nn.ConvTranspose1d: (
        qnn.QuantConvTranspose1d,
        {
            'input_quant': Int8ActPerTensorFloat,
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': False}),
    nn.ConvTranspose2d: (
        qnn.QuantConvTranspose2d,
        {
            'input_quant': Int8ActPerTensorFloat,
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': False}),
    nn.Linear: (
        qnn.QuantLinear,
        {
            'input_quant': Int8ActPerTensorFloat,
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': False})}

BIAS_BIT_WIDTH_MAP = {'int32': Int32Bias, 'int16': Int16Bias}

QUANT_IDENTITY_MAP = {
    'signed':
        (qnn.QuantIdentity, {
            'act_quant': Int8ActPerTensorFloat, 'return_quant_tensor': True}),
    'unsigned':
        (qnn.QuantIdentity, {
            'act_quant': Uint8ActPerTensorFloat, 'return_quant_tensor': True}),}

class Quantizer(object):
    def __init__(
            self,
            model,
            scaling_per_output_channel,
            act_quant_percentile,
            act_quant_type,
            scale_factor_type,
            weight_narrow_range=False,
            act_bit_width=32,
            weight_bit_width=32,
            bias_bit_width=32
    ):
        self.model = model
        act_quant_asym = None
        if act_quant_type == 'asymmetric':
            act_quant_asym = ShiftedUint8ActPerTensorFloat
        
        map = self.update_quant_map(
            LAYER_MAP,
            scale_factor_type,
            bias_bit_width = bias_bit_width,
            scaling_per_output_channel = scaling_per_output_channel,
            act_quant_percentile = act_quant_percentile,
            act_quant_asym = act_quant_asym,
            act_bit_width = act_bit_width,
            weight_bit_width = weight_bit_width,
            weight_narrow_range = weight_narrow_range
        )

    def update_quant_map(
        self,
        map, 
        scale_factor_type,
        bias_bit_width,
        scaling_per_output_channel,
        act_quant_percentile,
        act_quant_asym,
        act_bit_width,
        weight_bit_width,
        weight_narrow_range):
    
        act_kwargs = {'bit_width' : act_bit_width, 'high_percentile_q' : act_quant_percentile}

        if act_quant_asym is not None:
            act_kwargs['act_quant'] = act_quant_asym
            act_kwargs['low_percentile_q'] = 100.0 - act_quant_percentile

        weight_kwargs = {
            'scaling_per_output_channel': scaling_per_output_channel,
            'bit_width': weight_bit_width,
            'narrow_range': weight_narrow_range
        }

        scale_factor_dict = {}
        if scale_factor_type == 'po2':
            scale_factor_dict['restrict_scaling_type'] = RestrictValueType.POWER_OF_TWO
            scale_factor_dict['restrict_value_float_to_int_impl'] = CeilSte
        elif scale_factor_type == 'float32':
            scale_factor_dict['restrict_scaling_type'] = RestrictValueType.FP

        act_kwargs.update(scale_factor_dict)
        weight_kwargs.update(scale_factor_dict)

        def weight_kwargs_prefix(prefix):
            return {prefix + k: v for k, v in weight_kwargs.items()}
        
        def act_kwargs_prefix(prefix):
            updated_kwargs = {}
            for k, v in act_kwargs.items():
                key = k

                if prefix != '':
                    key = prefix + key.replace('act_', '')
                updated_kwargs[key] = v
            return updated_kwargs
        
        bias_quant = BIAS_BIT_WIDTH_MAP['int' + str(bias_bit_width)]

        for k, v in map.items():
            if v is None:
                # Non quantized layer, continue
                continue
            if issubclass(v[0], QuantWBIOL):
                map[k][1].update(weight_kwargs_prefix('weight_'))
                map[k][1]['bias_quant'] = bias_quant
                if act_quant_asym is not None:
                    map[k][1]['return_quant_tensor'] = False
                if 'input_quant' in v[1].keys():
                    map[k][1].update(act_kwargs_prefix('input_'))
            elif v[0] == QuantMultiheadAttention:
                map[k][1].update(weight_kwargs_prefix('in_proj_'))
                map[k][1].update(weight_kwargs_prefix('out_proj_'))
                map[k][1].update(act_kwargs_prefix('attn_output_weights_'))
                map[k][1].update(act_kwargs_prefix('q_scaled_'))
                map[k][1].update(act_kwargs_prefix('k_transposed_'))
                map[k][1].update(act_kwargs_prefix('v_'))
                map[k][1].update(act_kwargs_prefix('out_proj_input_'))
                map[k][1]['in_proj_bias_quant'] = bias_quant
                map[k][1]['out_proj_bias_quant'] = bias_quant
                if act_quant_asym is not None:
                    map[k][1]['return_quant_tensor'] = False
                if 'in_proj_input_quant' in v[1].keys():
                    # Add kwargs arguments to input_quant, if present
                    map[k][1].update(act_kwargs_prefix('in_proj_input_'))
            elif 'act_quant' in v[1].keys():
                # Add kwargs argument to activation quantizers.
                v[1].update(act_kwargs_prefix(''))

        return map
    
    def layerwise_quantize(self, model, idx, input_bit_width, weight_bit_width, compute_layer_map = LAYER_MAP):
        ignore_missing_keys_state = config.IGNORE_MISSING_KEYS
        config.IGNORE_MISSING_KEYS = True
        training_state = model.training
        model.eval()
        model = self.layer_handler(model, idx, input_bit_width, weight_bit_width, layer_map = compute_layer_map, requantize_output = False)
        model.train(training_state)
        config.IGNORE_MISSING_KEYS = ignore_missing_keys_state
        return model
    
    def layer_handler(self,
                      model,
                      idx, 
                      input_bit_width,
                      weight_bit_width,
                      layer_map,
                      requantize_output,
                      quant_identity_map=dict(),
                      quant_act_map=dict(),
                      usigned_act_tuple=dict()):

        for i, node in enumerate(model.graph.nodes):
            rewriters = []
            if node.op == 'call_module' and i == idx:
                module = get_module(model, node.target)
                if isinstance(module, tuple(layer_map.keys())):
                    if layer_map[type(module)] is not None:
                        quant_module_class, quant_module_kwargs = layer_map[type(module)]
                        quant_module_kwargs['input_bit_width'] = input_bit_width
                        quant_module_kwargs['weight_bit_width'] = weight_bit_width
                        rewriter = ModuleToModuleByInstance(
                            module, quant_module_class, **quant_module_kwargs
                        )
                        rewriters.append(rewriter)

            for rewriter in rewriters:
                model = rewriter.apply(model)
        return model

        
        
        
def are_inputs_quantized_and_aligned(model, node, quantized_modules_list, quant_act_map, same_sign):
    """
    Check if the inputs to `node` are quantized and aligned.
    If same_sign=True, aligned means that the inputs should have same sign and scale factor.
    Otherwise, they need to have only the same scale factors.
    If none of the previous conditions are met (e.g., FP input, or not aligned scales), the function
    returns False.
    """
    for inp_node in node.all_input_nodes:
        if inp_node.op == 'call_module':
            inp_module = get_module(model, inp_node.target)
            if isinstance(inp_module, tuple(quant_act_map.keys())):
                quantized_modules_list.append(None)
            elif isinstance(inp_module, tuple(PRECISION_PRESERVING_MODULES)) and (
                    not same_sign or
                (same_sign and isinstance(inp_module, tuple(SIGN_PRESERVING_MODULES)))):
                are_inputs_quantized_and_aligned(
                    model, inp_node, quantized_modules_list, quant_act_map, same_sign)
            elif hasattr(inp_module, 'act_quant'):
                aq = inp_module.act_quant
                if _tensor_quant_in_list(aq, quantized_modules_list, same_sign):
                    continue
                quantized_modules_list.append(aq)
            else:
                quantized_modules_list.append(None)
        elif inp_node.op == 'call_function':
            if inp_node.target in [torch.reshape, torch.flatten, torch.transpose]:
                are_inputs_quantized_and_aligned(
                    model, inp_node, quantized_modules_list, quant_act_map, same_sign)
            elif inp_node.target is CAT:
                are_inputs_quantized_and_aligned(
                    model, inp_node, quantized_modules_list, quant_act_map, True)
            elif inp_node.target in ADD_FNS:
                are_inputs_quantized_and_aligned(
                    model, inp_node, quantized_modules_list, quant_act_map, False)
            else:
                quantized_modules_list.append(None)
        elif inp_node.op == 'call_method':
            if inp_node.target in ['view', 'reshape', 'flatten', 't', 'permute']:
                are_inputs_quantized_and_aligned(
                    model, inp_node, quantized_modules_list, quant_act_map, same_sign)
            elif inp_node.target in ADD_METHODS:
                are_inputs_quantized_and_aligned(
                    model, inp_node, quantized_modules_list, quant_act_map, False)
            else:
                quantized_modules_list.append(None)
    if None in quantized_modules_list:
        return False
    elif len(quantized_modules_list) > 1:
        return False
    else:
        return True
    
def are_inputs_unsigned(model, node, is_unsigned_list, quant_act_map, unsigned_act_tuple):
    for inp_node in node.all_input_nodes:
        if inp_node.op == 'call_module':
            inp_module = get_module(model, inp_node.target)
            if isinstance(inp_module, tuple(quant_act_map.keys())) and isinstance(
                    inp_module, unsigned_act_tuple):
                is_unsigned_list.append(True)
            elif isinstance(inp_module, tuple(SIGN_PRESERVING_MODULES)):
                are_inputs_unsigned(
                    model, inp_node, is_unsigned_list, quant_act_map, unsigned_act_tuple)
            elif hasattr(inp_module, 'is_quant_act_signed'):
                is_unsigned_list.append(not inp_module.is_quant_act_signed)
            else:
                is_unsigned_list.append(False)
        elif inp_node.op == 'call_function':
            if inp_node.target in [torch.reshape, torch.flatten, torch.transpose, CAT] + ADD_FNS:
                are_inputs_unsigned(
                    model, inp_node, is_unsigned_list, quant_act_map, unsigned_act_tuple)
            else:
                is_unsigned_list.append(False)
        elif inp_node.op == 'call_method':
            if inp_node.target in ['view', 'reshape', 'flatten', 't', 'permute'] + ADD_METHODS:
                are_inputs_unsigned(
                    model, inp_node, is_unsigned_list, quant_act_map, unsigned_act_tuple)
            else:
                is_unsigned_list.append(False)
    return all(is_unsigned_list)

def _tensor_quant_in_list(act_quant, module_list, same_sign):
    tq = act_quant.fused_activation_quant_proxy.tensor_quant
    for m in module_list:
        if m is None:
            continue
        m_tq = m.fused_activation_quant_proxy.tensor_quant
        if same_sign and m_tq is tq:
            return True
        elif not same_sign and m_tq.scaling_impl is tq.scaling_impl and m_tq.int_scaling_impl is tq.int_scaling_impl:
            return True
    return False