import torch.nn as nn
import brevitas
import brevitas.nn as qnn
import operator

from brevitas.core.scaling.standalone import ParameterFromStatsFromParameterScaling
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
from brevitas.quant.experimental.float import Fp8e4m3Act
from brevitas.quant.experimental.float import Fp8e4m3ActPerTensorFloat
from brevitas.quant.experimental.float import Fp8e4m3ActPerTensorFloatMSE
from brevitas.quant.experimental.float import Fp8e4m3WeightPerChannelFloat
from brevitas.quant.experimental.float import Fp8e4m3WeightPerChannelFloatMSE
from brevitas.quant.experimental.float import Fp8e4m3WeightPerTensorFloat
from brevitas.quant.experimental.float import Fp8e4m3WeightPerTensorFloatMSE
from brevitas.quant.fixed_point import Int8ActPerTensorFixedPoint
from brevitas.quant.fixed_point import Int8ActPerTensorFixedPointMSE
from brevitas.quant.fixed_point import Int8WeightPerChannelFixedPoint
from brevitas.quant.fixed_point import Int8WeightPerChannelFixedPointMSE
from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPoint
from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPointMSE
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8ActPerTensorFloatMSE
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat
from brevitas.quant.scaled_int import Int8WeightPerChannelFloatMSE
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerTensorFloatMSE
from brevitas.quant.scaled_int import Int16Bias
from brevitas.quant.scaled_int import Int32Bias
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFixedPoint
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloatMSE
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloatMSE
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloatMSE

from brevitas.graph.standardize import DisableLastReturnQuantTensor


BIAS_BIT_WIDTH_MAP = {32: Int32Bias, 16: Int16Bias, None: None}
UNSIGNED_ACT_TUPLE = (nn.ReLU, nn.ReLU6, nn.Sigmoid, nn.Hardsigmoid)
WEIGHT_QUANT_MAP = {
    'int': {
        'float_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Int8WeightPerTensorFloat, 'asym': ShiftedUint8WeightPerTensorFloat},
                'per_channel': {
                    'sym': Int8WeightPerChannelFloat, 'asym': ShiftedUint8WeightPerChannelFloat}},
            'mse': {
                'per_tensor': {
                    'sym': Int8WeightPerTensorFloatMSE,
                    'asym': ShiftedUint8WeightPerTensorFloatMSE},
                'per_channel': {
                    'sym': Int8WeightPerChannelFloatMSE,
                    'asym': ShiftedUint8WeightPerChannelFloatMSE},},},
        'po2_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Int8WeightPerTensorFixedPoint},
                'per_channel': {
                    'sym': Int8WeightPerChannelFixedPoint},},
            'mse': {
                'per_tensor': {
                    'sym': Int8WeightPerTensorFixedPointMSE},
                'per_channel': {
                    'sym': Int8WeightPerChannelFixedPointMSE}},}},
    'float': {
        'float_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Fp8e4m3WeightPerTensorFloat},
                'per_channel': {
                    'sym': Fp8e4m3WeightPerChannelFloat}},
            'mse': {
                'per_tensor': {
                    'sym': Fp8e4m3WeightPerTensorFloatMSE},
                'per_channel': {
                    'sym': Fp8e4m3WeightPerChannelFloatMSE}}}}}

INPUT_QUANT_MAP = {
    'int': {
        'float_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Int8ActPerTensorFloat, 'asym': ShiftedUint8ActPerTensorFloat}},
            'mse': {
                'per_tensor': {
                    'sym': Int8ActPerTensorFloatMSE, 'asym': ShiftedUint8ActPerTensorFloatMSE}}},
        'po2_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Int8ActPerTensorFixedPoint, 'asym': ShiftedUint8ActPerTensorFixedPoint},
            },
            'mse': {
                'per_tensor': {
                    'sym': Int8ActPerTensorFixedPointMSE}},}},
    'float': {
        'float_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Fp8e4m3ActPerTensorFloat}},
            'mse': {
                'per_tensor': {
                    'sym': Fp8e4m3ActPerTensorFloat},}}}}



class Quantizer(object):
    def __init__(
            self,
            model,
            weight_bit_width,
            act_bit_width,
            bias_bit_width,
            weight_quant_granularity,
            act_quant_percentile,
            act_quant_type,
            scale_factor_type,
            quant_format,
            act_param_method,
            weight_param_method,
            weight_quant_type,
            act_quant_granularity = 'per_tensor',
            uint_sym_act_for_unsigned_values = True
    ):
        self.model = model
        weight_scale_type = scale_factor_type
        act_scale_type = scale_factor_type

        weight_quant_format = quant_format
        act_quant_format = quant_format

        weight_bit_width_dict = {}
        act_bit_width_dict = {}
        weight_bit_width_dict['weight_bit_width'] = weight_bit_width
        act_bit_width_dict['act_bit_width'] = act_bit_width

        quant_layer_map, quant_act_map, quant_identity_map = self.create_quant_maps(
            weight_bit_width = weight_bit_width,
            weight_param_method = weight_param_method,
            weight_scale_type = weight_scale_type,
            weight_quant_type = weight_quant_type,
            weight_quant_granularity = weight_quant_granularity,
            weight_narrow_range = True,
            weight_quant_format = weight_quant_format,
            act_quant_format = act_quant_format,
            uint_sym_act_for_unsigned_values = True,
            act_bit_width = act_bit_width,
            act_scale_type = act_scale_type,
            act_param_method = act_param_method,
            act_quant_type = act_quant_type,
            act_quant_granularity = act_quant_granularity,
            act_quant_percentile = act_quant_percentile
        )

        self.quantize_kwargs = {
            'compute_layer_map' : quant_layer_map,
            'quant_act_map' : quant_act_map,
            'quant_identity_map' : quant_identity_map
        }

        self.model = self.quantize_input(self.model)

    def create_quant_maps(
            self,
            bias_bit_width,
            weight_bit_width,
            weight_param_method,
            weight_scale_type,
            weight_quant_type,
            weight_quant_granularity,
            weight_narrow_range,
            weight_quant_format,
            act_quant_format,
            uint_sym_act_for_unsigned_values = True,
            act_bit_width = None,
            act_scale_type = None,
            act_param_method = None,
            act_quant_type = None,
            act_quant_granularity = None,
            act_quant_percentile = None
    ):
        
        def kwargs_prefix(prefix, weight_kwargs):
            return {prefix + k: v for k, v in weight_kwargs.items()}
        
        weight_bit_width_dict = {'bit_width' : weight_bit_width}
        act_bit_width_dict = {'bit_width': act_bit_width}

        bias_quant = BIAS_BIT_WIDTH_MAP[bias_bit_width] if act_bit_width is not None else None
        weight_quant = WEIGHT_QUANT_MAP[weight_quant_format][weight_scale_type][weight_param_method][weight_quant_granularity][weight_quant_type]
        weight_quant = weight_quant.let(**weight_bit_width_dict)

        act_quant = INPUT_QUANT_MAP[act_quant_format][act_scale_type][act_param_method][act_quant_granularity][act_quant_type]
        sym_act_quant = INPUT_QUANT_MAP[act_quant_format][act_scale_type][act_param_method][act_quant_granularity]['sym']
        per_tensor_act_quant = INPUT_QUANT_MAP[act_quant_format][act_scale_type][act_param_method]['per_tensor'][act_quant_type]
        act_quant = act_quant.let(**act_bit_width_dict)
        sym_act_quant = sym_act_quant.let(**act_bit_width_dict)
        per_tensor_act_quant = per_tensor_act_quant.let(**act_bit_width_dict)

        weight_quant = weight_quant.let(
            **{
                'narrow_range' : weight_narrow_range,
                'scaling_impl': ParameterFromStatsFromParameterScaling
            }
        )

        # TODO: if weight quantization is symmetric
        act_quant = act_quant.let(
            **{
                'high_percentile_q': act_quant_percentile, 'dtype' : torch.float32
            }
        )

        sym_act_quant = sym_act_quant.let(
            **{
                'high_percentile_q': act_quant_percentile, 'dtype': torch.float32
            }
        )

        per_tensor_act_quant = per_tensor_act_quant.let(
            **{
                'high_percentile_q': act_quant_percentile, 'dtype': torch.float32
            }
        )

        weight_quant_dict = {'weight_quant': weight_quant}

        quant_wbiol_kwargs = {
            **weight_quant_dict,
            'dtype': torch.float32,
            'return_quant_tensor': False,
            'bias_quant': bias_quant
        }

        quant_mha_kwargs = {
            **kwargs_prefix('in_proj_', weight_quant_dict),
            **kwargs_prefix('out_proj_', weight_quant_dict),
            'in_proj_input_quant': None,
            'in_proj_bias_quant': bias_quant,
            'softmax_input_quant': None,
            'attn_output_weights_quant': sym_act_quant,
            'q_scaled_quant': sym_act_quant,
            'k_transposed_quant': sym_act_quant,
            'v_quant': sym_act_quant,
            'out_proj_input_quant': act_quant,
            'out_proj_bias_quant': bias_quant,
            'out_proj_output_quant': None,
            # activation equalization requires packed_in_proj
            # since it supports only self-attention
            'packed_in_proj': True,
            'dtype': torch.float32,
            'return_quant_tensor': False}
        quant_act_kwargs = {'act_quant': act_quant, 'return_quant_tensor' : True}
        unsigned_quant_act_kwargs = quant_act_kwargs.copy()
        if uint_sym_act_for_unsigned_values:
            quant_mha_kwargs['attn_output_weights_signed'] = False
            unsigned_quant_act_kwargs['signed'] = False

        quant_layer_map = {
        torch.nn.Linear: (qnn.QuantLinear, quant_wbiol_kwargs),
        torch.nn.MultiheadAttention: (qnn.QuantMultiheadAttention, quant_mha_kwargs),
        torch.nn.Conv1d: (qnn.QuantConv1d, quant_wbiol_kwargs),
        torch.nn.Conv2d: (qnn.QuantConv2d, quant_wbiol_kwargs),
        torch.nn.ConvTranspose1d: (qnn.QuantConvTranspose1d, quant_wbiol_kwargs),
        torch.nn.ConvTranspose2d: (qnn.QuantConvTranspose2d, quant_wbiol_kwargs),}

        quant_act_map = {
            torch.nn.ReLU: (qnn.QuantReLU, {
                **unsigned_quant_act_kwargs}),
            torch.nn.ReLU6: (qnn.QuantReLU, {
                **unsigned_quant_act_kwargs}),
            torch.nn.Sigmoid: (qnn.QuantSigmoid, {
                **unsigned_quant_act_kwargs}),}
        quant_identity_map = {
            'signed': (qnn.QuantIdentity, {
                **quant_act_kwargs}),
            'unsigned': (qnn.QuantIdentity, {
                **unsigned_quant_act_kwargs}),}

        return quant_layer_map, quant_act_map, quant_identity_map
    
    def quantize_input(self,
                        model):
        input_quantizer = self.quantize_kwargs['quant_identity_map'].get('signed', None)
        ignore_missing_keys_state = config.IGNORE_MISSING_KEYS
        config.IGNORE_MISSING_KEYS = True
        training_state = model.training
        model.eval()

        rewriters = []
        if input_quantizer is None:
                return model
        for node in model.graph.nodes:
                if node.op == 'placeholder':
                    act_quant, kwargs_act_quant = input_quantizer
                    inp_quant = act_quant(**kwargs_act_quant)
                    name = node.name + '_quant'
                    model.add_module(name, inp_quant)
                    rewriters.append(InsertModuleCallAfter(name, node))
        for rewriter in rewriters:
                model = rewriter.apply(model)

        model = DisableLastReturnQuantTensor().apply(model)
        model.train(training_state)
        config.IGNORE_MISSING_KEYS = ignore_missing_keys_state
        return model

    def quantize(self,
                    model,
                    quant_identity_map = None,
                    compute_layer_map = None,
                    quant_act_map = None,
                    unsigned_act_tuple = UNSIGNED_ACT_TUPLE,
                    requantize_layer_handler_output = True
        ):

            ignore_missing_keys_state = config.IGNORE_MISSING_KEYS
            config.IGNORE_MISSING_KEYS = True,
            training_state = model.training
            model.eval()

'''  
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

'''