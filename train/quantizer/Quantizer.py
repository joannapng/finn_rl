from pickle import TRUE
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
from brevitas.graph.utils import del_module
from brevitas.graph.quantize import align_input_quant
from brevitas.graph.quantize_impl import inp_placeholder_handler, add_output_quant_handler, residual_handler, output_quant_handler
from brevitas.graph.quantize_impl import are_inputs_quantized_and_aligned
from brevitas.graph.base import InsertModuleCallAfter
import torch
from brevitas.graph.base import ModuleToModuleByInstance
from brevitas.graph.base import ModuleInstanceToModuleInstance
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
from brevitas.quant.scaled_int import Int8Bias
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFixedPoint
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloatMSE
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloatMSE
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloatMSE

from brevitas.graph.standardize import DisableLastReturnQuantTensor
from brevitas.graph.quantize_impl import SIGN_PRESERVING_MODULES
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType

from brevitas_examples.bnn_pynq.models.common import CommonActQuant

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
            bias_bit_width = bias_bit_width,
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

        self.layer_rewriters = []

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

        #bias_quant = BIAS_BIT_WIDTH_MAP[bias_bit_width] if act_bit_width is not None else None
        bias_quant = Int8Bias # if I set it to 8 bias the checks fail???? 
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
                'high_percentile_q': act_quant_percentile, 'dtype' : torch.float32,
                'scaling_imp': ParameterFromStatsFromParameterScaling
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
        
        ignore_missing_keys_state = config.IGNORE_MISSING_KEYS
        config.IGNORE_MISSING_KEYS = True
        training_state = model.training
        model.eval()

        input_quantizer = (qnn.QuantIdentity, {'act_quant' : CommonActQuant,
                                               'bit_width' : 8,
                                               'min_val' : -1.0,
                                               'max_val' : 1.0 - 2.0 ** (-7),
                                               'narrow_range' : False,
                                               'scaling_impl_type' : ScalingImplType.CONST})
        
        model = inp_placeholder_handler(model, input_quantizer)

        #model = DisableLastReturnQuantTensor().apply(model)
        model.train(training_state)
        config.IGNORE_MISSING_KEYS = ignore_missing_keys_state
        return model

    def quantize_act(self,
                     model,
                     act_idx, 
                     act_bit_width):
        ignore_missing_keys_state = config.IGNORE_MISSING_KEYS
        config.IGNORE_MISSING_KEYS = True
        training_state = model.training
        model.eval()

        layer_map = self.quantize_kwargs['quant_act_map']

        for i, node in enumerate(model.graph.nodes):
            if node.op == 'call_module' and i == act_idx:
                module = get_module(model, node.target)
                if isinstance(module, tuple(layer_map.keys())):
                    quant_module_class, quant_module_kwargs = layer_map[type(module)]
                    quant_module_kwargs['bit_width'] = act_bit_width
                    quant_module = quant_module_class(**quant_module_kwargs)
                
                    if len(node.users) == 1:
                        user_node = list(node.users.keys())[0]
                        if user_node.name.endswith('act_eq_mul'):
                            act_module = quant_module.act_quant.fused_activation_quant_proxy.activation_impl
                            mul_module = get_module(model, user_node.target)
                            quant_module.act_quant.fused_activation_quant_proxy.activation_impl = torch.nn.Sequential(
                                 *[act_module, mul_module]
                            )
                            user_node.replace_all_uses_with(node)
                            model.graph.erase_node(user_node)
                            del_module(model, user_node.target)
                    
                    rewriter = ModuleInstanceToModuleInstance(
                        module, quant_module
                    )
                    break
            
        model = rewriter.apply(model)

        #model = DisableLastReturnQuantTensor().apply(model)
        model.train(training_state)
        config.IGNORE_MISSING_KEYS = ignore_missing_keys_state

        return model
         
    def quantize_output(self,
                        model):
        ignore_missing_keys_state = config.IGNORE_MISSING_KEYS
        config.IGNORE_MISSING_KEYS = True
        training_state = model.training
        model.eval()

        quant_identity_map = self.quantize_kwargs['quant_identity_map']
        quant_act_map = self.quantize_kwargs['quant_act_map']
        unsigned_act_tuple = UNSIGNED_ACT_TUPLE

        model = add_output_quant_handler(
            model, quant_identity_map, quant_act_map, unsigned_act_tuple
        )
        
        #model = DisableLastReturnQuantTensor().apply(model)
        model.train(training_state)
        config.IGNORE_MISSING_KEYS = ignore_missing_keys_state
        
        return model

    def handle_residuals(self,
                        model):
        ignore_missing_keys_state = config.IGNORE_MISSING_KEYS
        config.IGNORE_MISSING_KEYS = True
        training_state = model.training
        model.eval()

        quant_identity_map = self.quantize_kwargs['quant_identity_map']
        quant_act_map = self.quantize_kwargs['quant_act_map']
        unsigned_act_tuple = UNSIGNED_ACT_TUPLE

        model = residual_handler(
            model, quant_identity_map, quant_act_map, unsigned_act_tuple, align_input_quant
        )
        
        #model = DisableLastReturnQuantTensor().apply(model)
        model.train(training_state)
        config.IGNORE_MISSING_KEYS = ignore_missing_keys_state

        return model
        
    def quantize_layer(self,
                       model,
                       layer_idx,
                       is_final_layer,
                       weight_bit_width):
        
        ignore_missing_keys_state = config.IGNORE_MISSING_KEYS
        config.IGNORE_MISSING_KEYS = True
        training_state = model.training
        model.eval()

        layer_map = self.quantize_kwargs['compute_layer_map']
        quant_identity_map = self.quantize_kwargs['quant_identity_map']
        quant_act_map = self.quantize_kwargs['quant_act_map']
        unsigned_act_tuple = UNSIGNED_ACT_TUPLE
        
        # TODO: check the requantize output
        for i, node in enumerate(model.graph.nodes):
            rewriters = []
            if node.op == 'call_module' and i == layer_idx:
                module = get_module(model, node.target)
                if isinstance(module, tuple(layer_map.keys())):
                    if len(node.users) > 1 and all(['getitem' in n.name for n in node.users]):
                        for n in node.users:
                            if len(n.users) > 0:
                                output_quant_handler(
                                    model,
                                    n,
                                    rewriters,
                                    is_sign_preserving=isinstance(module, SIGN_PRESERVING_MODULES),
                                    quant_identity_map=quant_identity_map,
                                    quant_act_map=quant_act_map,
                                    unsigned_act_tuple=unsigned_act_tuple)
                    else:
                        output_quant_handler(
                        model,
                        node,
                        rewriters,
                        is_sign_preserving=isinstance(module, SIGN_PRESERVING_MODULES),
                        quant_identity_map=quant_identity_map,
                        quant_act_map=quant_act_map,
                        unsigned_act_tuple=unsigned_act_tuple)
                    if layer_map[type(module)] is not None:
                        quant_module_class, quant_module_kwargs = layer_map[type(module)]
                        quant_module_kwargs['weight_bit_width'] = weight_bit_width

                        if not are_inputs_quantized_and_aligned(
                            model, node, [], quant_act_map, same_sign = False
                        ) and not 'input_quant' in quant_module_kwargs and len(quant_identity_map):
                            previous_node = node.all_input_nodes[0]
                            previous_node_users = list(previous_node.users.keys())
                            previous_node_users.remove(node)

                            act_quant, kwargs_act_quant = quant_identity_map['signed']
                            inp_quant = act_quant(**kwargs_act_quant)
                            name = node.name + '_input_quant'
                            model.add_module(name, inp_quant)
                            rewriter = InsertModuleCallAfter(
                                name, previous_node, tuple(previous_node_users)
                            )
                            rewriters.append(rewriter)
                        break
        rewriter = ModuleToModuleByInstance(
            module, quant_module_class, **quant_module_kwargs
        )
        rewriters.append(rewriter)

        for rewriter in rewriters:
            model = rewriter.apply(model)

        #model = DisableLastReturnQuantTensor().apply(model)
        model.train(training_state)
        config.IGNORE_MISSING_KEYS = ignore_missing_keys_state

        return model
    
    def finalize(self,
                 model):
        ignore_missing_keys_state = config.IGNORE_MISSING_KEYS
        config.IGNORE_MISSING_KEYS = True
        training_state = model.training
        model.eval()

        model = DisableLastReturnQuantTensor().apply(model)

        model.train(training_state)
        config.IGNORE_MISSING_KEYS = ignore_missing_keys_state
        return model
