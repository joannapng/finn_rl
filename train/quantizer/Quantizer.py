import torch
import torch.nn as nn

from copy import deepcopy

import brevitas.nn as qnn
from brevitas import config
from brevitas.core.scaling import ParameterScaling
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.graph.utils import get_module
from brevitas.graph.utils import del_module
from brevitas.graph.quantize_impl import add_output_quant_handler, residual_handler, output_quant_handler
from brevitas.graph.quantize_impl import are_inputs_quantized_and_aligned

from brevitas.graph.base import InsertModuleCallAfter
from brevitas.graph.base import ModuleToModuleByInstance
from brevitas.graph.base import ModuleInstanceToModuleInstance

from brevitas.quant.scaled_int import Int8BiasPerTensorFloatInternalScaling
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant.scaled_int import Int8ActPerTensorFloat

from brevitas.graph.standardize import DisableLastReturnQuantTensor
from brevitas.graph.quantize_impl import SIGN_PRESERVING_MODULES

from train.quantizer.utils import align_input_quant
from brevitas.core.scaling import ScalingImplType
from brevitas.inject.enum import *

UNSIGNED_ACT_TUPLE = (nn.ReLU, nn.ReLU6, nn.Sigmoid, nn.Hardsigmoid)
  
class Quantizer(object):
    def __init__(
            self,
            weight_bit_width,
            act_bit_width,
            residual_bit_width
    ):
        weight_bit_width_dict = {}
        act_bit_width_dict = {}
        weight_bit_width_dict['weight_bit_width'] = weight_bit_width
        act_bit_width_dict['act_bit_width'] = residual_bit_width # keep the residual bit width for activations

        quant_layer_map, quant_act_map, quant_identity_map = self.create_quant_maps(
            bias_bit_width = 8,
            weight_bit_width = weight_bit_width,
            act_bit_width = act_bit_width
        )

        self.quantize_kwargs = {
            'compute_layer_map' : quant_layer_map,
            'quant_act_map' : quant_act_map,
            'quant_identity_map' : quant_identity_map
        }

        self.quantizable_acts = [
            nn.ReLU,
            nn.ReLU6,
            nn.Sigmoid,
            qnn.QuantReLU,
            qnn.QuantSigmoid
        ]

        self.quantizable_layers = [
            nn.Linear,
            nn.MultiheadAttention,
            nn.Conv1d,
            nn.Conv2d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            qnn.QuantLinear,
            qnn.QuantMultiheadAttention,
            qnn.QuantConv1d,
            qnn.QuantConv2d,
            qnn.QuantConvTranspose1d,
            qnn.QuantConvTranspose2d
        ]

    def create_quant_maps(
            self,
            bias_bit_width,
            weight_bit_width,
            act_bit_width
    ):
        
        def kwargs_prefix(prefix, weight_kwargs):
            return {prefix + k: v for k, v in weight_kwargs.items()}
        
        weight_bit_width_dict = {'bit_width' : weight_bit_width}
        act_bit_width_dict = {'bit_width': act_bit_width}

        weight_quant = Int8WeightPerTensorFloat
        weight_quant = weight_quant.let(**weight_bit_width_dict)

        act_quant = Int8ActPerTensorFloat
        sym_act_quant =Int8ActPerTensorFloat
        per_tensor_act_quant = Int8ActPerTensorFloat

        act_quant = act_quant.let(**act_bit_width_dict)
        sym_act_quant = sym_act_quant.let(**act_bit_width_dict)
        per_tensor_act_quant = per_tensor_act_quant.let(**act_bit_width_dict)

        weight_quant = weight_quant.let(
            **{
                'high_percentile_q': 99.999, 'dtype' : torch.float32,
            }
        )

        act_quant = act_quant.let(
            **{
                'high_percentile_q': 99.999, 'dtype' : torch.float32,
            }
        )

        sym_act_quant = sym_act_quant.let(
            **{
                'high_percentile_q': 99.999, 'dtype': torch.float32
            }
        )

        per_tensor_act_quant = per_tensor_act_quant.let(
            **{
                'high_percentile_q': 99.999, 'dtype': torch.float32
            }
        )

        weight_quant_dict = {'weight_quant': weight_quant}

        quant_wbiol_kwargs = {
            **weight_quant_dict,
            'dtype': torch.float32,
            'return_quant_tensor': True,
            'bias_quant' : None,
            'weight_signed' : True,
            'weight_narrow_range' : True
        }

        quant_mha_kwargs = {
            **kwargs_prefix('in_proj_', weight_quant_dict),
            **kwargs_prefix('out_proj_', weight_quant_dict),
            'in_proj_input_quant': None,
            'in_proj_bias_quant': None,
            'softmax_input_quant': None,
            'attn_output_weights_quant': sym_act_quant,
            'q_scaled_quant': sym_act_quant,
            'k_transposed_quant': sym_act_quant,
            'v_quant': sym_act_quant,
            'out_proj_input_quant': act_quant,
            'out_proj_bias_quant': None,
            'out_proj_output_quant': None,
            # activation equalization requires packed_in_proj
            # since it supports only self-attention
            'packed_in_proj': True,
            'dtype': torch.float32,
            'return_quant_tensor': True}
        
        quant_act_kwargs = {'act_quant': act_quant, 'return_quant_tensor' : True}
        unsigned_quant_act_kwargs = quant_act_kwargs.copy()

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
    
    def quantize_model(self, model, strategy, quantizable_idx, num_quant_acts):
        ignore_missing_keys_state = config.IGNORE_MISSING_KEYS
        config.IGNORE_MISSING_KEYS = True
        training_state = model.training
        model.eval()

        # quantize input
        model = self.quantize_input(model)
        quantizable_idx = self.update_index(model, quantizable_idx)
        
        # quantize activations
        for i in range(num_quant_acts):
            model = self.quantize_act(model, quantizable_idx[i], int(strategy[i]))
            quantizable_idx = self.update_index(model, quantizable_idx)

        # quantize add outputs and handle residuals
        model = self.quantize_output(model)
        model = self.handle_residuals(model)
        quantizable_idx = self.update_index(model, quantizable_idx)

        # quantize compute layers
        for i in range(num_quant_acts, len(quantizable_idx)):
            model = self.quantize_layer(model, quantizable_idx[i], int(strategy[i]))
            quantizable_idx = self.update_index(model, quantizable_idx)

        model = DisableLastReturnQuantTensor().apply(model)
        model.train(training_state)
        config.IGNORE_MISSING_KEYS = ignore_missing_keys_state

        return model

    def update_index(self, model, quantizable_idx):
        idx = 0

        # update activation indices
        for i, node in enumerate(model.graph.nodes):
            if node.op == 'call_module':
                module = get_module(model, node.target)
                if type(module) in self.quantizable_acts:
                    quantizable_idx[idx] = i
                    idx += 1

        # update compute indices
        for i, node in enumerate(model.graph.nodes):
            if node.op == 'call_module':
                module = get_module(model, node.target)
                if type(module) in self.quantizable_layers:
                    quantizable_idx[idx] = i
                    idx += 1

        return quantizable_idx

    def quantize_input(self,
                        model):

        # Input quantizer fixed at 8 bits
        rewriters = []
        graph = model.graph
        for node in graph.nodes:
            if node.name == "sub": # after -1.0
                input_quantizer = ( qnn.QuantIdentity, {'act_quant' : Int8ActPerTensorFloat,
                                    'bit_width' : 8,
                                    'return_quant_tensor' : True})
                act_quant, kwargs_act_quant = input_quantizer
                inp_quant = act_quant(**kwargs_act_quant)
                name = node.name + '_quant'
                model.add_module(name, inp_quant)
                rewriters.append(InsertModuleCallAfter(name, node))

        for rewriter in rewriters:
            model = rewriter.apply(model)
        return model

    def quantize_act(self,
                     model,
                     act_idx, 
                     act_bit_width):
        
        layer_map = self.quantize_kwargs['quant_act_map']

        for i, node in enumerate(model.graph.nodes):
            if node.op == 'call_module' and i == act_idx:
                module = get_module(model, node.target)
                if isinstance(module, tuple(layer_map.keys())):
                    
                    quant_module_class, quant_module_kwargs = deepcopy(layer_map[type(module)])
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
        return model
         
    def quantize_output(self,
                        model):

        quant_identity_map = self.quantize_kwargs['quant_identity_map']
        quant_act_map = self.quantize_kwargs['quant_act_map']
        unsigned_act_tuple = UNSIGNED_ACT_TUPLE
        
        model = add_output_quant_handler(
            model, quant_identity_map, quant_act_map, unsigned_act_tuple
        )
        return model

    def handle_residuals(self,
                        model):

        quant_identity_map = self.quantize_kwargs['quant_identity_map']
        quant_act_map = self.quantize_kwargs['quant_act_map']
        unsigned_act_tuple = UNSIGNED_ACT_TUPLE

        model = residual_handler(
            model, quant_identity_map, quant_act_map, unsigned_act_tuple, align_input_quant
        )

        return model
        
    def quantize_layer(self,
                       model,
                       layer_idx,
                       weight_bit_width):

        layer_map = deepcopy(self.quantize_kwargs['compute_layer_map'])
        quant_identity_map = self.quantize_kwargs['quant_identity_map']
        quant_act_map = self.quantize_kwargs['quant_act_map']
        unsigned_act_tuple = UNSIGNED_ACT_TUPLE
        rewriters = []
        
        for i, node in enumerate(model.graph.nodes):
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
                        output_quant_identity_map = deepcopy(quant_identity_map)
                        
                        is_output = False
                        for n in node.users:
                            if n.op == 'output':
                                is_output = True
                                break
                        
                        if is_output:
                            # keep output quantization to 8 bits
                            output_quant_identity_map['signed'][1]['bit_width'] = 8
                            output_quant_identity_map['unsigned'][1]['bit_width'] = 8
                            output_quant_handler(
                            model,
                            node,
                            rewriters,
                            is_sign_preserving=isinstance(module, SIGN_PRESERVING_MODULES),
                            quant_identity_map=output_quant_identity_map,
                            quant_act_map=quant_act_map,
                            unsigned_act_tuple=unsigned_act_tuple)     
                        else:
                            output_quant_handler(
                            model,
                            node,
                            rewriters,
                            is_sign_preserving=isinstance(module, SIGN_PRESERVING_MODULES),
                            quant_identity_map=output_quant_identity_map,
                            quant_act_map=quant_act_map,
                            unsigned_act_tuple=unsigned_act_tuple)

                    if layer_map[type(module)] is not None:
                        quant_module_class, quant_module_kwargs = deepcopy(layer_map[type(module)])
                        quant_module_kwargs['weight_bit_width'] = weight_bit_width 

                        if weight_bit_width == 1:
                            # to avoid inf scale
                            quant_module_kwargs['scaling_impl'] = ParameterScaling(scaling_init=0.1)
                            quant_module_kwargs['weight_narrow_range'] = False
                        else:
                            quant_module_kwargs['scaling_impl'] = ScalingImplType.STATS
                            
                        if module.bias is not None:
                            # add bias quant if the module has bias
                            quant_module_kwargs['bias_quant'] = Int8BiasPerTensorFloatInternalScaling
                        else:
                            quant_module_kwargs['bias_quant'] = None

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

        return model