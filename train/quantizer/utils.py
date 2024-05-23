import brevitas.nn as qnn
from brevitas.core.scaling.standalone import ConstScaling
from brevitas.core.scaling.standalone import ParameterScaling

def align_input_quant(
        module, shared_quant_identity, shared_quant_identity_name, quant_identity_map, align_sign):
    """
    Based on the input module, the function decides how to align its output.
    """
    # If it is a QuantIdentity already, simply modify tensor_quant or the scaling implementations
    # based on whether we need to align the sign or not
    if isinstance(module, qnn.QuantIdentity):
        if align_sign or module.is_quant_act_signed == shared_quant_identity.is_quant_act_signed:
            return shared_quant_identity
        else:
            assert not module.is_quant_act_signed and shared_quant_identity.is_quant_act_signed
            quant_module_class, quant_module_kwargs = quant_identity_map['unsigned']
            return (
                quant_module_class,
                {
                    **quant_module_kwargs,
                    'bit_width_impl': 
                        shared_quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant
                        .msb_clamp_bit_width_impl,
                    'scaling_impl':
                        shared_quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant
                        .scaling_impl,
                    'int_scaling_impl':
                        shared_quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant
                        .int_scaling_impl})
    elif hasattr(module, 'output_quant'):
        return (type(module), {'output_quant': shared_quant_identity})
    # If it is a QuantAct where the scaling can be determined through stats (thus through calibration),
    # then adapt its act_quant according to align_sign.
    elif hasattr(module, 'act_quant') and not isinstance(
            module.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl,
        (ParameterScaling, ConstScaling)):
        module_type = type(module)
        if align_sign:
            partial_config = {
                'signed':
                    shared_quant_identity.act_quant.is_signed,
                'tensor_quant':
                    shared_quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant}
        else:
            partial_config = {
                'bit_width_impl': 
                    shared_quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant
                    .msb_clamp_bit_width_impl,
                'scaling_impl':
                    shared_quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant
                    .scaling_impl,
                'int_scaling_impl':
                    shared_quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant
                    .int_scaling_impl}
        injector = module.act_quant.quant_injector.let(**partial_config)
        return module_type(act_quant=injector, return_quant_tensor=True)
    # In all other cases, return the name of the QuantIdentity that will be added at the output of
    # the module
    else:
        return shared_quant_identity_name
    
