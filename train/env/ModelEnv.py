import copy
import math
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from enum import IntEnum

from brevitas.graph.utils import get_module
from brevitas.graph.quantize import preprocess_for_quantize
import brevitas.nn as qnn

from ..quantizer import Quantizer
from ..finetune import Finetuner
from ..utils import measure_model

from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env

class LayerTypes(IntEnum):
    LINEAR = 0
    MHA = 1
    CONV1D = 2
    CONV2D = 3
    CONVTRANSPOSE1D = 4
    CONVTRANSPOSE2D = 5

class ActTypes(IntEnum):
    RELU = 0
    RELU6 = 1
    SIGMOID = 2

class ModelEnv(gym.Env):
    def __init__(self, args, weights, model_config):
        self.args = args

        self.observation_space = spaces.Box(low = 0.0, high = 1.0, shape=(5, ), dtype = np.float32)
        self.action_space = spaces.Box(low = -1.0, high = 1.0, shape = (1, ), dtype = np.float32)
        self.num_objectives = 2
        self.utility_weights = weights

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

        self.finetuner = Finetuner(args, model_config)
        self.model = copy.deepcopy(self.finetuner.model)
        self.orig_model = copy.deepcopy(self.model)

        self.cur_ind = 0
        self.strategy = [] # quantization strategy
        
        self.min_bit = args.min_bit
        self.max_bit = args.max_bit   
        self.best_reward = -math.inf
        self.model_config = model_config
        
        self.build_index() # build the states for each layer
        self.index_to_quantize = self.quantizable_idx[self.cur_ind]

        self.quantizer = Quantizer(
            self.model,
            args.weight_bit_width,
            args.act_bit_width,
            args.bias_bit_width,
            args.weight_quant_granularity,
            args.act_quant_percentile,
            args.act_quant_type,
            args.scale_factor_type,
            args.quant_format,
            args.act_quant_calibration_type,
            args.weight_quant_calibration_type,
            args.weight_quant_type
        )
    
        self.orig_acc = self.finetuner.orig_acc
        self.prev_acc = self.orig_acc
        self.action_running_mean = 0

    def build_index(self, rebuild = False):
        '''
        Store the indices and the types of the layers that are quantizable
        Construct the static part of the state
        '''

        # if model is not already preprocessed for quantization
        if not rebuild:
            self.model = preprocess_for_quantize(self.model)
            measure_model(self.model, self.model_config['center_crop_shape'], 
                      self.model_config['center_crop_shape'], self.finetuner.in_channels) # measure feature maps for each model
        
        self.quantizable_idx = []
        self.layer_types = []
        self.num_quant_acts = 0
        layer_embedding = []

        # Activations first
        for i, node in enumerate(self.model.graph.nodes):
            this_state = []
            if node.op == 'call_module':
                module = get_module(self.model, node.target)
                if type(module) in self.quantizable_acts:
                    self.quantizable_idx.append(i)
                    self.layer_types.append(type(module))
                    this_state.append([i])
                    this_state.append([1])
                    
                    if type(module) == nn.ReLU or type(module) == qnn.QuantReLU:
                        this_state.append([ActTypes.RELU])
                    elif type(module) == nn.ReLU6 or type(module) == qnn.QuantReLU:
                        this_state.append([ActTypes.RELU6])
                    elif type(module) == nn.Sigmoid or type(module) == qnn.QuantSigmoid:
                        this_state.append([ActTypes.SIGMOID])

                    this_state.append([module.flops])
                    this_state.append([module.params])
                    layer_embedding.append(np.hstack(this_state))
        # number of activation layers
        self.num_quant_acts = len(self.quantizable_idx)

        # Compute layers
        for i, node in enumerate(self.model.graph.nodes):
            this_state = []
            if node.op == 'call_module':
                module = get_module(self.model, node.target)
                if type(module) in self.quantizable_layers:
                    self.quantizable_idx.append(i)
                    self.layer_types.append(type(module))
                    this_state.append([i])
                    this_state.append([0])

                    if type(module) == nn.Linear or type(module) == qnn.QuantLinear:
                        this_state.append([LayerTypes.LINEAR])
                    elif type(module) == nn.MultiheadAttention or type(module) == qnn.QuantMultiheadAttention:
                        this_state.append([LayerTypes.MHA])
                    elif type(module) == nn.Conv1d or type(module) == qnn.QuantConv1d:
                        this_state.append([LayerTypes.CONV1D])
                    elif type(module) == nn.Conv2d or type(module) == qnn.QuantConv2d:
                        this_state.append([LayerTypes.CONV2D])
                    elif type(module) == nn.ConvTranpose1d or type(module) == qnn.QuantConvTranspose1d:
                        this_state.append([LayerTypes.CONVTRANSPOSE1D])
                    elif type(module) == nn.ConvTranspose2d or type(module) == qnn.QuantConvTranspose2d:
                        this_state.append([LayerTypes.CONVTRANSPOSE2D])

                    weights = module.weight
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weights)
                    this_state.append([module.flops])
                    this_state.append([module.params])
                    layer_embedding.append(np.hstack(this_state))

        layer_embedding = np.array(layer_embedding, dtype=np.float32)
        # normalize to (0, 1)
        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])

            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)
        
        self.layer_embedding = layer_embedding

    def reset(self, seed = None, option = None):
        super().reset(seed = seed)

        self.model = copy.deepcopy(self.orig_model).to(self.finetuner.device)
        self.model = preprocess_for_quantize(self.model)
        self.model = self.quantizer.quantize_input(self.model)
        self.build_index(rebuild=False)
        self.model.to(self.finetuner.device)

        self.finetuner.model = self.model
        self.finetuner.model.to(self.finetuner.device)
        self.finetuner.init_finetuning_optim()
        self.finetuner.init_loss()
        
        self.cur_ind = 0
        self.index_to_quantize = self.quantizable_idx[self.cur_ind]
        self.action_running_mean = 0
        self.strategy = []
        self.num_actions = 0
        self.prev_acc = self.max_bit # initialize prev action to maximum allowed bit
        obs = self.layer_embedding[0].copy()
        info = {"info": 0}
        return obs, info
    
    def step(self, action):
        action = self.get_action(action)
        self.strategy.append(action)
        self.num_actions += 1

        # if not all activations have been quantized
        if self.cur_ind < self.num_quant_acts:
            self.model = self.quantizer.quantize_act(
                self.model,
                self.index_to_quantize,
                int(action[0])
            )
        
        # if activations have been quantized, quantize outputs and handle residuals
        if self.cur_ind == self.num_quant_acts:
            self.model = self.quantizer.quantize_output(self.model)
            self.build_index(rebuild = True)
            
            self.model = self.quantizer.handle_residuals(self.model)
            self.build_index(rebuild = True)        
            
        if self.cur_ind >= self.num_quant_acts:
            self.model = self.quantizer.quantize_layer(self.model, 
                                                       self.index_to_quantize, 
                                                       int(action[0]))
            # build index again, because quanize layers can insert quantizers
            self.build_index(rebuild = True)

        if self.is_final_layer():
            print(self.strategy)
            self.quantizer.finalize(self.model)

        self.finetuner.model = self.model
        self.finetuner.model.to(self.finetuner.device)

        # accuracy before finetuning
        acc = self.finetuner.validate(eval = False)
        
        # can only calibrate if residuals are handled
        if self.cur_ind >= self.num_quant_acts:
            self.finetuner.calibrate()

        if self.num_actions % self.args.finetune_every == 0 or self.is_final_layer():
            self.finetuner.init_finetuning_optim()
            self.finetuner.init_loss()

            # if it is final layer train for 10 epochs
            if self.is_final_layer():
                self.finetuner.finetuning_epochs = 10
            
            self.finetuner.finetune()
            self.num_actions = 0

        self.action_running_mean = ((action[0]) / (self.max_bit) + (self.cur_ind) * self.action_running_mean) / (self.cur_ind + 1)
        reward = self.reward(acc, action[0])

        if self.is_final_layer():
            obs = self.layer_embedding[self.cur_ind, :].copy()
            terminated = True
            acc = self.finetuner.validate(eval = False)
            info = {"accuracy": acc}
            print(f'Accuracy: {acc}, Size: {self.action_running_mean}')
            return obs, reward, terminated, False, info
        
        terminated = False
        self.cur_ind += 1 
        self.index_to_quantize = self.quantizable_idx[self.cur_ind]
        obs = self.layer_embedding[self.cur_ind, :].copy()
        self.prev_acc = acc
        acc = self.finetuner.validate(eval = False)
        info = {"accuracy": acc}
        return obs, reward, terminated, False, info
        
    def reward(self, acc, action):
        # in the case we have an increase in performance prev_acc - acc < 0 so r1 > 1
        r1 = (1 - (self.prev_acc - acc) / self.prev_acc) * 100
        r2 = - action / self.max_bit * 100

        return (np.array([r1, r2]) * self.utility_weights).sum()

    def get_action(self, action):
        action = action.astype(np.float32)
        lbound, rbound = self.min_bit, self.max_bit
        action = (action + 1) * (rbound - lbound) / 2.0 + self.min_bit - 0.5
        action = np.ceil(action).astype(int)
        return action

    def is_final_layer(self):
        return self.cur_ind == len(self.quantizable_idx) - 1