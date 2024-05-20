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

        self.observation_space = spaces.Box(low = 0.0, high = 1.0, shape=(6, ), dtype = np.float32)
        self.action_space = spaces.Box(low = -1.0, high = 1.0, shape = (1, ), dtype = np.float32)

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
        self.model_config = model_config
        self.orig_model = copy.deepcopy(self.model)

        self.cur_ind = 0
        self.strategy = [] # quantization strategy
        
        self.min_bit = args.min_bit
        self.max_bit = args.max_bit
        self.last_action = self.max_bit
        
        # init reward
        self.best_reward = -math.inf
        
        self.build_state_embedding() # build the states for each layer
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
        
        # get platform simulator
        self.simulator = None

    def build_state_embedding(self):
        self.model = preprocess_for_quantize(self.model)

        measure_model(self.model, self.model_config['center_crop_shape'], 
                    self.model_config['center_crop_shape'], self.finetuner.in_channels)
    
        self.quantizable_idx = []
        self.num_quant_acts = 0
        layer_embedding = []

        # Activations first
        for i, node in enumerate(self.model.graph.nodes):
            this_state = []
            if node.op == 'call_module':
                module = get_module(self.model, node.target)
                if type(module) in self.quantizable_acts:
                    self.quantizable_idx.append(i)
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
                    this_state.append([1.]) # last action
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

                    this_state.append([module.flops])
                    this_state.append([module.params])
                    this_state.append([1.]) # last action
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
        self.build_state_embedding()

        self.model.to(self.finetuner.device)

        self.finetuner.model = self.model
        self.finetuner.model.to(self.finetuner.device)
        self.finetuner.init_finetuning_optim()
        self.finetuner.init_loss()
        
        self.cur_ind = 0
        self.index_to_quantize = self.quantizable_idx[self.cur_ind]
        self.strategy = []

        obs = self.layer_embedding[0].copy()
        info = {"info": 0}
        return obs, info

    '''    
    def step(self, action):

        action = self.get_action(action)
        self.last_action = action
        self.strategy.append(action)

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
            self.model = self.quantizer.handle_residuals(self.model)

            # build index again, because quantize output and handle residuals can insert quantizers
            self.update_index()
        
        if self.cur_ind >= self.num_quant_acts:
            self.model = self.quantizer.quantize_layer(self.model, 
                                                       self.index_to_quantize, 
                                                       int(action[0]))
            # build index again, because quanize layers can insert quantizers
            self.update_index()

        self.finetuner.model = self.model
        self.finetuner.model.to(self.finetuner.device)

        self.finetuner.calibrate()

        # finetune only after all layers have been quantized
        if self.is_final_layer():
            self.quantizer.finalize(self.model)
            self.finetuner.validate()
            self.finetuner.init_finetuning_optim()
            self.finetuner.init_loss()
            self.finetuner.finetune()
        
        acc = self.finetuner.validate(eval = False)
        reward = self.reward(acc, action[0])

        if self.is_final_layer():
            obs = self.layer_embedding[self.cur_ind, :].copy()
            terminated = True
            info = {"accuracy": acc}
            print(f'Accuracy: {acc}')
            return obs, reward, terminated, False, info
        
        terminated = False
        self.cur_ind += 1 
        self.index_to_quantize = self.quantizable_idx[self.cur_ind]
        obs = self.layer_embedding[self.cur_ind, :].copy()
        if acc < 25.0:
            acc = self.finetuner.validate(eval = False)

        self.prev_acc = acc
        info = {"accuracy": acc}
        return obs, reward, terminated, False, info
    '''

    def step(self, action):
        action = self.get_action(action)
        self.last_action = action
        self.strategy.append([self.last_action])

        if self.is_final_layer():
            # check if model is feasible
            # TODO: return how much the model exceeds resources (maybe)
            self.final_action_wall()

            # quantize model
            self.quantizer.quantize_model(self.strategy)

            # calibrate model 
            self.finetuner.model = self.model 
            self.finetuner.model.to(self.finetuner.device)
            self.finetuner.calibrate()

            # finetuner model
            self.finetuner.init_finetuning_optim()
            self.finetuner.init_loss()
            self.finetuner.finetune()

            # validate model
            acc = self.finetuner.validate()
            reward = self.reward(acc)

            if reward > self.best_reward:
                self.best_reward = reward
            
            obs = self.layer_embedding[self.cur_ind, :].copy()
            done = True
            info = {}
            return obs, reward, True, False, info 
        
        reward = 0 
        done = False

        self.cur_ind += 1
        self.index_to_quantize = self.quantizable_idx[self.cur_ind]
        self.layer_embedding[self.cur_ind][-1] = float(self.last_action)
        obs = self.layer_embedding[self.cur_ind, :].copy()
        info = {}
        return obs, reward, False, False, info

    def reward(self, acc):
        return (acc - self.org_acc) * 0.1
        
    def get_action(self, action):
        action = float(action[0])
        lbound, rbound = self.min_bit - 0.5, self.max_bit + 0.5
        action = (rbound - lbound) * action + lbound
        action = int(np.round(action, 0))
        return action

    def is_final_layer(self):
        return self.cur_ind == len(self.quantizable_idx) - 1
    
    def final_action_wall(self):
        # quantize model with the original strategy
        model_for_measure = copy.deepcopy(self.model)
        model_for_measure = self.quantizer.quantize_model(model_for_measure,
                                                          self.quantizable_idx,
                                                          self.num_quant_acts)
        
        # convert qonnx to finn, streamline and convert to hw

        # get original resources

        # if resources exceed, reduce the bitwidth and quantize again
        
        # if resources exceed, start from the end, reduce the bitwidth and quantize again
        # return the final strategy
        pass