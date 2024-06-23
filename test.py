from pkgutil import get_data
import attr
import onnx
import onnx.numpy_helper as nph
import torch
import brevitas
from brevitas import config

import math
import numpy as np
import argparse
from brevitas.graph.utils import get_module
import importlib_resources as importlib
from urllib3 import disable_warnings

from train.env import ModelEnv
from stable_baselines3.common.monitor import Monitor
from pretrain.utils import get_model_config
from copy import deepcopy
from finn.util.basic import part_map
import random
import os
import brevitas.onnx as bo
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from copy import deepcopy
import multiprocessing as mp

rl_algorithms = {
    'A2C': A2C,
    'DDPG': DDPG,
    'PPO': PPO,
    'SAC': SAC,
    'TD3': TD3
}

model_names = ['LeNet5', 'resnet18', 'resnet34', 'resnet50', 'resnet100', 'resnet152', 'Simple']

def get_example_input(dataset):
    if dataset == "MNIST":
        raw_i = get_data("qonnx.data", "onnx/mnist-conv/test_data_set_0/input_0.pb")
        input_tensor = onnx.load_tensor_from_string(raw_i)
        input_tensor_npy = nph.to_array(input_tensor).copy()
    elif dataset == "CIFAR10":
        ref = importlib.files("finn.qnn-data") / "cifar10/cifar10-test-data-class3.npz"
        with importlib.as_file(ref) as fn:
            input_tensor_npy = np.load(fn)["arr_0"].astype(np.float32)
    
    return input_tensor_npy

parser = argparse.ArgumentParser(description = 'Train RL Agent')

### ----- TARGET MODEL ------ ###
parser.add_argument('--model-name', default='resnet18', metavar='ARCH', choices=model_names,
                    help = 'model_architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--model-path', default = None, help = 'Path to pretrained model')

### ----- DATASET PARAMETERS ----- ###
parser.add_argument('--datadir', default = './data', help='Directory where datasets are stored')
parser.add_argument('--dataset', default = 'MNIST', choices = ['MNIST', 'CIFAR10'], help = 'Name of dataset')
parser.add_argument('--batch-size-finetuning', default = 64, type = int, help = 'Batch size for finetuning')
parser.add_argument('--batch-size-testing', default = 64, type = int, help = 'Batch size for testing')
parser.add_argument('--num-workers', default = 32, type = int, help = 'Num workers')
parser.add_argument('--calib-subset', default = 0.5, type = float, help = 'Percentage of training dataset for calibration')
parser.add_argument('--finetuning-subset', default = 0.5, type = float, help = 'Percentage of dataset to use for finetuning')

### ----- FINETUNER PARAMETERS ----- ###
parser.add_argument('--finetuning-epochs', default = 5, type = int, help = 'Finetuning epochs')
parser.add_argument('--print-every', default = 100, type = int, help = 'How frequent to print progress')
parser.add_argument('--optimizer', default = 'Adam', choices = ['Adam', 'SGD'], help = 'Optimizer')
parser.add_argument('--finetuning-lr', default = 1e-5, type = float, help = 'Training learning rate')
parser.add_argument('--weight-decay', default = 0, type = float, help = 'Weight decay for optimizer')
parser.add_argument('--loss', default = 'CrossEntropy', choices = ['CrossEntropy'], help = 'Loss Function for training')

# Device Parameters
parser.add_argument('--device', default = 'GPU', help = 'Device for training')

### ----- QUANTIZATION PARAMETERS ----- ###
parser.add_argument('--scale-factor-type', default='float_scale', choices=['float_scale', 'po2_scale'], help = 'Type for scale factors (default: float)')
parser.add_argument('--act-bit-width', default=4, type=int, help = 'Default activations bit width (default: 4)')
parser.add_argument('--weight-bit-width', default=4, type=int, help = 'Default weight bit width (default: 4)')
parser.add_argument('--bias-bit-width', default=8, choices=[32, 16, 8], help = 'Bias bit width (default: 8)')
parser.add_argument('--bias-corr', default=True, action = 'store_true', help = 'Bias correction after calibration (default: enabled)')
parser.add_argument('--min-bit', type=int, default=1, help = 'Minimum bit width (default: 1)')
parser.add_argument('--max-bit', type=int, default=8, help = 'Maximum bit width (default: 8)')

### ----- AGENT ------ ###
parser.add_argument('--agent', default = 'TD3', choices = ['A2C', 'DDPG', 'PPO', 'SAC', 'TD3'], help = 'Choose algorithm to train agent')
parser.add_argument('--noise', default = 0.1, type = float, help = 'Std for added noise in agent')
parser.add_argument('--num-episodes', default = 500, type = int, help = 'Number of episodes (passes over the entire network) to train the agent for')
parser.add_argument('--log-every', default = 10, type = int, help = 'How many episodes to wait to log agent')
parser.add_argument('--output', default='./logs', type=str, help='')
parser.add_argument('--seed', default = 234, type = int, help = 'Seed to reproduce')

### --- DESIGN --- ###
parser.add_argument('--board', default = "U250", help = "Name of target board")
parser.add_argument('--shell-flow-type', default = "vitis_alveo", choices = ["vivado_zynq", "vitis_alveo"], help = "Target shell type")
parser.add_argument('--freq', type = float, default = 200.0, help = 'Frequency in MHz')
parser.add_argument('--max-freq', type = float, default = 300.0, help = 'Maximum device frequency in MHz')
parser.add_argument('--target-fps', default = 6000, type = float, help = 'Target fps when target is accuracy')

parser.add_argument('--output-dir', type = str, default = 'Model', help = 'Output dir for exported models')
parser.add_argument('--onnx-output', type = str, default = 'model', help = 'Onnx output name')

args = parser.parse_args()
args.fpga_part = part_map[args.board]

# set seed to reproduce
random.seed(args.seed)
torch.manual_seed(args.seed)
if args.device == 'GPU' and torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# create environment
env = Monitor(
    ModelEnv(args, get_model_config(args.model_name, args.dataset)))

n_actions = env.action_space.shape[-1]
agent = rl_algorithms[args.agent]("MlpPolicy", env, action_noise = None, verbose = 1)
rl_model = agent.load(f'agents/agent_{args.model_name}')

done = False
obs, _ = env.reset()
while not done:
    action, _states = rl_model.predict(obs)
    obs, rewards, done, _, info = env.step(action)

model = deepcopy(env.model)
model.eval()

model_config = get_model_config(args.model_name, args.dataset)
center_crop_shape = model_config['center_crop_shape']
img_shape = center_crop_shape

input_tensor_torch, _ = next(iter(env.finetuner.export_loader))
input_tensor_numpy = input_tensor_torch.detach().cpu().numpy().astype(np.float32)
input_tensor_numpy = np.transpose(input_tensor_numpy, (0, 2, 3, 1))
input_tensor_torch = input_tensor_torch / 255.0
np.save(f'{os.path.join(args.output_dir, "input.npy")}', input_tensor_numpy)

output_golden = model.forward(input_tensor_torch.to(env.finetuner.device)).detach().cpu().numpy()
output_golden = np.argmax(output_golden, axis = 1)
np.save(f'{os.path.join(args.output_dir, "expected_output.npy")}', output_golden)

# export quant model to qonnx
output = os.path.join(args.output_dir, args.onnx_output)
model.cpu()
device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
ref_input = torch.randn(1, env.finetuner.in_channels, img_shape, img_shape, device = device, dtype = dtype)

name = output + '_quant.onnx'
bo.export_qonnx(model, input_t = ref_input, export_path = name, opset_version = 11, keep_initializers_as_inputs = False)

# export original model to onnx
orig_model = env.orig_model
orig_model.eval()
orig_model.cpu()
name = output + '.onnx'
torch.onnx.export(orig_model, ref_input, name, export_params = True, opset_version=11)

