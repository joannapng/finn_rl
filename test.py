import os
import torch
import random
import numpy as np
import argparse
from copy import deepcopy

from train.env import ModelEnv
from pretrain.utils import get_model_config
from finn.util.basic import part_map
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

model_names = ['LeNet5', 'resnet18', 'resnet34', 'resnet50', 'resnet100', 'resnet152']

parser = argparse.ArgumentParser(description = 'Test RL Agent')

# Model Parameters
parser.add_argument('--model-name', default='resnet18', metavar='ARCH', choices=model_names,
                    help = 'model_architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--model-path', default = None, help = 'Path to pretrained model')

# Dataset Parameters
parser.add_argument('--datadir', default = './data', help='Directory where datasets are stored (default: ./data)')
parser.add_argument('--dataset', default = 'CIFAR10', choices = ['MNIST', 'CIFAR10'], help = 'Name of dataset (default: CIFAR10)')
parser.add_argument('--batch-size-finetuning', default = 64, type = int, help = 'Batch size for finetuning (default: 64)')
parser.add_argument('--batch-size-testing', default = 64, type = int, help = 'Batch size for testing (default: 64)')
parser.add_argument('--num-workers', default = 16, type = int, help = 'Num workers (default: 32)')
parser.add_argument('--calib-subset', default = 0.1, type = float, help = 'Percentage of training dataset for calibration (default: 0.1)')
parser.add_argument('--finetuning-subset', default = 0.5, type = float, help = 'Percentage of dataset to use for finetuning (default: 0.5)')

# Trainer Parameters
parser.add_argument('--finetuning-epochs', default = 2, type = int, help = 'Finetuning epochs (default: 2)')
parser.add_argument('--print-every', default = 100, type = int, help = 'How frequent to print progress (default: 100)')

# Optimizer Parameters
parser.add_argument('--optimizer', default = 'Adam', choices = ['Adam', 'SGD'], help = 'Optimizer (default: Adam)')
parser.add_argument('--finetuning-lr', default = 1e-5, type = float, help = 'Training finetuning learning rate (default: 1e-5)')
parser.add_argument('--weight-decay', default = 0, type = float, help = 'Weight decay for optimizer (default: 0)')

# Loss Parameters
parser.add_argument('--loss', default = 'CrossEntropy', choices = ['CrossEntropy'], help = 'Loss Function for training (default: CrossEntropy)')

# Device Parameters
parser.add_argument('--device', default = 'GPU', help = 'Device for training (default: GPU)')

# Quantization Parameters
parser.add_argument('--act-bit-width', default=4, type=int, help = 'Bit width for activations (default: 4)')
parser.add_argument('--weight-bit-width', default=4, type=int, help = 'Bit width for weights (default: 4)')
parser.add_argument('--min-bit', type=int, default=1, help = 'Minimum bit width (default: 1)')
parser.add_argument('--max-bit', type=int, default=8, help = 'Maximum bit width (default: 8)')

# Agent Parameters
parser.add_argument('--agent', default = 'TD3', choices = ['A2C', 'DDPG', 'PPO', 'SAC', 'TD3'], help = 'Choose algorithm to train agent (default: TD3)')
parser.add_argument('--seed', default = 234, type = int, help = 'Seed to reproduce (default: 234)')

# Design Parameters
parser.add_argument('--board', default = "U250", help = "Name of target board (default: U250)")
parser.add_argument('--shell-flow-type', default = "vitis_alveo", choices = ["vivado_zynq", "vitis_alveo"], help = "Target shell type (default: vitis_alveo)")
parser.add_argument('--freq', type = float, default = 300.0, help = 'Frequency in MHz (default: 300)')
parser.add_argument('--max-freq', type = float, default = 300.0, help = 'Maximum device frequency in MHz (default: 300)')
parser.add_argument('--target-fps', default = 6000, type = float, help = 'Target fps (default: 6000)')

parser.add_argument('--agent-path', type = str, default = '', help = 'Path to agent checkpoint')
parser.add_argument('--output-dir', type = str, default = 'Model', help = 'Output dir for exported models (default: Model)')
parser.add_argument('--onnx-output', type = str, default = 'model', help = 'Onnx output name (default: model)')

parser.add_argument('--use-custom-strategy', action = 'store_true', default = False, help = 'Use custom quantization strategy (overrides agent parameter, default: False)')
parser.add_argument('--strategy', type = str, default = '', help = 'Custom quantization strategy (example input: \"[7, 8, 1, 3, 1, 4]\")')

args = parser.parse_args()
args.fpga_part = part_map[args.board]

# set seed to reproduce
random.seed(args.seed)
torch.manual_seed(args.seed)

if args.device == 'GPU' and torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# create environment
env = ModelEnv(args, get_model_config(args.dataset), testing = True)

n_actions = env.action_space.shape[-1]


done = False
obs, _ = env.reset()

if not args.use_custom_strategy:
    agent = rl_algorithms[args.agent]("MlpPolicy", env, action_noise = None, verbose = 1)
    rl_model = agent.load(args.agent_path)
    while not done:
        action, _states = rl_model.predict(obs)
        obs, rewards, done, _, info = env.step(action)
else:
    # convert string strategy to list
    strategy = args.strategy.replace("[", "").replace("]", "")
    strategy = list(strategy.split(", "))
    strategy = [int(s) for s in strategy]

    idx = 0
    while not done:
        action = strategy[idx]
        done, info = env.step_(action)
        idx += 1

model = deepcopy(env.model)
model.eval()

model_config = get_model_config(args.dataset)
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

