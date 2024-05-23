import argparse
from gc import callbacks
import torch
import torchvision
import numpy as np
from train.env import ModelEnv
from pretrain.utils import get_model_config
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from train.callbacks.StopTrainingOnNoImprovementCallback import StopTrainingOnNoImprovementCallback
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from copy import deepcopy
import multiprocessing as mp
from finn.util.basic import part_map, alveo_default_platform

rl_algorithms = {
    'A2C': A2C,
    'DDPG': DDPG,
    'PPO': PPO,
    'SAC': SAC,
    'TD3': TD3
}

model_names = sorted(name for name in torchvision.models.__dict__ if name.islower() and not name.startswith("__") and
                     callable(torchvision.models.__dict__[name]) and not name.startswith("get_"))

parser = argparse.ArgumentParser(description = 'Train RL Agents')

### ----- TARGET MODEL ------ ###
# Model Parameters
parser.add_argument('--model-name', default='resnet18', metavar='ARCH', choices=model_names,
                    help = 'model_architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--custom-model-name', default = None, help = 'Custom model architecture. Overrides --model-name')
parser.add_argument('--model-path', default = None, help = 'Path to pretrained model')

# Dataset Parameters
parser.add_argument('--datadir', default = './data', help='Directory where datasets are stored')
parser.add_argument('--dataset', default = 'MNIST', choices = ['MNIST', 'CIFAR10'], help = 'Name of dataset')
parser.add_argument('--batch-size-finetuning', default = 64, type = int, help = 'Batch size for finetuning')
parser.add_argument('--batch-size-testing', default = 64, type = int, help = 'Batch size for testing')
parser.add_argument('--num-workers', default = 32, type = int, help = 'Num workers')
parser.add_argument('--calib-subset', default = 0.1, type = float, help = 'Percentage of training dataset for calibration')
parser.add_argument('--finetuning-subset', default = 0.5, type = float, help = 'Percentage of dataset to use for finetuning')

# Trainer Parameters
parser.add_argument('--finetuning-epochs', default = 5, type = int, help = 'Finetuning epochs')
parser.add_argument('--print-every', default = 100, type = int, help = 'How frequent to print progress')

# Optimizer Parameters
parser.add_argument('--optimizer', default = 'Adam', choices = ['Adam', 'SGD'], help = 'Optimizer')
parser.add_argument('--finetuning-lr', default = 1e-5, type = float, help = 'Training learning rate')
parser.add_argument('--weight-decay', default = 0, type = float, help = 'Weight decay for optimizer')

# Loss Parameters
parser.add_argument('--loss', default = 'CrossEntropy', choices = ['CrossEntropy'], help = 'Loss Function for training')

# Device Parameters
parser.add_argument('--device', default = 'GPU', help = 'Device for training')

### ----- QUANTIZATION PARAMETERS ----- ###
parser.add_argument('--scale-factor-type', default='float_scale', choices=['float_scale', 'po2_scale'], help = 'Type for scale factors (default: float)')
parser.add_argument('--act-bit-width', default=4, type=int, help = 'Activations bit width (default: 4)')
parser.add_argument('--weight-bit-width', default=4, type=int, help = 'Weight bit width (default: 4)')
parser.add_argument('--bias-bit-width', default=8, choices=[32, 16, 8], help = 'Bias bit width (default: 8)')
parser.add_argument('--bias-corr', default=True, action = 'store_true', help = 'Bias correction after calibration (default: enabled)')
parser.add_argument('--min-bit', type=int, default=1, help = 'Minimum bit width (default: 1)')
parser.add_argument('--max-bit', type=int, default=8, help = 'Maximum bit width (default: 8)')

### ----- AGENT ------ ###
parser.add_argument('--agent', default = 'TD3', choices = ['A2C', 'DDPG', 'PPO', 'SAC', 'TD3'], help = 'Choose algorithm to train agent')
parser.add_argument('--noise', default = 0.1, type = float, help = 'Std for added noise in agent')
parser.add_argument('--num-episodes', default = 100, type = int, help = 'Number of episodes (passes over the entire network) to train the agent for')
parser.add_argument('--log-every', default = 10, type = int, help = 'How many episodes to wait to log agent')

### --- DESIGN --- ###
parser.add_argument('--synth-clk-period-ns', type = float, default = 10.0, help = 'Target clock period in ns')
parser.add_argument('--board', default = "U250", help = "Name of target board")
parser.add_argument('--shell-flow-type', default = "vitis_alveo", choices = ["vivado_zynq", "vitis_alveo"], help = "Target shell type")
parser.add_argument('--min-target-fps', type = int, default = 60, help = 'Minimum target fps')
parser.add_argument('--max-target-fps', type = int, default = 100000, help = 'Maximum target fps')

def main():
    args = parser.parse_args()
    args.fpga_part = part_map[args.board]

    env = ModelEnv(args, get_model_config(args.model_name, args.custom_model_name, args.dataset))
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=args.noise * np.ones(n_actions))
    agent = rl_algorithms[args.agent]("MlpPolicy", env, action_noise = action_noise, verbose = 1)
    stop_train_callback = StopTrainingOnNoImprovementCallback(check_freq=500, patience = 3)
    agent.learn(total_timesteps=len(env.quantizable_idx) * args.num_episodes, 
                log_interval=args.log_every,
                callback=stop_train_callback)
    agent.save("agents/agent")
    
if __name__ == "__main__":
    main()