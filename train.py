import os
import torch
import random
import argparse
import numpy as np

from train.env import ModelEnv
from pretrain.utils import get_model_config

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from train.callbacks.StopTrainingOnNoImprovementCallback import StopTrainingOnNoImprovementCallback

from finn.util.basic import part_map

rl_algorithms = {
    'A2C': A2C,
    'DDPG': DDPG,
    'PPO': PPO,
    'SAC': SAC,
    'TD3': TD3
}

model_names = ['LeNet5', 'resnet18', 'resnet34', 'resnet50', 'resnet100', 'resnet152', 'Simple']

parser = argparse.ArgumentParser(description = 'Train RL Agent')

# Model Parameters
parser.add_argument('--model-name', default='resnet18', metavar='ARCH', choices=model_names,
                    help = 'model_architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--model-path', default = None, help = 'Path to pretrained model')

# Dataset Parameters
parser.add_argument('--datadir', default = './data', help='Directory where datasets are stored (default: ./data)')
parser.add_argument('--dataset', default = 'MNIST', choices = ['MNIST', 'CIFAR10'], help = 'Name of dataset (default: MNIST)')
parser.add_argument('--batch-size-finetuning', default = 64, type = int, help = 'Batch size for finetuning (default: 64)')
parser.add_argument('--batch-size-testing', default = 64, type = int, help = 'Batch size for testing (default: 64)')
parser.add_argument('--num-workers', default = 32, type = int, help = 'Num workers (default: 32)')
parser.add_argument('--calib-subset', default = 0.1, type = float, help = 'Percentage of training dataset for calibration (default: 0.1)')
parser.add_argument('--finetuning-subset', default = 0.5, type = float, help = 'Percentage of dataset to use for finetuning (default: 0.5)')

# Trainer Parameters
parser.add_argument('--finetuning-epochs', default = 5, type = int, help = 'Finetuning epochs (default: 5)')
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
parser.add_argument('--bias-corr', default=True, action = 'store_true', help = 'Bias correction after calibration (default: enabled)')
parser.add_argument('--min-bit', type=int, default=1, help = 'Minimum bit width (default: 1)')
parser.add_argument('--max-bit', type=int, default=8, help = 'Maximum bit width (default: 8)')

# Agent Parameters
parser.add_argument('--agent', default = 'TD3', choices = ['A2C', 'DDPG', 'PPO', 'SAC', 'TD3'], help = 'Choose algorithm to train agent (default: TD3)')
parser.add_argument('--noise', default = 0.1, type = float, help = 'Std for added noise in agent (default: 0.1)')
parser.add_argument('--num-episodes', default = 500, type = int, help = 'Number of episodes to train the agent for (default: 500)')
parser.add_argument('--log-every', default = 10, type = int, help = 'How many episodes to wait to log agent (default: 10)')
parser.add_argument('--save-every', default = 10, type = int, help = 'How many episodes to wait to save agent checkpoint (default: 10)')
parser.add_argument('--seed', default = 234, type = int, help = 'Seed to reproduce (default: 234)')

# Design Parameters
parser.add_argument('--board', default = "U250", help = "Name of target board (default: U250)")
parser.add_argument('--shell-flow-type', default = "vitis_alveo", choices = ["vivado_zynq", "vitis_alveo"], help = "Target shell type (default: vitis_alveo)")
parser.add_argument('--freq', type = float, default = 200.0, help = 'Frequency in MHz (default: 200)')
parser.add_argument('--max-freq', type = float, default = 300.0, help = 'Maximum device frequency in MHz (default: 300)')
parser.add_argument('--target-fps', default = 6000, type = float, help = 'Target fps (default: 6000)')

def main():
    args = parser.parse_args()

    # set seed to reproduce
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == 'GPU' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    args.fpga_part = part_map[args.board]
    args.output_dir = args.model_name

    env = Monitor(
        ModelEnv(args, get_model_config(args.dataset)),
        filename = 'monitor.csv',
        info_keywords=('accuracy', 'fps', 'avg_util', 'strategy')
    )

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=args.noise * np.ones(n_actions))

    agent = rl_algorithms[args.agent]("MlpPolicy", env, action_noise = action_noise, verbose = 1, seed = args.seed)
    
    # check every log_every episodes for improvement and if after 3 checks the model has not progressed, end training
    stop_train_callback = StopTrainingOnNoImprovementCallback(check_freq=len(env.quantizable_idx) * args.log_every, patience = 3)
    checkpoint_callback = CheckpointCallback(save_freq = args.save_every * len(env.quantizable_idx), save_path = 'agents', name_prefix = f'agent_{args.model_name}') 
    agent.learn(total_timesteps=len(env.quantizable_idx) * args.num_episodes, 
                log_interval=args.log_every,
                callback = [stop_train_callback, checkpoint_callback])
    agent.save(f'agents/agent_{args.model_name }')
    
if __name__ == "__main__":
    main()