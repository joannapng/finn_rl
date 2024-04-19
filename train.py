import argparse
import torch
import torchvision
import numpy as np
from train.env import ModelEnv
from pretrain.utils import get_model_config
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

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
parser.add_argument('--batch-size-validation', default = 64, type = int, help = 'Batch size for validation')
parser.add_argument('--num_workers', default = 32, type = int, help = 'Num workers')
parser.add_argument('--validation-split', default = 0.2, type = float, help = 'Validation split')
parser.add_argument('--finetuning-split', default = 0.2, type = float, help = 'Finetuning split')
parser.add_argument('--calib_subset', default = 0.1, type = float, help = 'Percentage of training dataset for calibration')

# Trainer Parameters
parser.add_argument('--finetuning-epochs', default = 2, type = int, help = 'Finetuning epochs')
parser.add_argument('--print_every', default = 100, help = 'How frequent to print progress')

# Optimizer Parameters
parser.add_argument('--optimizer', default = 'Adam', choices = ['Adam', 'SGD'], help = 'Optimizer')
parser.add_argument('--finetuning-lr', default = 1e-5, type = float, help = 'Training learning rate')
parser.add_argument('--weight_decay', default = 0, type = float, help = 'Weight decay for optimizer')

# Loss Parameters
parser.add_argument('--loss', default = 'CrossEntropy', choices = ['CrossEntropy'], help = 'Loss Function for training')

# Device Parameters
parser.add_argument('--device', default = 'GPU', help = 'Device for training')

### ----- QUANTIZATION PARAMETERS ----- ###
parser.add_argument('--scale-factor-type', default='float_scale', choices=['float_scale', 'po2_scale'], help = 'Type for scale factors (default: float)')
parser.add_argument('--act-bit-width', default=8, type=int, help = 'Activations bit width (default: 8)')
parser.add_argument('--weight-bit-width', default=8, type=int, help = 'Weight bit width (default: 8)')
parser.add_argument('--bias-bit-width', default=32, choices=[32, 16], help = 'Bias bit width (default: 32)')
parser.add_argument('--act-quant-type', default='sym', choices=['sym', 'asym'], help = 'Activation quantization type (default: sym)')
parser.add_argument('--weight-quant-type', default = 'sym', choices = ['sym', 'asym'], help = 'Weight quantization type (default: sym)')
parser.add_argument('--weight-quant-granularity', default = 'per_tensor', choices = ['per_tensor', 'per_channel'], help = 'Activation Quantization type (default: per_tensor)')
parser.add_argument('--weight-quant-calibration-type', default = 'stats', choices = ['stats', 'mse'], help = 'Weight quantization calibration type (default: stats)')
parser.add_argument('--act-equalization', default = None, choices = ['fx', 'layerwise', 'None'], help = 'Activation equalization type (default: None)')
parser.add_argument('-act-quant-calibration-type', default = 'stats', choices = ['stats', 'mse'], help = 'Activation quantization calibration type (default: stats)')
parser.add_argument('--act-quant-percentile', default=99.999, type=float, help = 'Percentile to use for stats of activation quantization (default: 99.999)')
parser.add_argument('--graph-eq-iterations', default = 20, type = int, help = 'Number of iterations for graph equalization (default: 20)')
parser.add_argument('--learned-round-iters', default = 1000, type = int, help = 'Number of iterations for learned round for each layer (default: 1000)')
parser.add_argument('--learned-round-lr', default = 1e-3, type = float, help = 'Learning rate for learned round (default: 1e-3)')
parser.add_argument('--scaling-per-output-channel', default=True, action = 'store_true', help = 'Weight Scaling per output channel (default: enabled)')
parser.add_argument('--bias-corr', default=True, action = 'store_true', help = 'Bias correction after calibration (default: enabled)')
parser.add_argument('--graph-eq-merge-bias', default = True, action = 'store_true', help = 'Merge bias when performing graph equaltion (default: enabled)')
parser.add_argument('--weight-narrow-range', default=False, help = 'Narrow range for weight quantization (default: enabled)')
parser.add_argument('--gpfq-p', default=1.0, type=float, help='P parameter for GPFQ (default: 1.0)')
parser.add_argument('--quant-format', default = 'int', choices = ['int', 'float'], help = 'Quantization format to use for weights and activations (default: int)')
parser.add_argument('--merge-bn', default = True, help = 'Merge BN layers before quantizing the model (default: enabled)')

parser.add_argument('--min-bit', type=int, default=1, help = 'Minimum bit width (default: 1)')
parser.add_argument('--max-bit', type=int, default=8, help = 'Maximum bit width (default: 8)')

### ----- AGENT ------ ###
parser.add_argument('--num_agents', default = 5, type = int, help = 'Number of agents')
parser.add_argument('--agent', default = 'TD3', choices = ['A2C', 'DDPG', 'PPO', 'SAC', 'TD3'], help = 'Choose algorithm to train agent')
parser.add_argument('--noise', default = 0.1, type = float, help = 'Std for added noise in agent')
parser.add_argument('--num_episodes', default = 100, type = int, help = 'Number of episodes (passes over the entire network) to train the agent for')
parser.add_argument('--log_every', default = 10, type = int, help = 'How many episodes to wait to log agent')

def get_weights(num_agents):
    weights = []

    for i in range(num_agents):
        w1 = i * 0.1
        w2 = 1 - w1
        weights.append([w1, w2])
    
    return weights

def main():
    args = parser.parse_args()
    num_agents = args.num_agents
    envs = []
    agents = []
    weights = get_weights(num_agents)

    for i in range(num_agents):
        env = ModelEnv(args, np.array(weights[i]), get_model_config(args.model_name, args.custom_model_name))
        envs.append(Monitor(env, f'agent_{weights[i][0]}_{weights[i][1]}', info_keywords = ('accuracy',)))

        n_actions = envs[-1].action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=args.noise * np.ones(n_actions))

        agent = rl_algorithms[args.agent]
        agents.append(agent("MlpPolicy", envs[-1], action_noise = action_noise, verbose = 1))
    
    for i, agent in enumerate(agents):
        agent.learn(total_timesteps = len(envs[i].quantizable_idx) * args.num_episodes, log_interval = args.log_every)
        agent.save("agents/agent_{}_{}".format(weights[i][0], weights[i][1]))
    
if __name__ == "__main__":
    main()