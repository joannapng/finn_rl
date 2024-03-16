import argparse
import torchvision
import numpy as np
from train.env import ModelEnv
from pretrain.utils import get_model_config
from stable_baselines3 import DDPG

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
parser.add_argument('--scale-factor-type', default='float32', choices=['float32', 'po2'], help = 'Type for scale factors (default: float32)')
parser.add_argument('--act-bit-width', default=32, type=int, help = 'Activations bit width (default: 32)')
parser.add_argument('--weight-bit-width', default=32, type=int, help = 'Weight bit width (default: 32)')
parser.add_argument('--bias-bit-width', default=32, choices=[32, 16], help = 'Bias bit width (default: 32)')
parser.add_argument('--act-quant-type', default='symmetric', choices=['symmetric', 'assymetric'], help = 'Activation quantization type (default: symmetric)')
parser.add_argument('--act-quant-percentile', default=99.999, type=float, help = 'Percentile to use for stats of activation quantization (default: 99.999)')
parser.add_argument('--scaling-per-output-channel', default=True, help = 'Weight Scaling per output channel (default: enabled)')
parser.add_argument('--bias-corr', default=True, help = 'Bias correction after calibration (default: enabled)')
parser.add_argument('--weight-narrow-range', default=True, help = 'Narrow range for weight quantization (default: enabled)')
parser.add_argument('--min-bit', type=int, default=1, help = 'Minimum bit width (default: 1)')
parser.add_argument('--max-bit', type=int, default=8, help = 'Maximum bit width (default: 8)')

### ----- AGENT ------ ###
parser.add_argument('--num_agents', default = 5, type = int, help = 'Number of agents')

def main():
    args = parser.parse_args()
    envs = []
    agents = []
    weights = [[1.0, 0.0000], [1.0, 0.0025], [1.0, 0.0050], [1.0, 0.0075], [1.0, 0.0100]]

    for i in range(args.num_agents):
        envs.append(ModelEnv(args, np.array(weights[i]), get_model_config(args.model_name, args.custom_model_name)))
        agents.append(DDPG("MlpPolicy", envs[-1], verbose = 1))
    
    for i, agent in enumerate(agents):
        agent.learn(total_timesteps = 20, log_interval = 10)
        agent.save("agent_{}_{}".format(weights[i][0], weights[i][1]))
    
if __name__ == "__main__":
    main()