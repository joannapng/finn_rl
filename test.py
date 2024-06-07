'''
from pkgutil import get_data
import onnx
import onnx.numpy_helper as nph
import torch
import torchvision
import numpy as np
import argparse
from brevitas.graph.utils import get_module
import importlib_resources as importlib

from train.env import ModelEnv
from pretrain.utils import get_model_config
import brevitas.onnx as bo
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import DDPG

model_names = ['LeNet5', 'resnet18', 'resnet34', 'resnet50', 'resnet100', 'resnet152']
parser = argparse.ArgumentParser(description = 'Test RL Agents')

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

### ----- TARGET MODEL ------ ###
# Model Parameters
parser.add_argument('--model-name', default='resnet18', metavar='ARCH', choices=model_names,
                    help = 'model_architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
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

# TODO: add parameters for float quantization
# TODO: add PTQ extra steps
parser.add_argument('--min-bit', type=int, default=1, help = 'Minimum bit width (default: 1)')
parser.add_argument('--max-bit', type=int, default=8, help = 'Maximum bit width (default: 8)')

### ----- AGENT ------ ###
#parser.add_argument('--num-agents', default = 5, type = int, help = 'Number of agents')

def main():
    args = parser.parse_args()
    weights = [[1.0, 0.0]]

    env = Monitor(ModelEnv(args, np.array(weights[0]), get_model_config(args.model_name, args.dataset)), f'agent_{weights[0][0]}_{weights[0][1]}')
    agent = DDPG("MlpPolicy", env, action_noise = None, verbose = 1)

    rl_model = agent.load("agents/agent_{}_{}".format(weights[0][0], weights[0][1]))
    done = False
    obs, _ = env.reset()
    while not done:
        action, _states = rl_model.predict(obs)
        obs, rewards, done, _, info = env.step(action)
    
    model = env.model
    model = model.eval()

    model_config = get_model_config(args.model_name, args.dataset)
    center_crop_shape = model_config['center_crop_shape']
    img_shape = center_crop_shape
    
    input_tensor_npy = get_example_input(args.dataset)
    input_tensor_torch = torch.from_numpy(input_tensor_npy).float() / 255.0
    input_tensor_torch = input_tensor_torch.detach().to(env.finetuner.device)
    input_tensor_npy = np.transpose(input_tensor_npy, (0, 2, 3, 1)) # N, H, W, C
    np.save("input.npy", input_tensor_npy)

    output_golden = model.forward(input_tensor_torch).detach().cpu().numpy()
    print(output_golden)
    output_golden = np.flip(output_golden.flatten().argsort())[:1]
    print(output_golden)
    np.save("expected_output.npy", output_golden)

    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    ref_input = torch.randn(1, 3, img_shape, img_shape, device = device, dtype = dtype)

    # export original model to onnx
    orig_model = env.orig_model
    orig_model.eval()
    name = f'model_{weights[0][0]}_{weights[0][1]}.onnx'
    torch.onnx.export(orig_model, ref_input, name, export_params = False, opset_version=11)
    # export quant model to qonnx
    name = f'model_{weights[0][0]}_{weights[0][1]}_quant.onnx'
    bo.export_qonnx(model, ref_input, export_path = name, keep_initializers_as_inputs = False, opset_version=11)

if __name__ == "__main__":
    main()

'''

from pkgutil import get_data
import onnx
import onnx.numpy_helper as nph
import torch
import math
import numpy as np
import argparse
from brevitas.graph.utils import get_module
import importlib_resources as importlib

from train.env import ModelEnv
from pretrain.utils import get_model_config
from copy import deepcopy
from finn.util.basic import part_map
from agent.ddpg import DDPG
import random
import os
import brevitas.onnx as bo

model_names = ['LeNet5', 'resnet18', 'resnet34', 'resnet50', 'resnet100', 'resnet152']

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
parser.add_argument('--calib-subset', default = 0.1, type = float, help = 'Percentage of training dataset for calibration')
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
parser.add_argument('--hidden1', default = 300, type = int, help = 'Hidden num of first fully connected layer')
parser.add_argument('--hidden2', default = 300, type = int, help = 'Hidden num of second fully connected layer')
parser.add_argument('--lr_c', default = 1e-3, type = float, help = 'Learning rate for actor')
parser.add_argument('--lr_a', default = 1e-4, type = float, help = 'Learning rate for critic')
parser.add_argument('--warmup', default = 20, type = float, help = 'Time without training but only filling the replay memory')
parser.add_argument('--discount', default=1., type=float, help='')
parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
parser.add_argument('--rmsize', default=128, type=int, help='memory size for each layer')
parser.add_argument('--window_length', default=1, type=int, help='')
parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
parser.add_argument('--init_delta', default=0.5, type=float, help='initial variance of truncated normal distribution')
parser.add_argument('--delta_decay', default=0.99, type=float, help='delta decay during exploration')
parser.add_argument('--n_update', default=1, type=int, help='number of rl to update each time')
parser.add_argument('--output', default='./logs', type=str, help='')
parser.add_argument('--weights-dir', default = './logs', type = str, help = 'Path to actor and critic weights')

parser.add_argument('--num-episodes', default = 100, type = int, help = 'Number of episodes (passes over the entire network) to train the agent for')
parser.add_argument('--log-every', default = 10, type = int, help = 'How many episodes to wait to log agent')
parser.add_argument('--seed', default = 234, type = int, help = 'Seed to reproduce')
parser.add_argument('--init_w', default=0.003, type=float, help='')
parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')

### --- DESIGN --- ###
parser.add_argument('--board', default = "U250", help = "Name of target board")
parser.add_argument('--shell-flow-type', default = "vitis_alveo", choices = ["vivado_zynq", "vitis_alveo"], help = "Target shell type")
parser.add_argument('--freq', type = float, default = 200.0, help = 'Frequency in MHz')
parser.add_argument('--max-freq', type = float, default = 300.0, help = 'Maximum device frequency in MHz')
parser.add_argument('--target-fps', default = 6000, type = float, help = 'Target fps when target is accuracy')

parser.add_argument('--onnx-output', type = str, default = 'model.onnx', help = 'Onnx output name')

def test(agent, env, output):
    observation = None
    while True:
        if observation is None:
            observation = deepcopy(env.reset())
        
        agent.reset(observation)
        action = agent.select_action(observation, episode = args.warmup + 1)

        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)
        observation = deepcopy(observation2)

        if done:
            model = env.finetuner.model
            model.eval()

            model_config = get_model_config(args.model_name, args.dataset)
            center_crop_shape = model_config['center_crop_shape']
            img_shape = center_crop_shape
            
            input_tensor_npy = get_example_input(args.dataset)
            input_tensor_torch = torch.from_numpy(input_tensor_npy).float() / 255.0
            input_tensor_torch = input_tensor_torch.detach().to(env.finetuner.device)
            input_tensor_npy = np.transpose(input_tensor_npy, (0, 2, 3, 1)) # N, H, W, C
            np.save("input.npy", input_tensor_npy)

            output_golden = model.forward(input_tensor_torch).detach().cpu().numpy()
            output_golden = np.flip(output_golden.flatten().argsort())[:1]
            np.save("expected_output.npy", output_golden)

            device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
            ref_input = torch.randn(1, env.finetuner.in_channels, img_shape, img_shape, device = device, dtype = dtype)

            # export original model to onnx
            orig_model = env.orig_model
            orig_model.eval()
            name = output + '.onnx'
            torch.onnx.export(orig_model, ref_input, name, export_params = False, opset_version=11)
            
            # export quant model to qonnx
            name = output + '_quant.onnx'
            bo.export_qonnx(model, ref_input, export_path = name, keep_initializers_as_inputs = True, opset_version=11)

            break

args = parser.parse_args()
args.fpga_part = part_map[args.board]

# set seed to reproduce
random.seed(args.seed)
torch.manual_seed(args.seed)
if args.device == 'GPU' and torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# create environment
env = ModelEnv(args, get_model_config(args.model_name, args.dataset))

nb_actions = env.action_space.shape[-1]
nb_states = env.observation_space.shape[-1]

# create agent
agent = DDPG(nb_states, nb_actions, args)
agent.load_weights(args.weights_dir)
agent.eval()

# train agent
test(agent, env, args.onnx_output)
