import onnx
import onnx.numpy_helper as nph
import argparse
import torch
import torchvision
import numpy as np
from train.env import ModelEnv
from pretrain.utils import get_model_config
import brevitas.onnx as bo
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import DDPG
from pkgutil import get_data

def get_example_input(dataset):
    # TODO: do something different here
    "Get example numpy input tensor for given dataset."

    if dataset == "MNIST":
        raw_i = get_data("qonnx.data", "onnx/mnist-conv/test_data_set_0/input_0.pb")
        onnx_tensor = onnx.load_tensor_from_string(raw_i)
        return nph.to_array(onnx_tensor)
    elif dataset == "CIFAR10":
        input_tensor = np.load("./data/cifar10-test-data-class3.npz")["arr_0"].astype(np.float32)
        return input_tensor
    else:
        raise Exception("Unknown dataset, can't return example input")
    

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
parser.add_argument('--act-bit-width', default=8, type=int, help = 'Activations bit width (default: 32)')
parser.add_argument('--weight-bit-width', default=8, type=int, help = 'Weight bit width (default: 32)')
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
parser.add_argument('--weight-narrow-range', default=True, help = 'Narrow range for weight quantization (default: enabled)')
parser.add_argument('--gpfq-p', default=1.0, type=float, help='P parameter for GPFQ (default: 1.0)')
parser.add_argument('--quant-format', default = 'int', choices = ['int', 'float'], help = 'Quantization format to use for weights and activations (default: int)')

# TODO: add parameters for float quantization
# TODO: add PTQ extra steps
parser.add_argument('--min-bit', type=int, default=1, help = 'Minimum bit width (default: 1)')
parser.add_argument('--max-bit', type=int, default=8, help = 'Maximum bit width (default: 8)')

### ----- AGENT ------ ###
#parser.add_argument('--num-agents', default = 5, type = int, help = 'Number of agents')

def main():
    args = parser.parse_args()
    weights = [[0.5, 0.5]]

    env = Monitor(ModelEnv(args, np.array(weights[0]), get_model_config(args.model_name, args.custom_model_name)), f'agent_{weights[0][0]}_{weights[0][1]}')
    agent = DDPG("MlpPolicy", env, action_noise = None, verbose = 1)

    rl_model = agent.load("agents/agent_{}_{}".format(weights[0][0], weights[0][1]))
    done = False
    obs, _ = env.reset()
    while not done:
        action, _states = rl_model.predict(obs)
        obs, rewards, done, _, info = env.step(action)
    
    model = env.model
    model = model.eval()
    input_tensor_npy = get_example_input(args.dataset)
    input_tensor_torch = torch.from_numpy(input_tensor_npy).float()
    input_tensor_torch /= 255
    input_tensor_torch = input_tensor_torch.detach().to(env.finetuner.device)
    np.save("input.npy", input_tensor_npy)

    output_golden = model.forward(input_tensor_torch).detach().cpu().numpy()
    output_golden = np.flip(output_golden.flatten().argsort())[:1]
    np.save("expected_output.npy", output_golden)

    model_config = get_model_config(args.model_name, args.custom_model_name)
    center_crop_shape = model_config['center_crop_shape']
    img_shape = center_crop_shape
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    ref_input = torch.ones(1, 1, img_shape, img_shape, device = device, dtype = dtype)

    # export original model to onnx
    orig_model = env.orig_model
    orig_model.eval()
    name = f'model_{weights[0][0]}_{weights[0][1]}.onnx'
    torch.onnx.export(orig_model, ref_input, name, export_params = False, opset_version = 9)
    # export quant model to qonnx
    name = f'model_{weights[0][0]}_{weights[0][1]}_quant.onnx'
    bo.export_qonnx(model, ref_input, export_path = name, opset_version=9)

if __name__ == "__main__":
    main()