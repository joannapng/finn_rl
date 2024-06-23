import random
import torch
import argparse
from pretrain.trainer import Trainer
from pretrain.utils import get_model_config

model_names = ['LeNet5', 'resnet18', 'resnet34', 'resnet50', 'resnet100', 'resnet152', 'Simple']

parser = argparse.ArgumentParser(description = 'Pretraining model parameters')

# Model Parameters
parser.add_argument('--model-name', default='resnet18', metavar='ARCH', choices=model_names,
                    help = 'model_architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--pretrained', action = 'store_true', default = False, help = 'Whether to use pretrained model')
parser.add_argument('--model-path', default = None, help = 'Path to pretrained model. Should be provided if pretrained is True')
parser.add_argument('--resume-from', default = None, help = 'If resume-from is not None, training resumes from specified checkpoint')

# Dataset Parameters
parser.add_argument('--datadir', default = './data', help='Directory where datasets are stored')
parser.add_argument('--dataset', default = 'MNIST', choices = ['MNIST', 'CIFAR10'], help = 'Name of dataset')
parser.add_argument('--batch-size-training', default = 128, type = int, help = 'Batch size for training')
parser.add_argument('--batch-size-validation', default = 64, type = int, help = 'Batch size for validation')
parser.add_argument('--num-workers', default = 32, type = int, help = 'Num workers')
parser.add_argument('--validation-split', default = 0.2, type = float, help = 'Training-Validation split')

# Trainer Parameters
parser.add_argument('--training-epochs', default = 10, type = int, help = 'Training epochs')
parser.add_argument('--save-dir', default = './checkpoints', help = 'Directory to save model and logs')
parser.add_argument('--print_every', type = int, default = 100, help = 'How frequent to print progress')
parser.add_argument('--checkpoint_every', default = 10, help = 'How many epochs to keep a checkpoint')

# Optimizer Parameters
parser.add_argument('--optimizer', default = 'SGD', choices = ['Adam', 'SGD'], help = 'Optimizer')
parser.add_argument('--training-lr', default = 0.1, type = float, help = 'Training learning rate')
parser.add_argument('--weight_decay', default = 5e-4, type = float, help = 'Weight decay for optimizer')
parser.add_argument('--momentum', default = 0.9, type = float, help = 'Value of momentum for optimizer')

# Scheduler Parameters
parser.add_argument('--scheduler', default = 'StepLR', choices = ['StepLR', 'CosineAnnealingLR'], help = 'Learning rate scheduler')
parser.add_argument('--step_size', default = 200, type = int, help = 'Period of learning decay')

# Loss Parameters
parser.add_argument('--loss', default = 'CrossEntropy', choices = ['CrossEntropy'], help = 'Loss Function for training')

# Device Parameters
parser.add_argument('--device', default = 'GPU', help = 'Device for training')

parser.add_argument('--seed', default = 234, type = int, help = 'Seed for reproducibility')

def main():
    args = parser.parse_args()

    # set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'GPU' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    trainer = Trainer(args, get_model_config(args.model_name, args.dataset))
    trainer.train_model()

if __name__ == "__main__":
    main()