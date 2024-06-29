import random
import torch
import argparse
from pretrain.trainer import Trainer
from pretrain.utils import get_model_config

parser = argparse.ArgumentParser(description = 'Pretraining model parameters')

# Model Parameters
parser.add_argument('--model-name', default='resnet18', help = 'Target model name')
parser.add_argument('--pretrained', action = 'store_true', default = False, help = 'Whether to use pretrained model (default: false)')
parser.add_argument('--model-path', default = None, help = 'Path to pretrained model. Should be provided if pretrained is True (default: None)')
parser.add_argument('--resume-from', default = None, help = 'If resume-from is not None, training resumes from specified checkpoint (default: None)')

# Dataset Parameters
parser.add_argument('--datadir', default = './data', help='Directory where datasets are stored (default: ./data)')
parser.add_argument('--dataset', default = 'CIFAR10', choices = ['MNIST', 'CIFAR10'], help = 'Name of dataset (default: CIFAR10)')
parser.add_argument('--batch-size-training', default = 128, type = int, help = 'Batch size for training (default: 128)')
parser.add_argument('--batch-size-validation', default = 64, type = int, help = 'Batch size for validation (default: 64)')
parser.add_argument('--num-workers', default = 8, type = int, help = 'Num workers (default: 8)')
parser.add_argument('--validation-split', default = 0.2, type = float, help = 'Training-Validation split (default: 0.2)')

# Trainer Parameters
parser.add_argument('--training-epochs', type = int, default = 100, help = 'Training epochs (default: 100)')
parser.add_argument('--save-dir', default = './checkpoints', help = 'Directory to save model and logs (default: ./checkpoints)')
parser.add_argument('--print_every', type = int, default = 100, help = 'How frequent to print progress (default: 100)')
parser.add_argument('--checkpoint_every', type = int, default = 10, help = 'How many epochs to keep a checkpoint (default: 10)')

# Optimizer Parameters
parser.add_argument('--optimizer', default = 'SGD', choices = ['Adam', 'SGD'], help = 'Optimizer (default: SGD)')
parser.add_argument('--training-lr', default = 0.1, type = float, help = 'Training learning rate (default: 0.1)')
parser.add_argument('--weight_decay', default = 5e-4, type = float, help = 'Weight decay for optimizer (default: 5e-4)')
parser.add_argument('--momentum', default = 0.9, type = float, help = 'Value of momentum for optimizer (default: 0.9)')

# Scheduler Parameters
parser.add_argument('--scheduler', default = 'StepLR', choices = ['StepLR', 'CosineAnnealingLR'], help = 'Learning rate scheduler (default: StepLR)')
parser.add_argument('--step_size', default = 200, type = int, help = 'Period of learning decay (default: 200)')

# Loss Parameters
parser.add_argument('--loss', default = 'CrossEntropy', choices = ['CrossEntropy'], help = 'Loss Function for training (default: CrossEntropy)')

# Device Parameters
parser.add_argument('--device', default = 'GPU', help = 'Device for training (default: GPU)')

parser.add_argument('--seed', default = 234, type = int, help = 'Seed for reproducibility (default: 234)')

def main():
    args = parser.parse_args()

    # set seed for reproducability
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == 'GPU' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    trainer = Trainer(args, get_model_config(args.dataset))
    trainer.train_model()

if __name__ == "__main__":
    main()