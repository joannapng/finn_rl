import argparse
import torchvision
from pretrain.trainer import Trainer
from pretrain.utils import get_model_config

model_names = sorted(name for name in torchvision.models.__dict__ if name.islower() and not name.startswith("__") and
                     callable(torchvision.models.__dict__[name]) and not name.startswith("get_"))

parser = argparse.ArgumentParser(description = 'Pretraining model parameters')

# Model Parameters
parser.add_argument('--model-name', default='resnet18', metavar='ARCH', choices=model_names,
                    help = 'model_architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--custom-model-name', default = None, help = 'Custom model architecture. Overrides --model-name')
parser.add_argument('--pretrained', action = 'store_true', default = False, help = 'Whether to use pretrained model')
parser.add_argument('--model-path', default = None, help = 'Path to pretrained model. Should be provided if pretrained is True')
parser.add_argument('--resume-from', default = None, help = 'If resume-from is not None, training resumes from specified checkpoint')

# Dataset Parameters
parser.add_argument('--datadir', default = './data', help='Directory where datasets are stored')
parser.add_argument('--dataset', default = 'MNIST', choices = ['MNIST', 'CIFAR10'], help = 'Name of dataset')
parser.add_argument('--batch-size-training', default = 128, type = int, help = 'Batch size for training')
parser.add_argument('--batch-size-validation', default = 64, type = int, help = 'Batch size for validation')
parser.add_argument('--num_workers', default = 32, type = int, help = 'Num workers')
parser.add_argument('--validation-split', default = 0.2, type = float, help = 'Training-Validation split')

# Trainer Parameters
parser.add_argument('--training-epochs', default = 10, type = int, help = 'Training epochs')
parser.add_argument('--save-dir', default = './checkpoints', help = 'Directory to save model and logs')
parser.add_argument('--print_every', type = int, default = 100, help = 'How frequent to print progress')
parser.add_argument('--checkpoint_every', default = 10, help = 'How many epochs to keep a checkpoint')

# Optimizer Parameters
parser.add_argument('--optimizer', default = 'Adam', choices = ['Adam', 'SGD'], help = 'Optimizer')
parser.add_argument('--training-lr', default = 0.01, type = float, help = 'Training learning rate')
parser.add_argument('--weight_decay', default = 5e-4, type = float, help = 'Weight decay for optimizer')
parser.add_argument('--momentum', default = 0.9, type = float, help = 'Value of momentum for optimizer')

# Scheduler Parameters
parser.add_argument('--scheduler', default = 'StepLR', choices = ['StepLR', 'CosineAnnealingLR'], help = 'Learning rate scheduler')
parser.add_argument('--step_size', default = 50, type = int, help = 'Period of learning decay')

# Loss Parameters
parser.add_argument('--loss', default = 'CrossEntropy', choices = ['CrossEntropy'], help = 'Loss Function for training')

# Device Parameters
parser.add_argument('--device', default = 'GPU', help = 'Device for training')

def main():
    args = parser.parse_args()
    trainer = Trainer(args, get_model_config(args.model_name, args.custom_model_name, args.dataset))
    trainer.train_model()

if __name__ == "__main__":
    main()