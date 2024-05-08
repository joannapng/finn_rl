import torchvision
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.models.resnet import BasicBlock, Bottleneck
from torch import Tensor
from torch import hub

root_url = 'https://github.com/Xilinx/brevitas/releases/download/'
resnet18_url =  f"{root_url}/a2q_cifar10_r1/float_resnet18-1d98d23a.pth"

def get_model_config(model_name, custom_model_name, dataset):
    config = dict()

    if model_name == 'inception_v3' or model_name == 'googlenet':
        config['inception_preprocessing'] = True
    else:
        config['inception_preprocessing'] = False

    if dataset == "MNIST":
        input_shape = 28
        resize_shape = 28
    elif dataset == "CIFAR10":
        input_shape = 32
        resize_shape = 32

    #TODO: ADD FOR IMAGENET
    config.update({'resize_shape': resize_shape, 'center_crop_shape': input_shape})
    return config

    '''
    # Set-up config parameters
    if custom_model_name is not None:
        input_shape = 28
        resize_shape = 28
        config['inception_preprocessing'] = False
    else:
        # parameters for imagenet
        if model_name == 'inception_v3' or model_name == 'googlenet':
            config['inception_preprocessing'] = True
        else:
            config['inception_preprocessing'] = False

        if model_name == 'inception_v3':
            input_shape = 299
            resize_shape = 342
        else:
            input_shape = 224
            resize_shape = 256

    config.update({'resize_shape': resize_shape, 'center_crop_shape': input_shape})
    return config
    '''

def get_torchvision_model(model_name, num_classes, device, pretrained=False, dataset = "CIFAR10"):
    model_fn = getattr(torchvision.models, model_name)
    if model_name == 'inception_v3' or model_name == 'googlenet':
        if pretrained == True:
            model = model_fn(transform_input=False, num_classes = num_classes, weights = "DEFAULT")
        else:
            model = model_fn(transform_input=False, num_classes = num_classes)
    else:
        if pretrained == True:
            '''
            if dataset == "CIFAR10" and "resnet18" in model_name:
                model = model_fn(num_classes = num_classes)
                state_dict = hub.load_state_dict_from_url(resnet18_url, progress = True)
                model.load_state_dict(state_dict, strict=True)
            '''
            #else:
            model = model_fn(num_classes = num_classes, weights = "DEFAULT")
        else:
            model = model_fn(num_classes = num_classes)
    return model.to(device)

def add_relu_after_bn():
    BasicBlock.forward = basic_block_forward
    Bottleneck.forward = bottleneck_forward

def bottleneck_forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.relu(self.downsample(x))

        out = out + identity
        out = self.relu(out)

        return out

def basic_block_forward(self, x: Tensor) -> Tensor:
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    if self.downsample is not None:
        identity = self.relu(self.downsample(x))

    out = out + identity
    out = self.relu(out)

    return out
