import torchvision
import torch
import torch.nn as nn
import torch.nn.init as init

def get_model_config(model_name, custom_model_name):
    config = dict()
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

def get_torchvision_model(model_name, num_classes, device, pretrained=False):
    model_fn = getattr(torchvision.models, model_name)
    if model_name == 'inception_v3' or model_name == 'googlenet':
        if pretrained == True:
            model = model_fn(transform_input=False, num_classes = num_classes, weights = "DEFAULT")
        else:
            model = model_fn(transform_input=False, num_classes = num_classes)
    else:
        if pretrained == True:
            model = model_fn(num_classes = num_classes, weights = "DEFAULT")
        else:
            model = model_fn(num_classes = num_classes)
    return model.to(device)