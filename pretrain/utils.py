import numpy as np
import torch

def get_model_config(dataset):
    config = dict()

    if dataset == "MNIST":
        input_shape = 28
        resize_shape = 28
    elif dataset == "CIFAR10":
        input_shape = 32
        resize_shape = 32

    config.update({'resize_shape': resize_shape, 'center_crop_shape': input_shape})
    return config

def find_indices(model):
    indices = []
    construct_flag = []

    for index, var_name in enumerate(model.state_dict()):
        if len(model.state_dict()[var_name].size()) == 4:
            indices.append(index)
            construct_flag.append(var_name)

    return indices, construct_flag

def check_channel(tensor):
    # tensor must be a conv layer
    size_0 = tensor.size()[0]
    size_1 = tensor.size()[1] * tensor.size()[2] * tensor.size()[3]
    tensor_resize = tensor.view(size_0, -1)

    channel_if_zero = np.zeros(size_0) # vector that keeps the indices of the zero channels

    for x in range(size_0):
        # if all zeros, channel_if_zero = 0
        channel_if_zero[x] = np.count_nonzero(tensor_resize[x].cpu().numpy()) != 0

    indices_nonzero = torch.LongTensor((channel_if_zero != 0).nonzero()[0])
    zeros = (channel_if_zero == 0).nonzero()[0]
    indices_zero = torch.LongTensor(zeros)

    return indices_zero, indices_nonzero

def extract_para(small_model):
    item = list(small_model.state_dict().items())

    kept_index_per_layer = {}
    kept_filter_per_layer = {}
    pruned_index_per_layer = {}
    before_pruning_filters_per_layer = {}

    indices, construct_flag = find_indices(small_model)
    
    for x in indices:
        indices_zero, indices_nonzero = check_channel(item[x][1])
        pruned_index_per_layer[item[x][0]] = indices_zero
        kept_index_per_layer[item[x][0]] = indices_nonzero
        kept_filter_per_layer[item[x][0]] = indices_nonzero.shape[0]
    
    # number of non-zero channels
    num_for_construct = []
    for key in construct_flag:
        num_for_construct.append(kept_filter_per_layer[key])
    
    # which indices of each layer are being kept
    index_for_construct = dict(
        (key, value) for (key, value) in kept_index_per_layer.items()
    )

    return kept_index_per_layer, pruned_index_per_layer

def get_pruned_model(model, model_builder, num_classes, in_channels, prune_rate, device):
    kept_index_per_layer, pruned_index_per_layer = extract_para(model)
    small_model = model_builder(num_classes = num_classes, in_channels = in_channels, prune_rate = prune_rate).to(device)

    big_state_dict = model.cpu().state_dict()
    keys_list = list(kept_index_per_layer.keys())
    small_state_dict = {}
    for index, [key, value] in enumerate(big_state_dict.items()):
        new_value = value
        if 'bn' in key:
            if 'weight' in key or 'bias' in key or 'running_mean' in key or 'running_var' in key:
                bn_split = key.split('.')
                bn_split[-1] = 'weight'
                conv_key = '.'.join(bn_split)
                conv_key = conv_key.replace('bn', 'conv')
                new_value = torch.index_select(new_value, 0, kept_index_per_layer[conv_key])
        elif 'conv' in key:
            if key in kept_index_per_layer.keys():
                new_value = torch.index_select(new_value, 0, kept_index_per_layer[key])
                indices = kept_index_per_layer[key]

            if '1' not in key:
                if key in keys_list:
                    conv_index = keys_list.index(key)
                    key_for_input = keys_list[conv_index - 1]
                    new_value = torch.index_select(new_value, 1, kept_index_per_layer[key_for_input])
        small_state_dict[key] = new_value

    small_model.load_state_dict(small_state_dict)
    return small_model