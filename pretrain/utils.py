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