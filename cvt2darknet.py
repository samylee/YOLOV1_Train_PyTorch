import numpy as np
import torch.nn as nn
from detect import model_init

print('load pytorch model ... ')
checkpoint_path = 'weights/yolov1_final.pth'
S, B, C = 7, 2, 20
model = model_init(checkpoint_path, S, B, C)

print('convert to darknet ... ')
with open('weights/yolov1-tiny-final.weights', 'wb') as f:
    np.asarray([0, 1, 0, 2560000], dtype=np.int32).tofile(f)

    for module in model.features.features:
        if isinstance(module[0], nn.Conv2d):
            conv_layer = module[0]
            if isinstance(module[1], nn.BatchNorm2d):
                bn_layer = module[1]
                # bn bias
                num_b = bn_layer.bias.numel()
                bn_b = bn_layer.bias.data.view(num_b).numpy()
                bn_b.tofile(f)
                # bn weights
                num_w = bn_layer.weight.numel()
                bn_w = bn_layer.weight.data.view(num_w).numpy()
                bn_w.tofile(f)
                # bn running mean
                num_rm = bn_layer.running_mean.numel()
                bn_rm = bn_layer.running_mean.data.view(num_rm).numpy()
                bn_rm.tofile(f)
                # bn running var
                num_rv = bn_layer.running_var.numel()
                bn_rv = bn_layer.running_var.data.view(num_rv).numpy()
                bn_rv.tofile(f)
            else:
                # conv bias
                num_b = conv_layer.bias.numel()
                conv_b = conv_layer.bias.data.view(num_b).numpy()
                conv_b.tofile(f)
            # conv weights
            num_w = conv_layer.weight.numel()
            conv_w = conv_layer.weight.data.view(num_w).numpy()
            conv_w.tofile(f)

    # addition module
    addition_module = model.additional
    conv_layer = addition_module[0]
    if isinstance(addition_module[1], nn.BatchNorm2d):
        bn_layer = addition_module[1]
        # bn bias
        num_b = bn_layer.bias.numel()
        bn_b = bn_layer.bias.data.view(num_b).numpy()
        bn_b.tofile(f)
        # bn weights
        num_w = bn_layer.weight.numel()
        bn_w = bn_layer.weight.data.view(num_w).numpy()
        bn_w.tofile(f)
        # bn running mean
        num_rm = bn_layer.running_mean.numel()
        bn_rm = bn_layer.running_mean.data.view(num_rm).numpy()
        bn_rm.tofile(f)
        # bn running var
        num_rv = bn_layer.running_var.numel()
        bn_rv = bn_layer.running_var.data.view(num_rv).numpy()
        bn_rv.tofile(f)
    else:
        # conv bias
        num_b = conv_layer.bias.numel()
        conv_b = conv_layer.bias.data.view(num_b).numpy()
        conv_b.tofile(f)
    # conv weights
    num_w = conv_layer.weight.numel()
    conv_w = conv_layer.weight.data.view(num_w).numpy()
    conv_w.tofile(f)

    # fc module
    fc_module = model.fc
    # fc bias
    num_b = fc_module.bias.numel()
    fc_b = fc_module.bias.data.view(num_b).numpy()
    fc_b.tofile(f)
    # fc weights
    num_w = fc_module.weight.numel()
    fc_w = fc_module.weight.data.view(num_w).numpy()
    fc_w.tofile(f)

print('done!')