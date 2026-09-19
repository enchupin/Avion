import torch
import torch.nn as nn

def conv_block(in_channels, out_channels, kernel_size, padding=0, activation=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)]
    if activation:
        layers.append(nn.PReLU(out_channels))
    return nn.Sequential(*layers)

def deconv_block(in_channels, out_channels, scale_factor=4):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=9, stride=scale_factor,
        padding=4, output_padding=scale_factor - 1
    )
