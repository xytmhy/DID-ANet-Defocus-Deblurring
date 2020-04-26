import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size//2), bias=bias)

def ResBlock(in_channels, out_channels, kernel_size, bias=True, batchNorm=False):
    if batchNorm:
        return nn.Sequential(
            conv(in_channels, out_channels, kernel_size, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            conv(in_channels, out_channels, kernel_size, bias=True),
            nn.BatchNorm2d(out_channels)
        )
    else:
        return nn.Sequential(
            conv(in_channels, out_channels, kernel_size, bias=True),
            nn.ReLU(True),
            conv(in_channels, out_channels, kernel_size, bias=True)
        )


# def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
#     if batchNorm:
#         return nn.Sequential(
#             nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
#             nn.BatchNorm2d(out_planes),
#             nn.LeakyReLU(0.1,inplace=True)
#         )
#     else:
#         return nn.Sequential(
#             nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
#             nn.LeakyReLU(0.1,inplace=True)
#         )

def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 1, kernel_size=3, stride=1, padding=1, bias=False)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1, inplace=True)
    )

def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]
