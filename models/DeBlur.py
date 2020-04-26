import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from models.util import ResBlock, conv, deconv, crop_like

__all__ = [
    'deblurnetde', 'deblurnetde_bn'
]

class DeBlurNetDE(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(DeBlurNetDE, self).__init__()

        n_resblocks = 15
        n_feats_a = 64
        n_feats_b = 128
        kernel_size = 3
        defocus_in = 3
        defocus_out = 1
        deblur_in = 4
        deblur_out = 3

        self.conv01a = conv(defocus_in, n_feats_a, kernel_size)
        self.convRes01a = ResBlock(n_feats_a, n_feats_a, kernel_size)
        self.convRes02a = ResBlock(n_feats_a, n_feats_a, kernel_size)
        self.convRes03a = ResBlock(n_feats_a, n_feats_a, kernel_size)
        self.convRes04a = ResBlock(n_feats_a, n_feats_a, kernel_size)
        self.conv02a = conv(n_feats_a, n_feats_a, kernel_size)
        self.conv03a = conv(n_feats_a, defocus_out, kernel_size)

        for p in self.parameters():
            p.requires_grad=False

        self.conv01b = conv(deblur_in, n_feats_b, kernel_size)
        self.convRes01b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes02b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes03b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes04b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes05b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes06b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes07b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes08b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes09b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes10b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes11b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes12b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes13b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes14b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes15b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.convRes16b = ResBlock(n_feats_b, n_feats_b, kernel_size)
        self.conv02b = conv(n_feats_b, n_feats_b, kernel_size)
        self.conv03b = conv(n_feats_b, deblur_out, kernel_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)


    def forward(self, input_image, target_deblur=None, target_defest=None):

        out_conv01a = self.conv01a(input_image)
        out_convRes01a = out_conv01a + self.convRes01a(out_conv01a)
        out_convRes02a = out_convRes01a + self.convRes02a(out_convRes01a)
        out_convRes03a = out_convRes02a + self.convRes03a(out_convRes02a)
        out_convRes04a = out_convRes03a + self.convRes04a(out_convRes03a)
        out_conv02a = self.conv02a(out_convRes04a)
        out_conv03a = self.conv03a(out_conv02a)

        out_conv01b = self.conv01b(torch.cat((input_image, out_conv03a), 1))
        out_convRes01b = out_conv01b + self.convRes01b(out_conv01b)
        out_convRes02b = out_convRes01b + self.convRes02b(out_convRes01b)
        out_convRes03b = out_convRes02b + self.convRes03b(out_convRes02b)
        out_convRes04b = out_convRes03b + self.convRes04b(out_convRes03b)
        out_convRes05b = out_convRes04b + 0.1 * self.convRes05b(out_convRes04b)
        out_convRes06b = out_convRes05b + 0.1 * self.convRes06b(out_convRes05b)
        out_convRes07b = out_convRes06b + 0.1 * self.convRes07b(out_convRes06b)
        out_convRes08b = out_convRes07b + 0.1 * self.convRes08b(out_convRes07b)
        out_convRes09b = out_convRes08b + 0.1 * self.convRes09b(out_convRes08b)
        out_convRes10b = out_convRes09b + 0.1 * self.convRes10b(out_convRes09b)
        out_convRes11b = out_convRes10b + 0.1 * self.convRes11b(out_convRes10b)
        out_convRes12b = out_convRes11b + 0.1 * self.convRes12b(out_convRes11b)
        out_convRes13b = out_convRes12b + 0.1 * self.convRes13b(out_convRes12b)
        out_convRes14b = out_convRes13b + 0.1 * self.convRes14b(out_convRes13b)
        out_convRes15b = out_convRes14b + 0.1 * self.convRes15b(out_convRes14b)
        out_convRes16b = out_convRes15b + 0.1 * self.convRes16b(out_convRes15b)
        out_conv02b = self.conv02b(out_convRes16b)
        out_conv03b = input_image + self.conv03b(out_conv02b)

        output = torch.cat((out_conv03a, out_conv03b), 1)

        return output

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def deblurnetde(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = DeBlurNetDE(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def deblurnetde_bn(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = DeBlurNetDE(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
