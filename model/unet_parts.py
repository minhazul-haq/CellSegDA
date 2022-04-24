# courtesy: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv2d


# code courtesy: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/condconv/condconv_layers.py
class CondConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()

        # doing for num_experts = 3

        self.routing_fn = Linear(in_channels, 3)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, input):
        pooled_input = (input.mean(2)).mean(2)
        routing_weight = F.sigmoid(self.routing_fn(pooled_input))

        print(routing_weight)

        x = self.conv1(input)
        y = self.conv2(input)
        z = self.conv3(input)

        output = (routing_weight[0][0] * x) + (routing_weight[0][1] * y) + (routing_weight[0][2] * z)

        return output


class CondDeconv2D(nn.Module):
    def __init__(self, a, b, c, d):
        super().__init__()

        # doing for num_experts = 3

        self.routing_fn = Linear(a, 3)
        self.deconv1 = nn.ConvTranspose2d(a, b, c, stride=d)
        self.deconv2 = nn.ConvTranspose2d(a, b, c, stride=d)
        self.deconv3 = nn.ConvTranspose2d(a, b, c, stride=d)

    def forward(self, input):
        pooled_input = (input.mean(2)).mean(2)
        routing_weight = F.sigmoid(self.routing_fn(pooled_input))

        x = self.deconv1(input)
        y = self.deconv2(input)
        z = self.deconv3(input)

        output = (routing_weight[0][0] * x) + (routing_weight[0][1] * y) + (routing_weight[0][2] * z)

        return output


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
