# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: Henrique Weber
# LVSN, Universite Labal
# Email: henrique.weber.1@ulaval.ca
# Copyright (c) 2018
#
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from envmap import EnvironmentMap
from pytorch_toolbox.network_base import NetworkBase
from learning_indoor_lighting.tools.utils import check_nans


class AutoEncoderNet(NetworkBase):
    def __init__(self):
        super(AutoEncoderNet, self).__init__()

        # encoder
        self.en_conv1 = nn.Conv2d(3, 64, 9, stride=1, padding=1)  # 64 x 64 x 32
        self.en_conv1_bn = nn.BatchNorm2d(64)
        self.en_conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 128 x 32 x 16
        self.en_conv2_bn = nn.BatchNorm2d(128)
        self.res1 = self.make_layer(ResidualBlock, 128, 256, 2)
        self.res2 = self.make_layer(ResidualBlock, 256, 256, 2, 2)
        self.res3 = self.make_layer(ResidualBlock, 256, 256, 2, 2)
        self.res4 = self.make_layer(ResidualBlock, 256, 256, 2, 2)
        self.en_fc1 = nn.Linear(8192, 128)

        # decoder
        self.de_fc1 = nn.Linear(128, 8192)
        self.de_conv1 = UpsampleConvLayer(256, 256, kernel_size=3, stride=1, upsample=2)
        self.de_conv1_bn = nn.BatchNorm2d(256)
        self.de_conv2 = UpsampleConvLayer(256, 256, kernel_size=3, stride=1, upsample=2)
        self.de_conv2_bn = nn.BatchNorm2d(256)
        self.de_conv3 = UpsampleConvLayer(256, 128, kernel_size=3, stride=1, upsample=2)
        self.de_conv3_bn = nn.BatchNorm2d(128)
        self.de_conv4 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.de_conv4_bn = nn.BatchNorm2d(64)
        self.de_conv5 = UpsampleConvLayer(64, 3, kernel_size=3, stride=1)

        self.sa = EnvironmentMap(64, 'LatLong').solidAngles()
        self.view_size = [256, 4, 8]
        self.saved_input = []

    def encode(self, x):
        x = self.en_conv1_bn(F.elu(self.en_conv1(x)))
        x = self.en_conv2_bn(F.elu(self.en_conv2(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = x.view(-1, 8192)
        x = self.en_fc1(x)
        return x

    def decode(self, x):
        x = self.de_fc1(x)
        x = x.view(-1, self.view_size[0], self.view_size[1], self.view_size[2])
        x = self.de_conv1_bn(F.elu(self.de_conv1(x)))
        x = self.de_conv2_bn(F.elu(self.de_conv2(x)))
        x = self.de_conv3_bn(F.elu(self.de_conv3(x)))
        x = self.de_conv4_bn(F.elu(self.de_conv4(x)))
        x = self.de_conv5(x)
        return x

    @staticmethod
    def get_name_model():
        return 'model_best.pth.tar'

    @staticmethod
    def make_layer(block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        self.saved_input = x
        # encoder
        x = self.encode(x)
        # decoder
        x = self.decode(x)

        check_nans(x)

        return x

    def loss(self, predictions, targets):
        weights = Variable(torch.from_numpy(self.sa), requires_grad=False).float().cuda()
        # weighted L1
        wl1 = torch.sum(weights * torch.abs(predictions[0] - self.saved_input))
        return wl1


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    source : https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/transformer_net.py
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out
