import torch
from torch import nn

from CloudSegmentation.modules.common import ConvBlock


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bias=False, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, (3, 3), stride=stride, padding=1, bias=bias)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (3, 3), 1, 1, bias=bias),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=False)

        self.downsample = downsample

    def forward(self, x):
        residual = x
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        if self.downsample:
            residual = self.downsample(residual)
        x3 = x2 + residual
        out = self.relu(x3)
        return out