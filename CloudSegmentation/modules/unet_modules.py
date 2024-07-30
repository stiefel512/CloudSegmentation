import torch
from torch import nn
from CloudSegmentation.modules.utils import conv3x3


class DualConvBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super(DualConvBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class DownConvBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super(DownConvBlock, self).__init__()

        self.pool = nn.MaxPool2d((2, 2), (2, 2))
        self.conv = DualConvBlock(in_planes, planes)

    def forward(self, x):
        down = self.pool(x)
        out = self.conv(down)

        return out
    

class UpConvBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super(UpConvBlock, self).__init__()

        self.up_conv = nn.ConvTranspose2d(in_planes, planes, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
        self.conv = DualConvBlock(in_planes, planes)

    def forward(self, x, skip):
        up = self.up_conv(x)
        x = torch.cat([skip, up], dim=1)
        out = self.conv(x)
        return out


if __name__ == "__main__":

    x = torch.rand((1,16,16, 16))
    block = nn.ConvTranspose2d(16, 8, (3, 3), (2, 2), (1, 1), (1, 1), bias=False)
    out = block(x)
    ...