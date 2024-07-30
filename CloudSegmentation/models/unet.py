import torch
from torch import nn
from CloudSegmentation.modules.unet_modules import DualConvBlock, DownConvBlock, UpConvBlock
from CloudSegmentation.modules.utils import conv1x1


class UNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(UNet, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        self.in_conv = DualConvBlock(num_channels, 64)
        self.down1 = DownConvBlock(64, 128)
        self.down2 = DownConvBlock(128, 256)
        self.down3 = DownConvBlock(256, 512)
        self.down4 = DownConvBlock(512, 1024)

        self.up4 = UpConvBlock(1024, 512)
        self.up3 = UpConvBlock(512, 256)
        self.up2 = UpConvBlock(256, 128)
        self.up1 = UpConvBlock(128, 64)

        self.out_conv = conv1x1(64, num_classes)

    def forward(self, x):
        x1 = self.in_conv(x)
        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = self.up4(x5, x4)
        x7 = self.up3(x6, x3)
        x8 = self.up2(x7, x2)
        x9 = self.up1(x8, x1)

        out = self.out_conv(x9)
        return out