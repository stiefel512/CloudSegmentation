import torch 
from torch import nn

from CloudSegmentation.modules.resnet_modules import ResidualBlock
from CloudSegmentation.modules.unet_modules import UpConvBlock


def _make_layer(in_channels, out_channels, stride, num_blocks):
    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    layers = []
    layers.append(ResidualBlock(in_channels, out_channels, stride, bias=False, downsample=downsample))
    for i in range(1, num_blocks):
        layers.append(ResidualBlock(out_channels, out_channels, 1))
    
    return nn.Sequential(*layers)


class ResNetUNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(ResNetUNet, self).__init__()

        self.down1 = _make_layer(num_channels, 64, stride=2, num_blocks=1)
        self.down2 = _make_layer(64, 128, 2, 3)
        self.down3 = _make_layer(128, 256, 2, 4)
        self.down4 = _make_layer(256, 512, 2, 6)
        self.down5 = _make_layer(512, 1024, 2, 3)

        self.up1 = UpConvBlock(1024, 512)
        self.up2 = UpConvBlock(512, 256)
        self.up3 = UpConvBlock(256, 128)
        self.up4 = UpConvBlock(128, 64)
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 2, 2),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1, 1, 0, bias=False)
        )

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        out = self.up5(x9)
        
        return out
    