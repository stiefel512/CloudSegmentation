import torch
from torch import nn


class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=None):
        super(CBR, self).__init__()
        padding = kernel_size//2 if padding is None else padding
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.b = nn.BatchNorm2d(out_channels)
        self.r = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.r(self.b(self.c(x)))
    

class CR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=None):
        super(CR, self).__init__()
        padding = kernel_size//2 if padding is None else padding
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.r = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.r(self.c(x))


class MFF_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MFF_Encoder, self).__init__()
        self.conv1 = CBR(in_channels=in_channels, out_channels=out_channels, kernel_size=3)

        self.conv2_1 = CR(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        self.conv2_2 = CR(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        self.conv2_3 = nn.Sequential(
            CR(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            CR(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        )

        self.conv3 = CBR(in_channels=out_channels*3, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        residual = x1

        x2_1 = self.conv2_1(x1)
        x2_2 = self.conv2_2(x1)
        x2_3 = self.conv2_3(x1)
        
        x2 = torch.cat([x2_1, x2_2, x2_3], dim=1)
        x3 = self.conv3(x2)

        out = x3 + residual
        return out, x3  # the output of conv3, before being added to the residual, is a skip connection to the decoder


class MFF_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MFF_Decoder, self).__init__()
        self.conv1 = CBR(in_channels=in_channels, out_channels=out_channels, kernel_size=3)

        self.conv2_1 = CR(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        self.conv2_2 = CR(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        self.conv2_3 = nn.Sequential(
            CR(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            CR(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        )

        self.conv3 = CBR(in_channels=out_channels*3, out_channels=out_channels, kernel_size=1)

    def forward(self, x, skip):
        x0 = torch.cat([x, skip], dim=1)
        x1 = self.conv1(x0)
        residual = x1

        x2_1 = self.conv2_1(x1)
        x2_2 = self.conv2_2(x1)
        x2_3 = self.conv2_3(x1)
        x2 = torch.cat([x2_1, x2_2, x2_3], dim=1)

        x3 = self.conv3(x2)
        out = x3 + residual

        return out
    

class MFF_Dilated(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2):
        super(MFF_Dilated, self).__init__()
        self.conv1 = CBR(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation=dilation, padding=2)

        self.conv2_1 = CR(in_channels=out_channels, out_channels=out_channels, kernel_size=1, dilation=dilation, padding=0)
        self.conv2_2 = CR(in_channels=out_channels, out_channels=out_channels, kernel_size=3, dilation=dilation, padding=2)
        self.conv2_3 = nn.Sequential(
            CR(in_channels=out_channels, out_channels=out_channels, kernel_size=3, dilation=dilation, padding=2),
            CR(in_channels=out_channels, out_channels=out_channels, kernel_size=3, dilation=dilation, padding=2)
        )

        self.conv3 = CBR(in_channels=out_channels*3, out_channels=out_channels, kernel_size=1, dilation=dilation, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        residual = x1

        x2_1 = self.conv2_1(x1)
        x2_2 = self.conv2_2(x1)
        x2_3 = self.conv2_3(x1)
        x2 = torch.cat([x2_1, x2_2, x2_3], dim=1)

        x3 = self.conv3(x2)
        out = x3 + residual

        return out
    

class CDSFF(nn.Module):
    def __init__(self, in_channels, num_classes, kernel_size, stride, padding):
        super(CDSFF, self).__init__()
        self.deco = nn.ConvTranspose2d(in_channels=in_channels, out_channels=2, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        self.conv = nn.Sequential(
            nn.Conv2d(2, num_classes, 1, 1, 0, bias=False),
            # nn.Softmax2d()
        )
    
    def forward(self, x):
        skip = self.deco(x)
        out = self.conv(skip)

        return out, skip
    

class MainOutput(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MainOutput, self).__init__()
        self.conv1 = CBR(in_channels, 2, 3)
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, num_classes, 1, 1, 0, bias=False),
            # nn.Softmax2d()
        )

    def forward(self, x, skip1, skip2, skip3):
        x1 = self.conv1(x)
        x1 = torch.cat([skip1, skip2, skip3, x1], dim=1)
        out = self.conv2(x1)
        return out
