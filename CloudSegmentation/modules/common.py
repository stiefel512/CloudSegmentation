import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_planes, planes, kernel_size, stride, padding, bias=False):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    

if __name__ == "__main__":
    test = torch.rand((2,3,16,16))
    block = ConvBlock(3, 16, (3, 3), (1, 1), (1, 1))
    out = block(test)
    ...