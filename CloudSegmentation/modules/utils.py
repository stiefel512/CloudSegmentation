from CloudSegmentation.modules.common import ConvBlock


def conv1x1(in_planes, planes):
    return ConvBlock(in_planes, planes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)


def conv3x3(in_planes, planes):
    return ConvBlock(in_planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

