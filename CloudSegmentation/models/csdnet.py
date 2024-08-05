import torch
from torch import nn
from CloudSegmentation.modules.csdnet_modules import MFF_Encoder, MFF_Dilated, MFF_Decoder, CDSFF, MainOutput


class CSDNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CSDNet, self).__init__()

        self.enc1 = MFF_Encoder(in_channels, 32)
        self.pool1 = nn.MaxPool2d((2, 2), (2, 2))
        self.enc2 = MFF_Encoder(32, 64)
        self.pool2 = nn.MaxPool2d((2, 2), (2, 2))
        self.enc3 = MFF_Encoder(64, 128)
        self.pool3 = nn.MaxPool2d((2, 2), (2, 2))
        self.enc4 = MFF_Dilated(128, 256)

        self.dec1 = MFF_Dilated(256, 256)
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.dec2 = MFF_Decoder(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.dec3 = MFF_Decoder(128, 64)
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.dec4 = MFF_Decoder(64, 32)

        self.cdsff1 = CDSFF(256, num_classes, 16, 8, 4)
        self.cdsff2 = CDSFF(128, num_classes, 8, 4, 2)
        self.cdsff3 = CDSFF(64, num_classes, 4, 2, 1)

        self.output = MainOutput(32, num_classes)

    def forward(self, x):
        enc1, skip1 = self.enc1(x)
        enc1 = self.pool1(enc1)
        enc2, skip2 = self.enc2(enc1)
        enc2 = self.pool2(enc2)
        enc3, skip3 = self.enc3(enc2)
        enc3 = self.pool3(enc3)
        enc4 = self.enc4(enc3)

        dec1 = self.dec1(enc4)
        dec2 = self.upconv1(dec1)
        dec2 = self.dec2(dec2, skip3)
        dec3 = self.upconv2(dec2)
        dec3 = self.dec3(dec3, skip2)
        dec4 = self.upconv3(dec3)
        dec4 = self.dec4(dec4, skip1)
        
        out1, oskip1 = self.cdsff1(dec1)
        out2, oskip2 = self.cdsff2(dec2)
        out3, oskip3 = self.cdsff3(dec3)

        out = self.output(dec4, oskip1, oskip2, oskip3)
        return out, out1, out2, out3


