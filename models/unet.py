import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.seq(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            DoubleConv(in_channels, 64),
            nn.MaxPool2d(2),
            DoubleConv(64, 128),
            nn.MaxPool2d(2)
        )
        self.bottleneck = DoubleConv(128, 256)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            DoubleConv(128, 128),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            DoubleConv(64, 64),
        )
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return self.final(x)
