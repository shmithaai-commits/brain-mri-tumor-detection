import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    """
    U-Net: The AI model that draws a mask around the tumor.
    Architecture: Encoder → Bottleneck → Decoder with skip connections.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        # Encoder
        ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv(ch, f))
            ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Decoder
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(f*2, f, 2, 2))
            self.decoders.append(DoubleConv(f*2, f))

        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i, (up, dec) in enumerate(zip(self.upconvs, self.decoders)):
            x = up(x)
            x = torch.cat([skips[i], x], dim=1)
            x = dec(x)

        return self.final(x)
