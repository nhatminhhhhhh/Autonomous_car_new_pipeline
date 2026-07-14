import torch
import torch.nn as nn

class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size,
                                    padding=padding, groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class SeparableDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.conv = nn.Sequential(
            SeparableConv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            SeparableConv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class RoadSegNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[8, 16, 32, 64]):
        super().__init__()
        ConvBlock = SeparableDoubleConv

        self.enc1 = ConvBlock(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(features[2], features[3])
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(features[3], features[3]*2)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = ConvBlock(features[3]*2 + features[3], features[2])
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = ConvBlock(features[2] + features[2], features[1])
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ConvBlock(features[1] + features[1], features[0])
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ConvBlock(features[0] + features[0], features[0])

        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d4 = self.dropout(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d3 = self.dropout(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d2 = self.dropout(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return out
