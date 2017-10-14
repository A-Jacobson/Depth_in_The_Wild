import torch
from torch import nn
import torch.nn.functional as F


class ConvReluBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(ConvReluBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, True)
        return self.bn(x)


class InceptionS(nn.Module):
    """
    1x1, 3x3, 5x5, 7x7 size filters
    """

    def __init__(self, in_channels, out_channels):
        super(InceptionS, self).__init__()
        # same padding = (kernel size - 1) / 2
        out_channels = int(out_channels / 4)  # filters per branch

        self.conv1x1 = ConvReluBN(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = ConvReluBN(
            out_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = ConvReluBN(
            out_channels, out_channels, kernel_size=5, padding=2)
        self.conv7x7 = ConvReluBN(
            out_channels, out_channels, kernel_size=7, padding=3)

    def forward(self, x):
        branch1x1 = self.conv1x1(x)

        branch3x3 = self.conv1x1(x)
        branch3x3 = self.conv3x3(branch3x3)

        branch5x5 = self.conv1x1(x)
        branch5x5 = self.conv5x5(branch5x5)

        branch7x7 = self.conv1x1(x)
        branch7x7 = self.conv7x7(branch7x7)
        return torch.cat([branch1x1, branch3x3, branch5x5, branch7x7], dim=1)


class InceptionL(nn.Module):
    """
    1x1, 3x3, 7x7, 11x11 size filters
    """

    def __init__(self, in_channels, out_channels):
        super(InceptionL, self).__init__()
        out_channels = int(out_channels / 4)
        self.conv1x1 = ConvReluBN(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = ConvReluBN(
            out_channels, out_channels, kernel_size=3, padding=1)
        self.conv7x7 = ConvReluBN(
            out_channels, out_channels, kernel_size=7, padding=3)
        self.conv11x11 = ConvReluBN(
            out_channels, out_channels, kernel_size=11, padding=5)

    def forward(self, x):
        branch1x1 = self.conv1x1(x)

        branch3x3 = self.conv1x1(x)
        branch3x3 = self.conv3x3(branch3x3)

        branch7x7 = self.conv1x1(x)
        branch7x7 = self.conv7x7(branch7x7)

        branch11x11 = self.conv1x1(x)
        branch11x11 = self.conv11x11(branch11x11)
        return torch.cat([branch1x1, branch3x3, branch7x7, branch11x11], dim=1)


class HourGlass(nn.Module):
    def __init__(self):
        super(HourGlass, self).__init__()
        self.batch_norm = nn.BatchNorm2d(3)
        self.avg_pool = nn.AvgPool2d(2)
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.A = InceptionL(128, 64)
        self.B = InceptionS(128, 128)
        self.C = InceptionL(128, 128)
        self.D = InceptionS(128, 256)
        self.E = InceptionS(256, 256)
        self.F = InceptionL(256, 256)
        self.G = InceptionS(256, 128)
        self.H = ConvReluBN(3, 128, 7, padding=3)
        self.I = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        x = self.batch_norm(x) # hack to normalize input (paper didn't mention mean centering?)
        x = self.H(x)  # 128, 240, 320
        channel_1 = self.A(x)  # 64, 240, 320

        # start c2
        channel_2 = self.avg_pool(x)  # 128, 120, 160
        channel_2 = self.B(channel_2)  # 128, 120, 160
        channel_2 = self.B(channel_2)  # 128, 120, 160

        # start c3
        channel_3 = self.avg_pool(channel_2)  # 128, 60, 80
        channel_3 = self.B(channel_3)  # 128, 60, 80
        channel_3 = self.D(channel_3)  # 256, 60, 80

        # start c4
        channel_4 = self.max_pool(channel_3)  # 256, 30, 40
        channel_4 = self.E(channel_4)  # 256, 30, 40
        channel_4 = self.E(channel_4)  # 256, 30, 40

        # start c5
        channel_5 = self.max_pool(channel_4)  # 256, 15, 20
        channel_5 = self.E(channel_5)  # 256, 15, 20
        channel_5 = self.E(channel_5)  # 256, 15, 20
        channel_5 = self.E(channel_5)  # 256, 15, 20
        channel_5 = self.upsample(channel_5)  # 256, 30, 40

        # 2nd half c4
        channel_4 = self.E(channel_4)  # 256, 30, 40
        channel_4 = self.E(channel_4)  # 256, 30, 40
        channel_4 = channel_4 + channel_5  # # 256, 30, 40
        channel_4 = self.E(channel_4)  # 256, 30, 40 processing on combined 4-5 features
        channel_4 = self.F(channel_4)  # 256, 30, 40
        channel_4 = self.upsample(channel_4)  # 256, 60, 80

        channel_3 = self.E(channel_3)  # 256, 60, 80
        channel_3 = self.F(channel_3)  # 256, 60, 80
        channel_3 = channel_4 + channel_3  # 256, 60, 80
        channel_3 = self.E(channel_3)  # 256, 60, 80
        channel_3 = self.G(channel_3)  # 128, 60, 80
        channel_3 = self.upsample(channel_3)  # 128, 120, 160

        channel_2 = self.B(channel_2)  # 128, 120, 160
        channel_2 = self.C(channel_2)  # 128, 120, 160
        channel_2 = channel_3 + channel_2  # 128, 120, 160
        channel_2 = self.B(channel_2)  # 128, 120, 160
        channel_2 = self.A(channel_2)  # 64, 120, 160
        channel_2 = self.upsample(channel_2)  # 64, 240, 320

        channel_1 = channel_2 + channel_1  # 64, 240, 320

        return self.I(channel_1)  # (1, 240, 320)
