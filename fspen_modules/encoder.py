import torch
import torch.nn as nn


class FullBandEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.full_en_convs = nn.ModuleList([
            FullBandConvBlock(2, 4, kernel_size=(1, 6), stride=(1, 2), padding=(0, 2), use_deconv=False),
            FullBandConvBlock(4, 16, kernel_size=(1, 8), stride=(1, 2), padding=(0, 3), use_deconv=False),
            FullBandConvBlock(16, 32, kernel_size=(1, 6), stride=(1, 2), padding=(0, 2), use_deconv=False),
            nn.Conv2d(32, 32, (1, 1), (1, 1))
        ])

    def forward(self, x):
        en_outs = []
        for i in range(len(self.full_en_convs)):
            x = self.full_en_convs[i](x)
            en_outs.append(x)
        return x, en_outs[0: 3]


class SubBandEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub_en_convs = nn.ModuleList([
            SubBandConvBlock(2, 32, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            SubBandConvBlock(2, 32, kernel_size=(1, 7), stride=(1, 3), padding=(0, 3)),
            SubBandConvBlock(2, 32, kernel_size=(1, 11), stride=(1, 5), padding=(0, 2)),
            SubBandConvBlock(2, 32, kernel_size=(1, 20), stride=(1, 10), padding=(0, 4)),
            SubBandConvBlock(2, 32, kernel_size=(1, 40), stride=(1, 20), padding=(0, 7))
        ])

    def forward(self, x):
        en_outs = []
        for i in range(len(self.sub_en_convs)):
            en_outs_i = self.sub_en_convs[i](x[i])
            en_outs.append(en_outs_i)
        output = torch.concat(en_outs, dim=-1)
        return output, en_outs


class FullBandConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_deconv):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SubBandConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x))
