import torch
import torch.nn as nn


class FullBandDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fb_dec_lists = nn.ModuleList([
            FullBandDeconvBlock(64, 16, (1, 6), (1, 2), padding=(0, 2)),
            FullBandDeconvBlock(32, 4, (1, 6), (1, 2), padding=(0, 2)),
            FullBandDeconvBlock(8, 2, (1, 9), (1, 2), padding=(0, 3))
        ])

    def forward(self, x, fb_res_list):
        n = len(self.fb_dec_lists)
        for i in range(n):
            x = self.fb_dec_lists[i](x, fb_res_list[n - 1 - i])
        return x


class FullBandDeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels // 2, 1, 1)
        self.deconv = nn.ConvTranspose2d(in_channels // 2, out_channels, kernel, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ELU()

    def forward(self, x, x_res):
        feat = torch.concat((x, x_res), dim=1)
        feat = self.conv(feat)
        feat = self.act(self.bn(self.deconv(feat)))
        return feat


class SubBandDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sb_dec_lists = nn.ModuleList([
            SubBandDecBlock(64, 2),
            SubBandDecBlock(64, 3),
            SubBandDecBlock(64, 6),
            SubBandDecBlock(64, 11),
            SubBandDecBlock(64, 20)
        ])

    def forward(self, x, sb_res_list):
        n = len(self.sb_dec_lists)
        idx_list = [0, 8, 14, 20, 26, 32]
        y = []
        for i in range(n):
            x_i = x[..., idx_list[i]: idx_list[i + 1]]
            y_i = self.sb_dec_lists[i](x_i, sb_res_list[i])
            y.append(y_i)

        output = []
        for i in range(len(y)):
            y_i = y[i]
            y_i = y_i.permute(0, 2, 1, 3).contiguous()
            B, T, M, N = y_i.shape
            y_i = torch.reshape(y_i, shape=(B, T, M * N))
            output.append(y_i)

        output = torch.cat(output, dim=-1)

        return output


class SubBandDecBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.fc = nn.Linear(in_channels, out_channels)
        self.act = nn.ReLU()

    def forward(self, x, sb_res):
        feat = torch.cat([x, sb_res], dim=1)
        feat = feat.permute(0, 3, 2, 1).contiguous()
        B, F, T, C = feat.shape
        feat = torch.reshape(feat, shape=(B * F, T, C))
        feat = self.act(self.fc(feat))
        feat = torch.reshape(feat, shape=(B, F, T, self.out_channels))
        return feat
