import torch
import torch.nn as nn


class SubBandSplit(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, specs):
        """
        Split the input specs to 5 groups through the frequency axis.
        The frequency bins in each group is {(1~16), (17~32), (33~64), (65~128), (129~257)}.
        The number of sub-bands in groups is {8, 6, 6, 6, 6}
        :param specs: [batch_size, 2, T, F]
        :return: [batch_size, 2, T, f] * 5
        """
        groups_0 = specs[..., 0: 16]
        groups_1 = specs[..., 16: 32]
        groups_2 = specs[..., 32: 64]
        groups_3 = specs[..., 64: 128]
        groups_4 = specs[..., 128: 257]

        x = [groups_0, groups_1, groups_2, groups_3, groups_4]
        return x


class FeatureMerge(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 32)
        self.conv = nn.Conv2d(32, 16, 1, 1)
        self.act = nn.ELU()

    def forward(self, fb_feat, sb_feat):
        feat = torch.concat((fb_feat, sb_feat), dim=-1)
        feat = self.fc(feat)
        feat = self.act(feat)
        feat = self.conv(feat)  # [B, C, T, F_c]
        return feat


class FeatureSplit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 32, 1, 1)
        self.fc = nn.Linear(32, 64)
        self.act = nn.ELU()

    def forward(self, x):
        feat = self.conv(x)
        # feat = feat.permute(0, 3, 1, 2).contiguous()
        B, C, T, F = feat.shape
        feat = torch.reshape(feat, shape=(B * C, T, F))
        feat = self.act(self.fc(feat))
        feat = torch.reshape(feat, shape=(B, F, T, F * 2))
        return feat


class Mask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fb_mask, sb_mask, spec, amplitude):
        enhanced_spec = spec * fb_mask
        enhanced_spec = torch.abs(enhanced_spec)

        sb_mask = sb_mask.unsqueeze(1)
        enhanced_amplitude = amplitude * sb_mask
        phase = torch.angle(spec)
        enhanced_sig = (enhanced_spec + enhanced_amplitude) / 2 * torch.exp(1j * phase)
        return enhanced_sig