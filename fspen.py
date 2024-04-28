"""
FSPEN: sub-band encoder + full-band encoder + DPE + sub-band decoder + full-band decoder
"""
import torch
import numpy as np
import torch.nn as nn
from fspen_modules.encoder import FullBandEncoder, SubBandEncoder
from fspen_modules.decoder import FullBandDecoder, SubBandDecoder
from fspen_modules.feature import SubBandSplit, FeatureSplit, FeatureMerge, Mask
from fspen_modules.dpe import DPE


class FSPEN(nn.Module):
    def __init__(self, groups):
        super(FSPEN, self).__init__()
        self.groups = groups

        self.sb_split = SubBandSplit()
        self.fb_encoder = FullBandEncoder()
        self.sb_encoder = SubBandEncoder()
        self.fm = FeatureMerge()

        self.dpe1 = DPE(16, 32, 16, self.groups)
        self.dpe2 = DPE(16, 32, 16, self.groups)
        self.dpe3 = DPE(16, 32, 16, self.groups)

        self.fs = FeatureSplit()
        self.fb_decoder = FullBandDecoder()
        self.sb_decoder = SubBandDecoder()

        self.mask_padding = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=0.0)

        self.mask = Mask()

    def forward(self, spec):
        amp = torch.abs(spec)
        spec = torch.view_as_real(spec)
        spec = spec.permute(0, 3, 2, 1).contiguous()
        spec_ref = spec
        amp = amp.permute(0, 2, 1).contiguous()
        amp = amp.unsqueeze(1)
        fb_feat, fb_en_outs = self.fb_encoder(spec)
        sb_feat = self.sb_split(amp)
        sb_feat, sb_en_outs = self.sb_encoder(sb_feat)
        feat = self.fm(fb_feat, sb_feat)

        B, C, T, F_c = feat.shape

        hidden_state = torch.zeros(self.groups, 1, B * F_c // self.groups, 16, device=feat.device)
        feat, hidden_state = self.dpe1(feat, hidden_state)
        feat, hidden_state = self.dpe2(feat, hidden_state)
        feat, hidden_state = self.dpe3(feat, hidden_state)
        del hidden_state
        feat = self.fs(feat)
        C = feat.shape[-1]
        fb_feat, sb_feat = feat[..., :C // 2], feat[..., C // 2:]

        fb_mask = self.fb_decoder(fb_feat, fb_en_outs)
        sb_mask = self.sb_decoder(sb_feat, sb_en_outs)
        sb_mask = self.mask_padding(sb_mask)

        enhanced_spec = self.mask(fb_mask, sb_mask, spec_ref, amp)

        return enhanced_spec


if __name__ == "__main__":
    from thop import profile, clever_format
    import soundfile as sf

    x = torch.randn(1, 16000)
    spec = torch.stft(x, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=True)
    print(x.shape)
    model = FSPEN(groups=8).eval()
    y = model(spec)
    print("output", y.shape)

    flops, params = profile(model, inputs=(spec,))
    flops, params = clever_format(nums=[flops, params], format="%0.4f")
    print(f"flops: {flops} \nparams: {params}")
