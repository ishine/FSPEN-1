"""
FSPEN: sub-band encoder + full-band encoder + DPE + sub-band decoder + full-band decoder
Ultra tiny, 39.63 MMACs, 23.67 K params
"""
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
from fspen_modules.encoder import FullBandEncoder, SubBandEncoder
from fspen_modules.decoder import FullBandDecoder, SubBandDecoder
from fspen_modules.feature import SubBandSplit, FeatureSplit, FeatureMerge, Mask
from fspen_modules.dpe import DPE


class FSPEN(nn.Module):
    def __init__(self):
        super(FSPEN, self).__init__()
        self.sb_split = SubBandSplit()
        self.fb_encoder = FullBandEncoder()
        self.sb_encoder = SubBandEncoder()
        self.fm = FeatureMerge()

        self.dpe1 = DPE(16, 32, 16)
        self.dpe2 = DPE(16, 32, 16)
        self.dpe3 = DPE(16, 32, 16)

        self.fs = FeatureSplit()
        self.fb_decoder = FullBandDecoder()
        self.sb_decoder = SubBandDecoder()

        self.mask_padding = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=0.0)

        self.mask = Mask()

    def forward(self, spec):
        spec_ref = spec  # [16, 2, F, T]
        amp = torch.abs(spec)
        fb_feat, fb_en_outs = self.fb_encoder(spec)  # [B, 32, T, 16]

        sb_feat = self.sb_split(amp)
        sb_feat, sb_en_outs = self.sb_encoder(sb_feat)

        feat = self.fm(fb_feat, sb_feat)  # [B, C, T, F_c]
        feat = self.dpe1(feat)
        feat = self.dpe2(feat)
        feat = self.dpe3(feat)
        feat = self.fs(feat)  # [B, T, F, C]
        C = feat.shape[-1]
        fb_feat, sb_feat = feat[..., 0: C // 2], feat[..., C // 2:]  # [B, C, T, F]

        fb_mask = self.fb_decoder(fb_feat, fb_en_outs)

        sb_mask = self.sb_decoder(sb_feat, sb_en_outs)
        sb_mask = self.mask_padding(sb_mask)

        enhanced_spec = self.mask(fb_mask, sb_mask, spec_ref, amp)

        return enhanced_spec


if __name__ == "__main__":
    from ptflops import get_model_complexity_info
    import soundfile as sf

    x = torch.randn(8, 2, 251, 257)
    print(x.shape)
    print("========")
    model = FSPEN().eval()
    y = model(x)
    print(y.shape)

    flops, params = get_model_complexity_info(model, (2, 63, 257), as_strings=True,
                                              print_per_layer_stat=False, verbose=True)
    print(flops, params)

    wav, sr = sf.read('./test.wav')
    wav = torch.from_numpy(wav).float()
    noisy = torch.stft(wav[80000: 160000], 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=True)
    noisy = torch.view_as_real(noisy)
    noisy = noisy.unsqueeze(0)
    noisy = noisy.permute(0, 3, 2, 1)
    print(noisy.shape)
    enhanced = model(noisy)
    print("enhanced ", enhanced.shape)
    enhanced = enhanced.permute(0, 3, 2, 1)
    print(enhanced.shape)
    enhanced_r = enhanced[..., 0].float()
    enhanced_i = enhanced[..., 1].float()
    enhanced = torch.complex(enhanced_r, enhanced_i)
    print(enhanced.shape)

    enhanced = torch.istft(enhanced, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    sf.write("out.wav", enhanced.detach().cpu().numpy()[0], samplerate=16000)
