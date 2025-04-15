import torch
import torch.nn as nn
import torch.nn.functional as F

def _conv(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False, groups=in_channel),
        nn.BatchNorm2d(in_channel),
        nn.LeakyReLU(),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.LeakyReLU(),
    )

class _up_sample(nn.Module):
    def __init__(self, skip_ch, out_ch):
        super().__init__()
        # self.skip_conv  = _conv(skip_ch, dim)   #!!
        # self.out_conv   = _conv(dim * 2, dim)#!!
        self.out_conv   = _conv(skip_ch, out_ch)
    
    def forward(self, feat, skip):
        up_size = [skip.size(2), skip.size(3)]
        feat = F.interpolate(feat, size=up_size, mode="bilinear", align_corners=True)
        # feat = torch.concat([feat, self.skip_conv(skip)], dim=1)
        feat = torch.concat([feat, skip], dim=1)

        return self.out_conv(feat)


class Decoder(nn.Module): # Interpolation & Conv
    def __init__(self, feat_ch=[320, 112, 40, 24, 16], dim=16):
        super(Decoder, self).__init__()
        self.dim = dim

        self.up1 = _up_sample(feat_ch[0]    + feat_ch[1], self.dim * 8)
        self.up2 = _up_sample(self.dim * 8  + feat_ch[2], self.dim * 4)
        self.up3 = _up_sample(self.dim * 4  + feat_ch[3], self.dim * 2)
        self.up4 = _up_sample(self.dim * 2  + feat_ch[4], self.dim * 1)
        # self.in_conv = _conv(feat_ch[0], self.dim)
        
        # self.up1 = _up_sample(feat_ch[1], self.dim)
        # self.up2 = _up_sample(feat_ch[2], self.dim)
        # self.up3 = _up_sample(feat_ch[3], self.dim)
        # self.up4 = _up_sample(feat_ch[4], self.dim)

    def forward(self, x, skip_buff=[]):
        # x = self.in_conv(x)
        x = self.up1(x, skip_buff[-1])
        x = self.up2(x, skip_buff[-2])
        x = self.up3(x, skip_buff[-3])
        x = self.up4(x, skip_buff[-4])

        return x

if __name__ == "__main__":
    from flops_profiler.profiler import get_model_profile

    B = 1

    class encap(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder_test = Decoder(feat_ch=[320, 112, 40, 24, 16])

        def forward(self, x):
            dummy = torch.rand([B, 320, 8, 10])
            f1 = torch.rand([B, 16, 114, 152])
            f2 = torch.rand([B, 24, 57, 76])
            f3 = torch.rand([B, 40, 29, 38])
            f4 = torch.rand([B, 112, 15, 19])
            flist = [f1, f2, f3, f4]

            output = self.decoder_test(dummy, flist)
            # print(output.size())
            pass
    
    test = encap()
    flops, macs, params = get_model_profile(model=test, input_shape=(1,1), print_profile=False)
    print(f"FLOPs: {flops/1000000000:.2f}G | MACS: {macs/1000000000:.2f}G | Params: {params/1000:.2f}K")
