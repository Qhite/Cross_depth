import torch
import torch.nn as nn
import torch.nn.functional as F
import net

class DepthNet(nn.Module):
    def __init__(self, depth_range=[0.01, 10],lidar_point=116, f_size=[320,8,10], feat_ch=[320, 112, 40, 24, 16], d_model=256, bin_size=256, dim=16):
        super(DepthNet, self).__init__()
        self.depth_range = depth_range
        self.encoder = net.Encoder(lidar_point=lidar_point)
        self.attent  = net.Attention_Block(f_size=f_size, d_model=d_model, bin_size=bin_size)
        self.decoder = net.Decoder(feat_ch=feat_ch, dim=dim)

        self.probability_map_header = nn.Sequential(
            nn.Conv2d(dim, bin_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bin_size),
            nn.LeakyReLU(),
        )

    def forward(self, image, lidar):
        B, C, H, W = image.size()

        # Encoder
        img_f, ld_f, buff = self.encoder(image, lidar)

        # Attention Stage
        feat, bins = self.attent(img_f, ld_f)

        # Decoder
        feat = self.decoder(feat, buff)

        # Depth Probability Map
        p_map = self.probability_map_header(feat)
        p_map = F.softmax(p_map, dim=1)

        # Depth Bin
        depth_scale = (self.depth_range[1] - self.depth_range[0])
        bins = bins / bins.sum(axis=1, keepdim=True)
        bins = depth_scale * bins + self.depth_range[0]

        # Bin Centers
        bin_width = F.pad(bins, (1,0), mode="constant", value=1e-3)
        bin_edge  = bin_width.cumsum(dim=1)
        centers = 0.5 * (bin_edge[:, :-1]+bin_edge[:, 1:])
        centers = centers.unsqueeze(2).unsqueeze(2)
        
        # Depth map
        predict = (p_map * centers).sum(axis=1, keepdim=True)
        predict = F.interpolate(predict, size=[H, W], mode="bilinear", align_corners=True)

        return predict, centers.squeeze()