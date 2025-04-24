import torch
import net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = net.DepthNet(lidar_point=60, bin_size=64).to(device)
model.load_state_dict(torch.load("./pre_trained/bin_64.pth.tar", weights_only=True))
model.eval()

import dataloader
import torch.functional as F
from tqdm import tqdm
from math import pi
import matplotlib.pyplot as plt

PI = torch.tensor(pi)

def to_pcl(depth):
    B, _, H, W = depth.size()

    fx, fy = 582.62448167737955, 582.69103270988637
    cx, cy = 313.04475870804731, 238.44389626620386

    scale_x, scale_y = H / 640, W / 480

    fx *= scale_x
    fy *= scale_y
    cx *= scale_x
    cy *= scale_y

    u = torch.arange(0, W, device=device).view(1, -1).expand(H, -1)
    v = torch.arange(0, H, device=device).view(-1, 1).expand(-1, W)
    u = u.unsqueeze(0).expand(B, -1, -1)
    v = v.unsqueeze(0).expand(B, -1, -1)

    z = depth.squeeze(1).to(device)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return x, y, z

class pseudo_lidar():
    def __init__(self, lidar_points=116):
        self.lp = lidar_points
        self.min_fov  = -22.2
        self.max_fov  =  11.6
        self.fov_step = 0.325
        self.max_step = int((self.max_fov - self.min_fov) // self.fov_step) + 1 # 104
    
    def __call__(self, depth, epoch):
        B, _, H, W = depth.size()

        x, y, z = to_pcl(depth=depth)
        d = torch.sqrt(x**2 + z**2)
        theta = torch.atan2(y, d) / PI * 180
        
        top     = int(50)
        bottom  = int(51)
        sel = torch.randint(top, bottom, (B,1,1), device=device)

        mask_up = torch.ge(theta, self.min_fov + 50 * self.fov_step)
        mask_lo = torch.le(theta, self.min_fov + 51 * self.fov_step)

        mask = torch.logical_and(mask_lo, mask_up).unsqueeze(1)

        lidar = depth.to(device) * mask

        mask_hit = mask.sum(dim=2)

        lidar = lidar.sum(dim=2)

        lidar = lidar / mask_hit

        indices = torch.linspace(0, lidar.size(2)-1, self.lp).int()

        lidar = lidar[:, :, indices]

        return lidar

def cal_metric(predict, target):
    assert predict.size(0) == 1, "Batch size must be 1 for validation metric calculation."
    if predict.size(3) != target.size(3):
        _, _ , H, W = target.size()   
        predict = F.interpolate(predict, size=[H, W], mode="bilinear", align_corners=True)

    mask_p = torch.ge(predict, 0.01)
    mask_t = torch.ge(target, 0.01)
    mask = torch.logical_and(mask_p, mask_t)

    p = predict[mask]
    t = target[mask]

    diff = torch.abs(p - t)
    ratio = torch.max(p / t, t / p)

    RMS = torch.sqrt(torch.pow(diff, 2).mean())                             # Root Mean Square Error
    Log = (torch.abs( torch.log10(p+1e-3) - torch.log10(t+1e-3) )).mean()   # Average log10 Error
    Rel = (diff / t).mean()                                                 # Relative Error
    SqRel = torch.sqrt(( torch.pow(diff, 2) / t).mean())                    # Squared Relative Error

    delta1 = torch.sum( ratio < 1.25 ) / p.size(0)                          # Threshold Accuarcy 1.25
    delta2 = torch.sum( ratio < 1.25**2 ) / p.size(0)                       # Threshold Accuarcy 1.25^2
    delta3 = torch.sum( ratio < 1.25**3 ) / p.size(0)                       # Threshold Accuarcy 1.25^3

    return torch.tensor([RMS, Log, Rel, SqRel, delta1, delta2, delta3])

PATH = "/root"
valid_NYU = dataloader.get_dataloader(path=PATH, batch_size=1, split="test", shuffle=False, num_workers=12)
valid_tqdm = tqdm(enumerate(valid_NYU), total=len(valid_NYU))
get_lidar = pseudo_lidar(lidar_points=60)
metric_sum = torch.zeros(7)

starter = torch.cuda.Event(enable_timing=True)
ender   = torch.cuda.Event(enable_timing=True)

with torch.no_grad():
    for i, data in valid_tqdm:
        image, depth = data[0], data[1]
        lidar = get_lidar(depth=depth, epoch=20)

        image = image.to(device)
        depth = depth.to(device)
        lidar = lidar.to(device)

        starter.record()
        predict, centers = model(image, lidar)
        ender.record()
        torch.cuda.synchronize()

        p, c = predict.clone().detach(), centers.clone().detach()
        
        metric = cal_metric(p, depth)
        metric_sum += metric

        valid_tqdm.set_description(f"Valid: Time: {starter.elapsed_time(ender):3.2f} | RMS: {metric_sum[0]/(i+1):.3f} | REL: {metric_sum[2]/(i+1):.3f} | D1: {metric_sum[4]/(i+1):.3f}")
        plt.imsave(f"/root/output/{i}_gt.png", depth.cpu().squeeze(), cmap="magma_r", vmin=0.01, vmax=depth.max()*1.1)
        plt.imsave(f"/root/output/{i}_pr.png", p.cpu().squeeze(), cmap="magma_r", vmin=0.01, vmax=depth.max()*1.1)
