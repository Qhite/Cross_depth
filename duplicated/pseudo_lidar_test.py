import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from datasets import load_dataset
from flops_profiler.profiler import get_model_profile
import math

PI = torch.tensor(math.pi)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

depth = torch.rand([1,1,228,304])*10
B, _, H, W = depth.size()

fx, fy = 582.62448167737955, 582.69103270988637
cx, cy = 313.04475870804731, 238.44389626620386

scale_x, scale_y = H / 640, W / 480

fx *= scale_x
fy *= scale_y
cx *= scale_x
cy *= scale_y

# K = torch.(tensor(
#     [fx, 0, cx],
#     [0, fy, cy],
#     [0,  0,  1]
# )

u = torch.arange(0, W).view(1, -1).expand(H, -1)
v = torch.arange(0, H).view(-1, 1).expand(-1, W)
u = u.unsqueeze(0).expand(B, -1, -1)
v = v.unsqueeze(0).expand(B, -1, -1)

z = depth.squeeze(1)

x = (u - cx) * z / fx
y = (v - cy) * z / fy
d = torch.sqrt(x**2 + z**2)
theta = torch.atan2(y, d) / PI * 180.0

min_fov = -22.2
max_fov = 11.6
fov_step = 0.325
steps = int((max_fov - min_fov) // fov_step) + 1

print(steps)

sel = torch.randint(20, 95, (B,1,1))

# sel = 57

# print(theta.size())

mask_up = torch.ge(theta, min_fov+sel*fov_step)
mask_lo = torch.le(theta, min_fov+(sel+1)*fov_step)

mask = torch.logical_and(mask_lo, mask_up).unsqueeze(1)

plt.imshow(mask.squeeze())
plt.show()

lidar = depth * mask


mask_hit = mask.sum(dim=2)


lidar = lidar.sum(dim=2)
print(lidar.size())

print(lidar)
lidar = lidar/mask_hit


indices = torch.linspace(0, lidar.size(2)-1, 116).int()
lidar = lidar[:, :, indices]
print(lidar.size())




# depth = mask.unsqueeze(1) * depth

# print(depth.size())
# plt.imshow(x[0])
# plt.imshow(y[0])
# plt.imshow(theta[0])
# plt.imshow(mask[0].squeeze())
# plt.imshow(depth[0].squeeze())
# plt.show()

# xyz = torch.stack((x,y,z), dim=1)
# xyz = xyz.view(B, 3, -1).permute(0, 2, 1) # B N 3

# print(xyz.size())