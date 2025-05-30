import torch
import torch.nn as nn
import net
import matplotlib.pyplot as plt
from flops_profiler.profiler import get_model_profile

B = 1

class encap(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = net.DepthNet(lidar_point=60, f_size=[320,8,10], feat_ch=[320, 112, 40, 24, 16], d_model=256, bin_size=64).cuda()

    def forward(self, x):
        dummy_x = torch.rand([B, 3, 228, 304]).cuda()
        dummy_y = torch.rand([B, 1, 60]).cuda()

        predict, centers = self.model(dummy_x, dummy_y)
        # print(predict.size(), centers.size())

test = encap()
flops, macs, params = get_model_profile(model=test, input_shape=(1,1), print_profile=False)
print(f"FLOPs: {flops/1000000000:.2f}G | MACS: {macs/1000000000:.2f}G | Params: {params/1000000:.2f}M")
print("\n\n\n")

# ========================================================================================================================

from torch.profiler import profile, record_function, ProfilerActivity

model = net.DepthNet(lidar_point=60, f_size=[320,8,10], feat_ch=[320, 112, 40, 24, 16], d_model=256, bin_size=64).cuda()

dummy_x = torch.rand([B, 3, 228, 304]).cuda()
dummy_y = torch.rand([B, 1, 60]).cuda()

with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(dummy_x, dummy_y)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))