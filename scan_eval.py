import torch
import net
import torch.functional as F
from tqdm import tqdm
from math import pi
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image

class scan_dataset(Dataset):
    def __init__(self):
        super(scan_dataset, self).__init__()

        self.Resize_Crop = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.CenterCrop((228, 304)),
        ])

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


        self.path = "/root/lab_scan"
        self.l_csv = pd.read_csv(f"{self.path}/Lidar.csv")

    def __getitem__(self, idx):
        frame = self.l_csv.iloc[idx].to_list()
        l_f = frame[-30:]+frame[:30]
        lidar = torch.tensor(l_f, dtype=torch.float).unsqueeze(0) / 1000

        image=Image.open(f"{self.path}/capture_{idx}_color.png")
        depth=Image.open(f"{self.path}/capture_{idx}_depth.png")
        
        image = self.Resize_Crop(image)
        depth = self.Resize_Crop(depth)

        r_image =np.array(image, dtype=np.uint8)
        image = self.image_transform(image)
        # depth = transforms.ToTensor()(depth)
        depth = np.array(depth, dtype=np.uint16)
        depth = torch.from_numpy(depth).float() / 1000.0

        return [image, lidar, depth.unsqueeze(0), r_image]

    def __len__(self):
        return 508

def scan_loader():
    transform_test = scan_dataset()
    dataloader_test = DataLoader(dataset=transform_test, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    return dataloader_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = net.DepthNet(lidar_point=60, bin_size=64).to(device)
model.load_state_dict(torch.load("./pre_trained/bin_64.pth.tar", weights_only=True))
model.eval()

scan_set = scan_dataset()

PI = torch.tensor(pi)

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

scan_set = scan_loader()
scan_tqdm = tqdm(enumerate(scan_set), total=len(scan_set))

metric_sum = torch.zeros(7)

starter = torch.cuda.Event(enable_timing=True)
ender   = torch.cuda.Event(enable_timing=True)

with torch.no_grad():
    for i, data in scan_tqdm:
        image, lidar, depth, r_image = data[0], data[1], data[2], data[3]

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

        scan_tqdm.set_description(f"Valid: Time: {starter.elapsed_time(ender):3.2f} | RMS: {metric_sum[0]/(i+1):.3f} | REL: {metric_sum[2]/(i+1):.3f} | D1: {metric_sum[4]/(i+1):.3f}")
        plt.imsave(f"/root/output_scan/{i}_im.png", r_image.cpu().squeeze().numpy().astype(np.uint8))
        plt.imsave(f"/root/output_scan/{i}_gt.png", depth.cpu().squeeze(), cmap="magma_r", vmin=0.01, vmax=3)
        plt.imsave(f"/root/output_scan/{i}_pr.png", p.cpu().squeeze(), cmap="magma_r", vmin=0.01, vmax=3)
