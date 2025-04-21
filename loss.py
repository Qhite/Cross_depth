import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from pytorch3d.loss import chamfer_distance

class Loss_functions(nn.Module):
    def __init__(self, depth_min=0.001, alpha=10.0, beta1=0.1, beta2=0.001):
        super().__init__()
        self.SILL     = SILogLoss(depth_min=depth_min)
        self.BC_depth = BinsChamferLoss(depth_min=depth_min)
        self.BC_Lidar = BinsChamferLoss(depth_min=depth_min)
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2

    def forward(self, predict, centers, target, lidar):
        if predict.size(3) != target.size(3):
            _, _, H, W = target.size()
            predict = F.interpolate(predict, size=[H, W], mode="bilinear", align_corners=True)
        
        SIL_loss = self.SILL(predict, target)
        BCL_depth_loss = self.BC_depth(centers, target)
        BCL_Lidar_loss = self.BC_Lidar(centers, lidar)

        loss = self.alpha * SIL_loss + self.beta1 * BCL_depth_loss + self.beta2 * BCL_Lidar_loss

        return loss

class SILogLoss(nn.Module):  
    def __init__(self, lamb=0.85, depth_min=0.001):
        super(SILogLoss, self).__init__()
        self.lamb = lamb
        self.d_min = depth_min

    def forward(self, predict, target):
        mask_predict = predict.ge(self.d_min)
        mask_target  = target.ge(self.d_min)
        mask = torch.logical_and(mask_target, mask_predict)

        masked_predict = (predict * mask).flatten(1)
        masked_target  = (target  * mask).flatten(1)

        g = torch.log(masked_predict + 1e-5) - torch.log(masked_target + 1e-5)

        Dg = torch.var(g) + (1 - self.lamb) * torch.pow(torch.mean(g), 2)
        losses = torch.sqrt(Dg)

        return losses

class BinsChamferLoss(nn.Module):
    def __init__(self, depth_min=0.001):
        super(BinsChamferLoss, self).__init__()
        self.d_min = depth_min
    
    def forward(self, bin_centers, target):
        if len(bin_centers.shape) == 1:
            bin_centers = bin_centers.unsqueeze(0).unsqueeze(2)
        else:
            bin_centers = bin_centers.unsqueeze(2)

        target_points = target.flatten(1).float()

        mask = target_points.ge(self.d_min)
        target_points = [p[m] for p, m in zip(target_points, mask)]

        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target.device)

        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)

        loss, _ = chamfer_distance(x=bin_centers, y=target_points, y_lengths=target_lengths)

        return loss


if __name__ == "__main__":
    dummy_predict = torch.rand([10,1,228,304])
    dummy_depth   = torch.rand([10,1,228,304])
    dummy_centers = torch.rand([10,128])
    dummy_lidar   = torch.rand([10,116])

    SIL = SILogLoss()
    BCL = BinsChamferLoss()
    LOSS= Loss_functions()

    loss_l = SIL(dummy_predict, dummy_depth)
    loss_b = BCL(dummy_centers, dummy_depth)
    loss   = LOSS(dummy_predict, dummy_centers, dummy_depth, dummy_lidar)

    print(loss)
