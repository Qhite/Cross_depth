import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Encoder(nn.Module):
    def __init__(self, lidar_point=116):
        super(Encoder, self).__init__()
        self.backbone = torchvision.models.efficientnet_b0(weights="DEFAULT").features
        
        #Encoder
        self.enc_stage1 = self.backbone[:2]
        self.enc_stage2 = self.backbone[2:3]
        self.enc_stage3 = self.backbone[3:4]
        self.enc_stage4 = self.backbone[4:6]
        self.enc_stage5 = self.backbone[6:8]
        self.enc_end    = self.backbone[8:9]

        self.l_enc = nn.Sequential(
            nn.Linear(lidar_point, 256),
            nn.LeakyReLU(),
            nn.Conv1d(1, 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(2, 4, kernel_size=1, stride=1),
            nn.LeakyReLU(),
        )
    
    def forward(self, x, y):
        B, C, H, W = x.size()
        skip_connection_buffer = []

        # Encoder
        x = self.enc_stage1(x)
        skip_connection_buffer.append(x)
        x = self.enc_stage2(x)
        skip_connection_buffer.append(x)
        x = self.enc_stage3(x)
        skip_connection_buffer.append(x)
        x = self.enc_stage4(x)
        skip_connection_buffer.append(x)
        x = self.enc_stage5(x)

        y = self.l_enc(y)

        return x, y, skip_connection_buffer

if __name__ == "__main__":
    from flops_profiler.profiler import get_model_profile

    B = 1

    class encap(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder(lidar_point=116)

        def forward(self, x):
            dummy_x = torch.rand([B, 3, 228, 304])
            dummy_y = torch.rand([B, 1, 116])

            img_f, ld_f, buff = self.encoder(dummy_x, dummy_y)
            # print(img_f.size(), ld_f.size())
            pass
    
    test = encap()
    flops, macs, params = get_model_profile(model=test, input_shape=(1,1), print_profile=False)
    print(f"FLOPs: {flops/1000000000:.2f}G | MACS: {macs/1000000000:.2f}G | Params: {params/1000000:.2f}M")
