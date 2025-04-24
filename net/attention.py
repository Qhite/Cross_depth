import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, blocks):
        super(Residual, self).__init__()
        self.blocks = blocks
    def forward(self, x):
        res = x
        x = self.blocks(x)
        x += res
        return x

class Self_stage(nn.Module):
    def __init__(self, f_size=[320,8,10], d_model=256, num_head=16):
        super(Self_stage, self).__init__()
        assert (d_model % num_head) == 0, "InVaild Number of Multihead "
        self.f_size     = f_size
        self.d_model    = d_model
        self.num_head   = num_head
        self.head_dim   = d_model // num_head

        # Patch Embedding
        self.Patch_Embedding = nn.Sequential(
            nn.Conv2d(self.f_size[0], self.f_size[0], kernel_size=2, stride=2),
            nn.Conv2d(self.f_size[0], self.d_model  , kernel_size=1, stride=1),
            nn.GELU()
        )

        # Positional Encoding
        pe = self.f_size[1]//2 * self.f_size[2]//2
        self.PE = nn.Parameter(torch.rand(1, pe, self.d_model), requires_grad=True)

        # Query, Key, Value
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)

        # LayerNorm & Residual
        self.LNR = Residual(
            nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.d_model * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.d_model * 2, self.d_model)
            )
        )

        # Recover TransConv
        self.TransConv = nn.Sequential(
            nn.ConvTranspose2d(self.d_model, self.f_size[0], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(self.f_size[0])
        )

    def forward(self, img):
        B, C, H, W = img.size()
        ff = int(H * W / 4)

        # Patch Embedding
        x = self.Patch_Embedding(img)
        x = x.flatten(2).permute(0, 2, 1)
        x = x + self.PE
        
        # Q, K, V Projection
        Q = self.W_q(x).view(B, ff, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        K = self.W_k(x).view(B, ff, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        V = self.W_v(x).view(B, ff, self.num_head, self.head_dim).permute(0, 2, 1, 3)

        # Score
        Score = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        Score = F.softmax(Score, dim=-1)

        # Matmul Score and Value
        Attention_Value = torch.matmul(Score, V)

        # Concatenate and Feed-Forward Layer
        Concate_Atten_Value = Attention_Value.permute(0,2,1,3).contiguous().view(B, ff, self.d_model)
        x = self.LNR(Concate_Atten_Value)
        x = x.view(B, self.d_model, H//2, W//2)
        x = self.TransConv(x)

        return x

class Cross_stage(nn.Module):
    def __init__(self, f_size=[320,8,10], d_model=256, num_head=8, bin_size=256):
        super(Cross_stage, self).__init__()
        assert (d_model % num_head) == 0, "InVaild Number of Multihead "
        self.f_size     = f_size
        self.d_model    = d_model
        self.num_head   = num_head
        self.head_dim   = d_model // num_head
        self.bin_size   = bin_size


        # Patch Embedding
        self.Patch_Embedding = nn.Sequential(
            nn.Conv2d(self.f_size[0], self.f_size[0]//2, kernel_size=1, stride=1),
            nn.GELU()
        )
        self.Patch_Linear = nn.Sequential(
            nn.Linear(self.f_size[0]//2 * self.f_size[2], d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU()
        )
        self.Lidar_Embedding = nn.Sequential(
            nn.Linear(384, self.d_model),
            nn.GELU()
        )

        # Positional Encoding
        self.PE = nn.Parameter(torch.rand(1, self.f_size[1]+1, 1), requires_grad=True)

        # Query, Key, Value
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)

        # LayerNorm & Residual
        self.LNR = Residual(
            nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.d_model * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.d_model * 2, self.d_model)
            )
        )

        # Recover
        self.Width_Linear = nn.Sequential(
            nn.Linear(self.d_model, self.f_size[2]*self.d_model//2),
            nn.GELU(),
        )
        self.out_Conv = nn.Sequential(
            nn.Conv2d(self.d_model//2, self.f_size[0]//4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.f_size[0]//4)
        )

        # Bin Output
        self.bin_out = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model, self.bin_size, bias=False),
            nn.ReLU(),
        )

    def forward(self, img, ld):
        B, C, H, W = img.size()

        # Patch Embedding
        x = self.Patch_Embedding(img)
        x = x.permute(0, 2, 1, 3).flatten(2)
        x = self.Patch_Linear(x)
        y = self.Lidar_Embedding(ld)

        # Concate & Positional Embedding
        x = torch.concat((x, y), dim=1)
        x = x + self.PE
        
        # Q, K, V Projection
        Q = self.W_q(x).view(B, H+1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        K = self.W_k(x).view(B, H+1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        V = self.W_v(x).view(B, H+1, self.num_head, self.head_dim).permute(0, 2, 1, 3)

        # Score
        Score = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        Score = F.softmax(Score, dim=-1)

        # Matmul Score and Value
        Attention_Value = torch.matmul(Score, V)

        # Concatenate and Feed-Forward Layer
        Concate_Atten_Value = Attention_Value.permute(0,2,1,3).contiguous().view(B, H+1, self.d_model)
        x = self.LNR(Concate_Atten_Value)
        x, bin = x[:, :-1, :], x[:, -1, :]
        x = self.Width_Linear(x)
        x = x.view(B, H, -1, W)
        x = x.permute(0, 2, 1, 3)
        x = self.out_Conv(x)
        bin = self.bin_out(bin)

        return x, bin

class Attention_Block(nn.Module):
    def __init__(self, f_size=[320,8,10], d_model=256, bin_size=256):
        super(Attention_Block, self).__init__()

        self.Self_stage1 = Self_stage(f_size=f_size, d_model=d_model, num_head=16)
        self.Self_stage2 = Self_stage(f_size=f_size, d_model=64     , num_head=8 )
        self.Self_stage3 = Self_stage(f_size=f_size, d_model=d_model, num_head=16) # Output 320 ch
        
        self.Cross_stage = Cross_stage(f_size=f_size, d_model=128, num_head=8, bin_size=bin_size) # Output 320/4 ch

        self.concat_Conv = nn.Sequential(
            nn.Conv2d(400, 320, kernel_size=1, stride=1, bias=False), # 320 + 80
            nn.LeakyReLU()
        )
    
    def forward(self, img_f, ld_f):
        feat = self.Self_stage1(img_f)
        # feat = self.bottleneck_in(feat)
        feat = self.Self_stage2(feat)
        # feat = self.bottleneck_out(feat)
        feat = self.Self_stage3(feat)
        feat_c, bin = self.Cross_stage(feat, ld_f)
        feat = torch.concat((feat, feat_c), dim=1)
        feat = self.concat_Conv(feat)

        return feat, bin


if __name__ == "__main__":
    from flops_profiler.profiler import get_model_profile

    B = 1

    from flops_profiler.profiler import get_model_profile

    class encap(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention_Block(bin_size=64)

        def forward(self, x):
            dummy_x = torch.rand([B, 320, 8, 10])
            dummy_y = torch.rand([B, 1, 384])

            output, bin = self.attention(dummy_x, dummy_y)
            # print(output.size(), bin.size())
            pass
    
    test = encap()
    flops, macs, params = get_model_profile(model=test, input_shape=(1,1), print_profile=False)
    print(f"FLOPs: {flops/1000000000:.2f}G | MACS: {macs/1000000000:.2f}G | Params: {params/1000000:.2f}M")