import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Residual(nn.Module):
    def __init__(self, blocks):
        super(Residual, self).__init__()
        self.blocks = blocks
    def forward(self, x):
        res = x
        x = self.blocks(x)
        x += res
        return x

class Self_Attention_Block(nn.Module):
    def __init__(self, num_head=4, d_model=160, f_size=[8,10], feat_dim=None):
        super(Self_Attention_Block, self).__init__()
        self.num_head = num_head
        self.d_model = d_model
        self.head_dim = d_model // num_head
        self.f_size = f_size

        # Patch Encodeing
        self.patch_conv = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 2, 2), # patch size 2x2
            nn.Conv2d(feat_dim, d_model, 1),
            nn.LeakyReLU()
        )

        # Positional Embedding
        pe = math.floor(self.f_size[0]/2) * math.floor(self.f_size[1]/2)
        self.PE = nn.Parameter(torch.rand(pe+1, 1), requires_grad=True)
        
        # Q, K, V Linear layers
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)

        # Layer Norm & Residual
        self.LNR = Residual(nn.LayerNorm(self.d_model))

    def forward(self, x):
        batch_size = x.size(0)

        # Patch Encoding
        f = self.patch_conv(x).flatten(2).permute(0, 2, 1)
        f = F.pad(f, (0,0,0,1)) + self.PE

        # Q, K, V Linear Projection
        Q = self.W_q(f).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        K = self.W_k(f).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        V = self.W_v(f).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)

        # Get Score
        Score = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        Score = F.softmax(Score, dim=-1)

        # Matmul Score and Value
        Attention_Value = torch.matmul(Score, V)

        # Concatenate and Feed Forward/Dense Layer
        Concate_Attention_Value = Attention_Value.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        feat = Concate_Attention_Value[:,:-1,:]
        feat = self.LNR(feat)
        feat = feat.permute(0, 2, 1).view(batch_size, -1, self.f_size[0]//2, self.f_size[1]//2)
        feat = F.interpolate(feat, size=self.f_size, mode="nearest")
        
        return feat

class Cross_Attention_Block(nn.Module):
    def __init__(self, num_head=4, d_model=256, f_size=[8,10], feat_dim=None):
        super(Cross_Attention_Block, self).__init__()
        self.num_head = num_head
        self.d_model = d_model
        self.head_dim = d_model // num_head
        self.f_size = f_size

        # Patch Encoding
        self.Conv = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim//2, 1, 1, 0),
            nn.LeakyReLU()
        )
        self.patch_linear = nn.Sequential(
            nn.Linear(self.f_size[1]*feat_dim//2, self.f_size[1]*feat_dim//4),
            nn.LeakyReLU()
        )

        # LiDAR to Token
        self.L2T = nn.Sequential(
            nn.Linear(384, self.f_size[1]*feat_dim//4),
            nn.LeakyReLU()
        )

        # Positional Embedding
        self.PE = nn.Parameter(torch.rand(self.f_size[0]+1, 1), requires_grad=True)

        # Q, K, V Linear layers
        self.W_q = nn.Linear(self.f_size[1]*feat_dim//4, self.d_model, bias=False)
        self.W_k = nn.Linear(self.f_size[1]*feat_dim//4, self.d_model, bias=False)
        self.W_v = nn.Linear(self.f_size[1]*feat_dim//4, self.d_model, bias=False)

        # Layer Norm & Residual
        self.LNR = Residual(nn.LayerNorm(self.d_model))

        # Output
        self.b_o = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        
        self.f_o = nn.Sequential(
            nn.Linear(d_model, self.f_size[1]*feat_dim//8)
        )

    def forward(self, d_dict):
        x, y = d_dict["feat"], d_dict["lidar"]
        batch_size = x.size(0)

        f = self.Conv(x)
        f = f.transpose(1,2).contiguous().flatten(2)
        f = self.patch_linear(f)
        f = F.pad(f, (0,0,0,1))

        y = self.L2T(y)
        f[:,-1,:] = y[:,0,:]

        f += self.PE

        # Q, K, V Linear Projection
        Q = self.W_q(f).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        K = self.W_k(f).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        V = self.W_v(f).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)

        # Get Score
        Score = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        Score = F.softmax(Score, dim=-1)

        # Matmul Score and Value
        Attention_Value = torch.matmul(Score, V)
        
        # Concatenate and Feed Forward/Dense Layer
        Concate_Attention_Value = Attention_Value.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        Concate_Attention_Value = self.LNR(Concate_Attention_Value)

        bins = Concate_Attention_Value[:,-1,:]
        bins = self.b_o(bins)

        feat = Concate_Attention_Value[:,:-1,:]
        feat = self.f_o(feat)
        
        feat = feat.view(batch_size, self.f_size[0], -1, self.f_size[1]).permute(0,2,1,3).contiguous()

        return feat, bins

class Attention_Block(nn.Module):
    def __init__(self, input_size=[], feat_dim=None):
        super(Attention_Block, self).__init__()

        self.Self_Atten = nn.Sequential(
            Self_Attention_Block(num_head=8, d_model=256, feat_dim=320),
            Self_Attention_Block(num_head=4, d_model=64 , feat_dim=256),
            Self_Attention_Block(num_head=8, d_model=256, feat_dim=64 ),
        )
        self.Cross_Atten = Cross_Attention_Block(num_head=8, d_model=256, feat_dim=256)

        self.Conv = nn.Sequential(
            nn.Conv2d(288, 288, 1, 1, 0),
            nn.BatchNorm2d(288),
            nn.LeakyReLU()
        )

    def forward(self, img_f, ld_f):
        x, y = img_f, ld_f
        x = self.Self_Atten(x)
        feat, bins = self.Cross_Atten({"feat":x, "lidar":y})

        x = torch.cat((x, feat), 1)
        x = self.Conv(x)

        return x, bins
    
if __name__ == "__main__":
    from flops_profiler.profiler import get_model_profile

    B = 10

    from flops_profiler.profiler import get_model_profile

    class encap(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention_Block()

        def forward(self, x):
            dummy_x = torch.rand([B, 320, 8, 10])
            dummy_y = torch.rand([B, 1, 384])

            output, bin = self.attention(dummy_x, dummy_y)
            # print(output.size(), bin.size())
            pass
    
    test = encap()
    flops, macs, params = get_model_profile(model=test, input_shape=(1,1), print_profile=False)
    print(f"FLOPs: {flops/1000000000:.2f}G | MACS: {macs/1000000000:.2f}G | Params: {params/1000000:.2f}M")