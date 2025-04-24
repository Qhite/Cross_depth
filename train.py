import torch
import torch.nn.functional as F
import net
import dataloader as dataloader
import loss

from tqdm import tqdm
import wandb, datetime, pytz, math, yaml

PI = torch.tensor(math.pi)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(13)
torch.cuda.manual_seed(13)

torch.backends.cudnn.benchmark      = True
torch.backends.cudnn.deterministic  = True

PATH = "/root/lhw"

batch_size      = 80
learning_rate   = 4e-4
epochs          = 10
weight_decay    = 1e-2
d_model         = 128
bin_size        = 64
lidar_points    = 60

depth_range=[0.01, 10]

now = datetime.datetime.now()
now.astimezone(pytz.timezone("Asia/Seoul"))
run_time_tag = now.strftime("RUN_%m-%d-%I-%M-%S")

run_memo = f"Torch Transformer Bin: {bin_size}"

wandb.init(
    project="Cross_depth_rev",
    config={
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,    
        "weight_decay": weight_decay,
        "d_model": d_model,
        "bin_size": bin_size,
        "memo": run_memo
    }
)
wandb.run.name = run_time_tag
wandb.run.save()

# Model
Model = net.DepthNet(lidar_point=lidar_points, f_size=[320,8,10], feat_ch=[320, 112, 40, 24, 16], bin_size=bin_size).to(device)

# Dataloader
train_NYU = dataloader.get_dataloader(path=PATH, batch_size=batch_size, split="train", shuffle=True,  num_workers=12)
valid_NYU = dataloader.get_dataloader(path=PATH, batch_size=1, split="test", shuffle=False, num_workers=12)

# Optimizer
encoder = list(map(id, Model.encoder.parameters()))
params = filter(lambda p: id(p) not in encoder, Model.parameters())
optimizer = torch.optim.AdamW([{"params": Model.encoder.parameters(), "lr": learning_rate * 0.1},
                             {"params": params, "lr": learning_rate, },],
                             weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, learning_rate, epochs=epochs, steps_per_epoch=len(train_NYU),
                                              cycle_momentum=True,
                                              base_momentum=0.85, max_momentum=0.95,
                                              div_factor=25,
                                              final_div_factor=100)

# Loss
Loss = loss.Loss_functions(depth_min=depth_range[0])

# Functions
def update_lr(optim, epoch):
    decay_epoch = [10]
    if epoch in decay_epoch:
        for param_group in optim.param_groups:
            param_group['lr'] *= 0.1

def to_pcl(depth):
    B, _, H, W = depth.size()

    fx, fy = 582.62448167737955, 582.69103270988637
    cx, cy = 313.04475870804731, 238.44389626620386

    scale_x, scale_y = H / 640, W / 480

    fx *= scale_x
    fy *= scale_y
    cx *= scale_x
    cy *= scale_y

    # K = torch.tensor(
    #     [fx, 0, cx],
    #     [0, fy, cy],
    #     [0,  0,  1]
    # )

    u = torch.arange(0, W, device=device).view(1, -1).expand(H, -1)
    v = torch.arange(0, H, device=device).view(-1, 1).expand(-1, W)
    u = u.unsqueeze(0).expand(B, -1, -1)
    v = v.unsqueeze(0).expand(B, -1, -1)

    z = depth.squeeze(1).to(device)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # xyz = torch.stack((x,y,z), dim=1)
    # xyz = xyz.view(B, 3, -1).permute(0, 2, 1) # B N 3

    return x, y, z

class pseudo_lidar():
    def __init__(self, lidar_points=116):
        self.lp = lidar_points
        self.min_fov  = -22.2
        self.max_fov  =  11.6
        self.fov_step = 0.325
        self.max_step = int((self.max_fov - self.min_fov) // self.fov_step) + 1 # 104
    
    def __call__(self, depth, epoch, vaild=False):
        B, _, H, W = depth.size()

        x, y, z = to_pcl(depth=depth)
        d = torch.sqrt(x**2 + z**2)
        theta = torch.atan2(y, d) / PI * 180
        
        top     = int(42 - 0.5 * epoch)
        bottom  = int(63 + 0.5 * epoch)
        sel = torch.randint(top, bottom, (B,1,1), device=device)

        if vaild:
            sel = 52

        mask_up = torch.ge(theta, self.min_fov + sel     * self.fov_step)
        mask_lo = torch.le(theta, self.min_fov + (sel+1) * self.fov_step)

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

    mask_p = torch.ge(predict, depth_range[0])
    mask_t = torch.ge(target, depth_range[0])
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

def print_metric(metric):
    print(f"RMS: {metric[0]} | REL: {metric[2]} | D1: {metric[4]}")

def save_model(model):
    data = {
        "batch_size":       batch_size,
        "learning_rate":    learning_rate,
        "epochs":           epochs,
        "weight_decay":     weight_decay,
        "d_model":          d_model,
        "bin_size":         bin_size,
        "lidar_points":     lidar_points,
        "memo":             run_memo,
    }
    with open(f"{PATH}/Cross_depth/trained/{run_time_tag}.yaml", 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    torch.save(model.state_dict(), f"{PATH}/Cross_depth/trained/{run_time_tag}.pth.tar")

# Training
def train(model):
    print(f"Run Name: {run_time_tag}")
    for epoch in range(epochs):
        model.train()

        # update_lr(optim=optimizer, epoch=epoch)

        train_tqdm = tqdm(enumerate(train_NYU), total=len(train_NYU))
        loss_sum = torch.zeros(1)
        get_lidar = pseudo_lidar(lidar_points)
        
        for i, data in train_tqdm:
            image, depth = data[0], data[1]
            lidar = get_lidar(depth=depth, epoch=epoch)
            
            image = image.to(device)
            depth = depth.to(device)
            lidar = lidar.to(device)

            optimizer.zero_grad()

            predict, centers = model(image, lidar)

            loss = Loss(predict, centers, depth, lidar)
            loss.backward()
            
            loss_sum += loss.clone().detach().cpu()

            optimizer.step()
            scheduler.step()

            if (i>0) & (i%10==0):
                wandb.log({
                    'train_loss per 10 Step': float(loss.clone().detach().cpu())
                })

            train_tqdm.set_description(f"Train: Epoch {epoch+1:2d}/{epochs:2d} | Loss {float(loss_sum)/(i+1):.3f}")

        avg_loss, avg_metric = validate(epoch, model)

        wandb.log(data={
            'train_loss': float(loss_sum/len(train_NYU)),
            'valid_loss': float(avg_loss),
            'RMS': float(avg_metric[0]),
            'Log10': float(avg_metric[1]),
            'REL': float(avg_metric[2]),
            'SqREL': float(avg_metric[3]),
            'd1': float(avg_metric[4]),
            'd2': float(avg_metric[5]),
            'd3': float(avg_metric[6]),
        })
        print("\n============================")
    save_model(model)

# Validation
def validate(epoch, model):
    model.eval()
    valid_tqdm = tqdm(enumerate(valid_NYU), total=len(valid_NYU))
    metric_sum = torch.zeros(7)
    loss_sum = torch.zeros(1)
    get_lidar = pseudo_lidar(lidar_points)

    for i, data in valid_tqdm:
        image, depth = data[0], data[1]
        lidar = get_lidar(depth=depth, epoch=epoch, vaild=True)

        image = image.to(device)
        depth = depth.to(device)
        lidar = lidar.to(device)

        with torch.no_grad():
            predict, centers = model(image, lidar)
            p, c = predict.clone().detach(), centers.clone().detach()
        
        loss = Loss(p, c, depth, lidar).detach()
        loss_sum += loss.cpu()
        metric = cal_metric(p, depth)
        metric_sum += metric

        valid_tqdm.set_description(f"Valid: Epoch {epoch+1:2d}/{epochs:2d} | Loss {float(loss_sum)/(i+1):.3f} | RMS: {metric_sum[0]/(i+1):.3f} | REL: {metric_sum[2]/(i+1):.3f} | D1: {metric_sum[4]/(i+1):.3f}")

    avg_loss = loss_sum/len(valid_NYU)
    avg_metric = metric_sum/len(valid_NYU)
    return avg_loss, avg_metric

if __name__ == "__main__":
    train(Model)
