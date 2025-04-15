import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
import pandas as pd
from PIL import Image

class NYU_dataset(Dataset):
    def __init__(self, path="/root",split="train"):
        super(NYU_dataset, self).__init__()

        if split == "train":
            self.scale = 10
        elif split == "test":
            self.scale = 1e-3

        self.path = path
        self.datalist_file = f"{self.path}/data/nyu2_{split}.csv"
        self.frame = pd.read_csv(self.datalist_file, header=None)

        __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]), 
        'eigvec': torch.Tensor([ [-0.5675,  0.7192,  0.4009], 
                                [-0.5808, -0.0045, -0.8140], 
                                [-0.5836, -0.6948,  0.4203] ])
        }

        self.Resize_Crop = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.CenterCrop((228, 304)),
        ])

        if split == "train":
            self.RandomFlip = RandomHorizontalFlipBoth()
        else:
            self.RandomFlip = RandomHorizontalFlipBoth(0)

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.frame)
    
    def __getitem__(self, index):
        image_name=self.frame.iloc[index, 0]
        depth_name=self.frame.iloc[index, 1]

        image = Image.open(f"{self.path}/"+image_name)
        depth = Image.open(f"{self.path}/"+depth_name)

        image = self.Resize_Crop(image)
        depth = self.Resize_Crop(depth)

        image, depth = self.RandomFlip(image, depth)

        image = self.image_transform(image)
        depth = transforms.ToTensor()(depth) * self.scale

        return [image, depth]

def get_dataloader(path="/root", batch_size=13, split="train", shuffle=True, num_workers=6):
    NYU = NYU_dataset(path=path, split=split)
    return DataLoader(dataset=NYU, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

class RandomHorizontalFlipBoth(object):
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, depth):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            depth = F.hflip(depth)
        return image, depth

class Lighting(object):
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec
    
    def __call__(self, image):
        if self.alphastd == 0:
            return image

        alpha = image.new(3).normal_(0, self.alphastd)

        rgb = self.eigvec.type_as(image).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        image = image.add(rgb.view(3, 1, 1).expand_as(image))

        return image


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train_NYU = get_dataloader("/root", 10, "train", True, 12)
    valid_NYU = get_dataloader("/root", 1, "test", False, 12)

    for i, d in train_NYU:
        plt.imshow(i[0].squeeze().permute(1,2,0))
        print(d.max())
        plt.pause(0.1)
        print(i.size(), d.size())
        
