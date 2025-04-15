#! /bin/bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda==12.1 -c pytorch -c nvidia -y
conda install scipy opencv numpy pandas pyarrow matplotlib yacs tqdm -c conda-forge -y
pip install importlib-metadata

# pip install swing gymnasium[box2d]

# Addtional

conda install datasets h5py -y

pip install wandb

conda install fvcore iopath -c fvcore -c iopath -c conda-forge -y
pip install "git+https://github.com/facebookresearch/pytorch3d.git"