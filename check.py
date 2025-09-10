import awkward as ak
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch, h5py

from torch.utils.data import DataLoader
from glob import glob
#from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from models.particle_transformer import ParticleTransformer
from utils.data import Dataset, cycle
from tqdm import tqdm

train_files = glob('data/train/DYJetsToMuMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/*.hdf5')
dataset_train = Dataset(files=train_files, max_PF_num=1024)
dataloader_train = DataLoader(
    dataset=dataset_train, batch_size=256, sampler=None,
    num_workers=8, pin_memory=True, drop_last=True, shuffle=True,
)
train_loader = tqdm(dataloader_train, desc=f"Epoch 0")
for index, (X, Y) in enumerate(train_loader):
    X, Y = X.to('cuda'), Y.to('cuda')
    if not torch.isfinite(X).all():
        print(index)
        print(X)
