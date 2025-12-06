
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import torchvision.transforms as transforms
import numpy as np
import cv2 as cv
import torch


class dicDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.Speckles_frame = pd.read_csv(csv_file,header=None)
        self.root_dir = root_dir

        self.transform = transform

    def __len__(self):
        return len(self.Speckles_frame)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        Ref_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 0])
        Def_name = os.path.join(self.root_dir,  self.Speckles_frame.iloc[idx, 1])
        dispx_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 2])
        dispy_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 3])

        Ref_img = np.genfromtxt(Ref_name, delimiter=',').astype(np.float32)
        Def_img = np.genfromtxt(Def_name, delimiter=',').astype(np.float32)
        dispx = np.genfromtxt(dispx_name, delimiter=',').astype(np.float32)
        dispy = np.genfromtxt(dispy_name, delimiter=',').astype(np.float32)
        sample = {'Ref':Ref_img, 'Def': Def_img ,'dispx': dispx, 'dispy': dispy}
        if self.transform:
            sample = self.transform(sample)
        return  sample

