import cv2
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import cv2 as cv
import torch

class R3DicDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):

        self.Speckles_frame = pd.read_csv(csv_file,header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.Speckles_frame)

    def __getitem__(self, idx):

        Ref_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 0])
        Def_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 1])
        Dispx_name = os.path.join(self.root_dir,  self.Speckles_frame.iloc[idx, 2])
        Dispy_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 3])
        # ✅ 读取图像（RGB格式）
        Ref_img = cv2.imread(Ref_name)  # 或者你原本的读取方式
        if Ref_img is None:
            raise FileNotFoundError(f"Image not found or cannot be opened: {self.ref_list[idx]}")

        Ref_img = cv2.cvtColor(Ref_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        Ref_img = cv2.cvtColor(Ref_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # RGB

        Def_img = cv2.imread(Def_name, cv2.IMREAD_COLOR)
        Def_img = cv2.cvtColor(Def_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # ✅ 读取 CSV（位移场）
        Dispx = np.genfromtxt(Dispx_name, delimiter=',').astype(np.float32)
        Dispy = np.genfromtxt(Dispy_name, delimiter=',').astype(np.float32)

        # ✅ 添加通道维度（H, W, C）→（C, H, W）
        Ref_img = np.transpose(Ref_img, (2, 0, 1))
        Def_img = np.transpose(Def_img, (2, 0, 1))
        Dispx = Dispx[np.newaxis, ...]
        Dispy = Dispy[np.newaxis, ...]

        sample = {
            'Ref': Ref_img,
            'Def': Def_img,
            'Dispx': Dispx,
            'Dispy': Dispy
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class Normalization(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        Ref, Def, Dispx, Dispy= sample['Ref'], sample['Def'], sample['Dispx'], sample['Dispy']

        self.mean = 0.0
        self.std = 255.0
        self.mean1 = 0.0
        self.std1 = 1.0

        return {'Ref': torch.from_numpy((Ref - self.mean) / self.std).float(),
                'Def': torch.from_numpy((Def - self.mean) / self.std).float(),
                'Dispx': torch.from_numpy((Dispx - self.mean1) / self.std1).float(),
                'Dispy': torch.from_numpy((Dispy - self.mean1) / self.std1).float(),

                }


