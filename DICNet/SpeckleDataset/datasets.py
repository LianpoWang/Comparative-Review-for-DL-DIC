import glob

import numpy as np
import torch
import torch.utils.data as data
import os
import cv2
import struct
import pandas as pd


#读取bin文件转化为numpy数组
def read_bin(filename, shape):
    with open(filename, 'rb') as f:
        data = struct.unpack('%df' % (2 * (shape[0]) * (shape[1])), f.read())
        return np.asarray(data).reshape(2, shape[0], shape[1])


class SpeckleDataset(data.Dataset):

    def __init__(self, csv_file,root_dir):
        self.Speckles_frame = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.file_list = sorted(glob.glob(os.path.join(root_dir, 'def_*.png')))  # 获取所有 tar 图像路径

    def __len__(self):
        return len(self.Speckles_frame)

    def __getitem__(self, idx):
        ref_path = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 0])
        tar_path = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 1])
        dispx_path = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 2])
        dispy_path= os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 3])


        ref=pd.read_csv(ref_path, header=None).values
        tar=pd.read_csv(tar_path, header=None).values
        dispx = pd.read_csv(dispx_path, header=None).values  # (H, W)
        dispy = pd.read_csv(dispy_path, header=None).values  # (H, W)

        # 确保形状一致
        h, w = ref.shape
        assert tar.shape == (h, w), f"Size mismatch: tar {tar.shape} vs ref {ref.shape}"
        assert dispx.shape == (h, w), f"Size mismatch: dispx {dispx.shape} vs ref {ref.shape}"
        assert dispy.shape == (h, w), f"Size mismatch: dispy {dispy.shape} vs ref {ref.shape}"

        # 转换为 Torch Tensor
        ref = torch.tensor(ref[np.newaxis, ...], dtype=torch.float32)  # (1, H, W)
        tar = torch.tensor(tar[np.newaxis, ...], dtype=torch.float32)  # (1, H, W)
        dispx = torch.tensor(dispx, dtype=torch.float32)  # (H, W)
        dispy = torch.tensor(dispy, dtype=torch.float32)  # (H, W)

        # 合并 dispx 和 dispy，形成 (2, H, W) 的张量
        disp = torch.stack((dispx, dispy), dim=0)  # (2, H, W)

        return ref, tar, disp

