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


import glob
import numpy as np
import torch
import torch.utils.data as data
import os
import cv2
import pandas as pd


class SpeckleDataset(data.Dataset):
    def __init__(self, csv_file, root_dir):
        self.Speckles_frame = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.Speckles_frame)

    def __getitem__(self, idx):
        # 构造路径
        ref_path = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 0])
        tar_path = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 1])
        dispx_path = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 2])
        dispy_path = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 3])

        # 使用 cv2 读取图像
        ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        tar = cv2.imread(tar_path, cv2.IMREAD_GRAYSCALE)

        if ref is None:
            raise ValueError(f"Failed to load reference image: {ref_path}")
        if tar is None:
            raise ValueError(f"Failed to load target image: {tar_path}")

        # 用 pandas 读取 csv 形式的位移图
        dispx = pd.read_csv(dispx_path, header=None).values  # (H, W)
        dispy = pd.read_csv(dispy_path, header=None).values  # (H, W)

        h, w = ref.shape
        assert tar.shape == (h, w), f"Size mismatch: tar {tar.shape} vs ref {ref.shape}"
        assert dispx.shape == (h, w), f"Size mismatch: dispx {dispx.shape} vs ref {ref.shape}"
        assert dispy.shape == (h, w), f"Size mismatch: dispy {dispy.shape} vs ref {ref.shape}"

        # 转换为 tensor
        ref = torch.tensor(ref[np.newaxis, ...], dtype=torch.float32)  # (1, H, W)
        tar = torch.tensor(tar[np.newaxis, ...], dtype=torch.float32)  # (1, H, W)
        dispx = torch.tensor(dispx, dtype=torch.float32)
        dispy = torch.tensor(dispy, dtype=torch.float32)
        disp = torch.stack((dispx, dispy), dim=0)  # (2, H, W)

        # 最后一项 torch.ones((h, w)) 可代表有效 mask
        return ref, tar, disp, torch.ones((h, w), dtype=torch.float32)
