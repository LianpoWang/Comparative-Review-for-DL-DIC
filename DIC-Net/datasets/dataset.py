import glob
import os

import cv2
import pandas as pd
import scipy.io as sio
import torch.utils.data as data_utils
import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image
def default_loader(path):
    return Image.open(path)
# light_index_2 = [2,7,15,8,4,22,13,57,54,40,91,21,29,84,71,25,28,51,67,62,34,46,93,87]
class MyDataset(data_utils.Dataset):
    def __init__(self, root_dir,total):
        print(f"root_dir: {root_dir}, type: {type(root_dir)}")  # 调试用
        self.root_dir = root_dir
        self.file_list = sorted(glob.glob(os.path.join(root_dir, 'def_*.png')))
        self.total = len(self.file_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        tar_path = self.file_list[index]
        refimg_name = tar_path.replace("def", "ref")
        # 生成完整的路径
        ref_path = os.path.join(self.root_dir, refimg_name)

        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Image file not found: {ref_path}")

        ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        if ref is None:
            raise ValueError(f"Reference image not found: {ref_path}")
        # 读取当前 tar
        tar = cv2.imread(tar_path, cv2.IMREAD_GRAYSCALE)
        if tar is None:
            raise ValueError(f"Failed to load target image: {tar_path}")

        ref_resized = cv2.resize(ref, (128, 128))  # OpenCV 的 resize
        tar_resized = cv2.resize(tar, (128, 128))  # OpenCV 的 resize

        # 转换为 Tensor
        img_1 = ToTensor()(ref_resized)
        img_2 = ToTensor()(tar_resized)
        imgs = torch.cat((img_1, img_2), 0)  # 拼接成 2 通道输入


        # 读取对应的 dispx 和 dispy（CSV 格式）
        dispx_path = tar_path.replace("def", "dispx").replace(".png", ".csv")
        dispy_path = tar_path.replace("def", "dispy").replace(".png", ".csv")

        if not os.path.exists(dispx_path) or not os.path.exists(dispy_path):
            raise ValueError(f"Displacement files missing: {dispx_path} or {dispy_path}")

        dispx = pd.read_csv(dispx_path, header=None).values  # (H, W)
        dispy = pd.read_csv(dispy_path, header=None).values  # (H, W)

        # 统一尺寸（假设原始大小与图像相同）
        dispx = dispx[::2, ::2]  # 下采样（步长=2）
        dispy = dispy[::2, ::2]  # 下采样（步长=2）

        # 拼接成 (2, H, W) 形状
        gt = np.stack([dispx, dispy], axis=0).astype(np.float32)

        return imgs, gt
