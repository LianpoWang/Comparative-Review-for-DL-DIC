
from PIL import Image
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from scipy.ndimage import zoom
def default_loader(path):
    return Image.open(path)
# light_index_2 = [2,7,15,8,4,22,13,57,54,40,91,21,29,84,71,25,28,51,67,62,34,46,93,87]
class MyDataset(Dataset):
    def __init__(self, csv_file,root_dir,total):
        self.Speckles_frame = pd.read_csv(csv_file,header=None)
        self.root_dir = root_dir
    def __len__(self):
        return len(self.Speckles_frame)

    def __getitem__(self, idx):
        Ref_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 0])
        Def_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 1])
        Dispx_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 2])
        Dispy_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 3])

        ref = pd.read_csv(Ref_name, header=None).values  # (H, W)
        tar = pd.read_csv(Def_name, header=None).values  # (H, W)
        dispx = pd.read_csv(Dispx_name, header=None).values  # (H, W)
        dispy = pd.read_csv(Dispy_name, header=None).values  # (H, W)
            # 计算缩放比例
        target_size = (128, 128)
        scale_x = target_size[1] / ref.shape[1]  # W 方向缩放比例
        scale_y = target_size[0] / ref.shape[0]  # H 方向缩放比例

        # 使用 scipy.ndimage.zoom 进行插值缩放
        ref_resized = zoom(ref, (scale_y, scale_x))
        tar_resized = zoom(tar, (scale_y, scale_x))
        dispx_resized = zoom(dispx, (scale_y, scale_x))
        dispy_resized = zoom(dispy, (scale_y, scale_x))

        # 转换为 Tensor
        img_1 = ToTensor()(ref_resized.astype(np.float32))  # 转换为 float32 避免 PyTorch 报错
        img_2 = ToTensor()(tar_resized.astype(np.float32))
        imgs = torch.cat((img_1, img_2), 0)  # 拼接成 2 通道输入

        # 统一尺寸
        gt = np.stack([dispx_resized, dispy_resized], axis=0).astype(np.float32)  # (2, H, W)
        return imgs, torch.from_numpy(gt)

def build_train_dataset():
    train_dataset = MyDataset("/home/dell/DATA/wh/DATASET/Train_annotations_1.csv",'/home/dell/DATA/wh/DATASET/Train10/', 34560)
    return train_dataset
def build_val_dataset():
    val_dataset= MyDataset("/home/dell/DATA/wh/DATASET/Val_annotations_1.csv",'/home/dell/DATA/wh/DATASET/Train10/',8640 )
    return val_dataset
def build_test_dataset():
    test_dataset = MyDataset("/home/dell/DATA/wh/DATASET/Test_annotations1.csv",'/home/dell/DATA/wh/DATASET/Test10/', 10800)
    return test_dataset
