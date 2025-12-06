import glob
import numpy as np
import torch
import torch.utils.data as data
import os
import cv2
import struct
import pandas as pd


# 读取 bin 文件转化为 numpy 数组
def read_bin(filename, shape):
    with open(filename, 'rb') as f:
        data = struct.unpack('%df' % (2 * (shape[0]) * (shape[1])), f.read())
        return np.asarray(data).reshape(2, shape[0], shape[1])


class DICDataset(data.Dataset):
    def __init__(self, root_dir,csv_file, is_pretrain=True):
        """
        初始化数据集类

        :param root_dir: 数据文件的根目录
        :param is_pretrain: 是否为预训练阶段，预训练阶段需要标签数据
        """
        self.Speckles_frame = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.is_pretrain = is_pretrain
        self.file_list = sorted(glob.glob(os.path.join(root_dir, 'def_*.png')))  # 获取所有变形图像路径

    def __len__(self):
        """
        返回数据集的长度
        """
        return len(self.Speckles_frame)

    def __getitem__(self, idx):
        Ref_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 0])
        Def_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 1])
        Dispx_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 2])
        Dispy_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 3])

        Ref = torch.tensor(np.genfromtxt(Ref_name, delimiter=',')).unsqueeze(0).float()
        Def = torch.tensor(np.genfromtxt(Def_name, delimiter=',')).unsqueeze(0).float()
        Dispx = torch.tensor(np.genfromtxt(Dispx_name, delimiter=',')).unsqueeze(0).float()
        Dispy = torch.tensor(np.genfromtxt(Dispy_name, delimiter=',')).unsqueeze(0).float()

        return Ref, Def, Dispx,Dispy



def build_train_dataset():
    train_dataset = DICDataset('/home/dell/DATA/wh/DATASET/Train1',"/home/dell/DATA/wh/DATASET/Train_annotations_1.csv", is_pretrain=True)
    return train_dataset


def build_val_dataset():
    val_dataset = DICDataset('/home/dell/DATA/wh/DATASET/Val1',"/home/dell/DATA/wh/DATASET/Val_annotations_1.csv", is_pretrain=True)
    return val_dataset


def build_test_dataset():
    test_dataset = DICDataset('/home/dell/DATA/wh/DATASET/Test1',"/home/dell/DATA/wh/DATASET/Test_annotations1.csv", is_pretrain=True)
    return test_dataset