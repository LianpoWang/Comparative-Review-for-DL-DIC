import torch
from imageio import imread
import numpy as np
import os
from torch.utils.data import Dataset
import pandas as pd


def default_loader(root, re_img_name, tar_img_name, dispx_name, dispy_name, network_arch):
    """
    加载图像和位移场数据
    Args:
        root: 图像数据的路径
        re_img_name: 参考图像文件名
        tar_img_name: 目标图像文件名
        dispx_name: x 方向位移场文件名
        dispy_name: y 方向位移场文件名
        network_arch: 网络架构

    Returns:
        re_img: 参考图像
        tar_img: 目标图像
        output_dis: 真实的位移场
    """
    # 读取图像数据
    re_img = np.array(pd.read_csv(os.path.join(root, re_img_name), header=None))
    tar_img = np.array(pd.read_csv(os.path.join(root, tar_img_name), header=None))
    # 归一化
    re_img = (re_img - np.mean(re_img)) / np.max(np.abs(re_img - np.mean(re_img)))
    tar_img = (tar_img - np.mean(tar_img)) / np.max(np.abs(tar_img - np.mean(tar_img)))

    # 根据网络架构选择位移数据
    if network_arch == 'StrainNet_f':
        output_dis_u = np.array(pd.read_csv(os.path.join(root, dispx_name), header=None))
        output_dis_v = np.array(pd.read_csv(os.path.join(root, dispy_name), header=None))
        output_dis = np.zeros([2, output_dis_u.shape[0], output_dis_u.shape[1]])
        output_dis[0, :, :] = output_dis_u
        output_dis[1, :, :] = output_dis_v
    else:
        output_dis = np.array(pd.read_csv(os.path.join(root, dispx_name), header=None))

    return re_img, tar_img, output_dis


class image_dataset(Dataset):
    """
    自定义数据集类
    Args:
        root: 数据路径
        csv_file: 存储图像和位移场文件名的 CSV 文件
        network_arch: 网络架构
        loader: 数据加载函数，默认为 default_loader
    """
    def __init__(self, root, csv_file, network_arch='U_DICNet', loader=default_loader):
        self.root = root
        self.csv_file = csv_file
        self.network_arch = network_arch
        self.loader = loader

        # 读取 CSV 文件，存储文件名列表
        self.speckles_frame = pd.read_csv(csv_file, header=None)

    def __getitem__(self, index):
        # 获取当前索引对应的文件名
        re_img_name = self.speckles_frame.iloc[index, 0]
        tar_img_name = self.speckles_frame.iloc[index, 1]
        dispx_name = self.speckles_frame.iloc[index, 2]
        dispy_name = self.speckles_frame.iloc[index, 3]

        # 加载数据
        re_img, tar_img, target = self.loader(self.root, re_img_name, tar_img_name, dispx_name, dispy_name, self.network_arch)

        # 转换为 Tensor
        input1 = torch.from_numpy(re_img).float()
        input2 = torch.from_numpy(tar_img).float()
        target = torch.from_numpy(target).float()

        # 增加通道维度
        input1 = input1[np.newaxis, ...]
        input2 = input2[np.newaxis, ...]

        # 处理输入格式
        if self.network_arch == 'StrainNet_f':
            input_img = torch.cat((input1, input2), 0)
            input_img = torch.cat((input_img, input_img, input_img), 0)
        else:
            input_img = torch.cat((input1, input2), 0)
            target = target[np.newaxis, ...]

        return input_img, target

    def __len__(self):
        return len(self.speckles_frame)  # 改为正确的 CSV 数据长度
