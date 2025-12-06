import os.path
import os
import pandas as pd

from .image_dataset import image_dataset
import glob


def make_dataset(file_path, csv_file, network_arch):
    """
    生成数据集的文件路径列表。
    Args:
        file_path (str): 图像数据的路径。
        csv_file (str): 存储图像和位移场文件名的 CSV 文件。
        network_arch (str): 网络架构。
    Returns:
        tuple: 包含参考图像、目标图像和位移场的文件路径列表。
    """
    ref_img_list, tar_img_list, dis_u_list, dis_v_list, displacement_list = [], [], [], [], []

    # 读取 CSV 文件
    speckles_frame = pd.read_csv(csv_file, header=None)

    for _, row in speckles_frame.iterrows():
        ref_img = os.path.join(file_path, row[0])
        tar_img = os.path.join(file_path, row[1])
        dis_name_u = os.path.join(file_path, row[2])
        dis_name_v = os.path.join(file_path, row[3])

        ref_img_list.append(ref_img)
        tar_img_list.append(tar_img)
        dis_u_list.append(dis_name_u)
        dis_v_list.append(dis_name_v)

        # 处理不同网络架构的输出
        if network_arch == 'StrainNet_f' and os.path.isfile(dis_name_v):
            displacement_list.append([dis_name_u, dis_name_v])
        else:
            displacement_list.append(dis_name_u)

    return ref_img_list, tar_img_list, displacement_list


def speckle_dataset(train_root, val_root, network_arch):
    train_csv = "/home/dell/DATA/wh/DATASET/Train_annotations_10_30_50.csv"
    val_csv = "/home/dell/DATA/wh/DATASET/Val_annotations_10_30_50.csv"

    train_re_img_list, train_tar_img_list, train_displacement_list = make_dataset(train_root, train_csv, network_arch)
    val_re_img_list, val_tar_img_list, val_displacement_list = make_dataset(val_root, val_csv, network_arch)


    train_dataset = image_dataset(train_root, train_csv, network_arch)
    val_dataset = image_dataset(val_root, val_csv, network_arch)

    return train_dataset,val_dataset
