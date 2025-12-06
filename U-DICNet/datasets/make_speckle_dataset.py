import csv
from .image_dataset import image_dataset
def make_dataset(csv_file, network_arch):
    """
    Args:
        csv_file: 包含所有文件名的csv文件路径
        network_arch: 网络架构 (StrainNet_f / U_StrainNet_f / U_DICNet)

    Returns:
        re_img_list: 参考图像文件名列表
        tar_img_list: 目标图像文件名列表
        displacement_list: 位移场文件名列表
    """
    re_img_list = []
    tar_img_list = []
    displacement_list = []

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue  # 跳过不完整行
            re_img, tar_img, u_file, v_file = row[0], row[1], row[2], row[3]

            re_img_list.append(re_img)
            tar_img_list.append(tar_img)

            # 根据网络结构决定用哪些位移文件
            if network_arch == 'StrainNet_f':
                displacement_list.append([u_file, v_file])
            else:
                displacement_list.append(u_file)

    return re_img_list, tar_img_list, displacement_list

def speckle_dataset(train_csv_file, val_csv_file, network_arch):

    train_re_img_list, train_tar_img_list, train_displacement_list = make_dataset(train_csv_file, network_arch)
    test_re_img_list, test_tar_img_list, test_displacement_list = make_dataset(val_csv_file, network_arch)

    train_dataset = image_dataset(train_root, train_re_img_list, train_tar_img_list, train_displacement_list, network_arch)
    test_dataset = image_dataset(test_root, test_re_img_list, test_tar_img_list, test_displacement_list, network_arch)
    return train_dataset, test_dataset
