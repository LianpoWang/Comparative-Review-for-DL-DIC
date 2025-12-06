import time
import numpy as np
import psutil
from PIL import Image
from thop import profile
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torchvision
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from util import AverageMeter
from Net.DICNet_corr import DICNet
from dataset1.dataset_real import build_test_dataset1
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 加载网络和预训练权重
net = DICNet().to(device)


weightpath = "/home/dell/DATA/wh/DICNet_Unsupervised/unDICNet_coor/result/DIC_Un1_Epoch_200.pth"
checkpoint = torch.load(weightpath)
net.load_state_dict(checkpoint['model_state_dict'])


test_writer=SummaryWriter()
# 创建输出文件夹
testlog_path = "/home/dell/DATA/wh/DICNet_Unsupervised/unDICNet_coor/result/test_log1/"
os.makedirs(testlog_path, exist_ok=True)
test_writer=SummaryWriter(testlog_path)
test_dataset = build_test_dataset1()
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)

print(f"=>Number of test samples: {len(test_dataset)}")
# 网络推理并保存位移结果
net.eval()
results = []


def pearson_corr_loss(x, y):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)

    cov_xy = torch.sum((x - x_mean) * (y - y_mean))
    std_x = torch.sqrt(torch.sum((x - x_mean) ** 2))
    std_y = torch.sqrt(torch.sum((y - y_mean) ** 2))

    corr_coeff = cov_xy / (std_x * std_y + 1e-6)  # 防止除0错误
    return 1 - corr_coeff


with torch.no_grad():
    MSEs=AverageMeter()
    MAEs=AverageMeter()
    AEEs=AverageMeter()
    Losses=AverageMeter()
    start = time.time()
    for i,(ref,tar,gt_x, gt_y) in enumerate(test_dataloader):  # 假设有10组图像

        ref, tar = ref.to(device), tar.to(device)

        image = torch.cat((ref, tar), dim=1).to(device)

        image = torchvision.transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])(image)
        # 推理
        dis = net(image).cpu()

        # 提取 x 和 y 位移
        pre_x = dis[0][0].numpy()
        pre_y = dis[0][1].numpy()

        # 调整 x_displacement 的形状，确保它和 gt 形状一致
        pre_x = torch.tensor(pre_x, dtype=torch.float32)  # 转换为 Tensor
        pre_y = torch.tensor(pre_y, dtype=torch.float32)  # 转换为 Tensor
        pre_x=pre_x.unsqueeze(0)
        pre_y=pre_y.unsqueeze(0)


        # 计算混合损失
        mse_loss_x = F.mse_loss(pre_x, gt_x)
        mse_loss_y = F.mse_loss(pre_y, gt_y)

        corr_loss_x = pearson_corr_loss(pre_x, gt_x)
        corr_loss_y = pearson_corr_loss(pre_y, gt_y)

        total_loss_x = mse_loss_x + corr_loss_x
        total_loss_y = mse_loss_y + corr_loss_y
        total_loss = (total_loss_x + total_loss_y) / 2
        # 计算 AEE, MAE, MSE
        aee = torch.mean(torch.sqrt((pre_x - gt_x) ** 2 + (pre_y - gt_y) ** 2))

        mae_loss_x = F.l1_loss(pre_x, gt_x)
        mae_loss_y = F.l1_loss(pre_y, gt_y)
        mae = (mae_loss_x + mae_loss_y) / 2

        mse=(mse_loss_x + mse_loss_y) / 2
        test_writer.add_scalar("Test Loss", total_loss.item(), i)
        test_writer.add_scalar("Test AEE", aee.item(), i)
        MAEs.update(mae.item())
        AEEs.update(aee.item())
        MSEs.update(mse.item())
        Losses.update(total_loss.item())
        print(f"Image Pair[{i+1}/{len(test_dataloader)}]] - Loss: {total_loss.item():.3f}  AEE: {aee.item():.3f} ")
    total_time=(time.time()-start)*1000
    print(f"Average MSE:{MSEs.avg:.3f}  AEE:{AEEs.avg:.3f}  Loss:{Losses.avg:.3f}  MAE:{MAEs.avg:.3f}")
    flops, params = profile(net, inputs=(image,))
    print(f"Model Complexity: FLOPs={flops / 1e9:.2f} G\t Params={params / 1e6:.2f} M")
    cpu_usage = psutil.cpu_percent()
    if device.type == 'cuda':
        gpu_memory = torch.cuda.memory_allocated(device) / 1024 ** 2  # MB
    else:
        gpu_memory = 0  # 如果使用 CPU，则 GPU 占用为 0

    print(
        f'Total Inferenve Time:{total_time:.2f}ms\nAverage Inference Time: {total_time / len(test_dataset):.2f} ms\nCPU Usage: {cpu_usage:.2f}%\nGPU Memory: {gpu_memory:.2f} MB')



