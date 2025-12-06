# This is a sample Python script.

import sys
import psutil
import scipy.io as scio
import torchvision.transforms as transforms
from thop import profile
from torch.utils.data import DataLoader
import time
from torch.utils.tensorboard import SummaryWriter
from datasets.datasets_test1 import Augmentation, dicDataset
from model.Un_DICnet import UnDICnet_D
import os
import shutil
from tqdm import tqdm
import torch
import cv2 as cv
import numpy as np
import pandas as pd


# 参数配置
arg = {
    'smooth_level': 'final',
    'smooth_type': 'edge',
    'smooth_order_1_weight': 0.5,
    'smooth_order_2_weight': 0,
    'photo_loss_type': 'Pach_ZNSSD',
    'photo_loss_delta': 0.4,
    'photo_loss_census_weight': 0.5,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AverageMeter:
    """ 计算和存储平均值和当前值 """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LossManager:
    def fetch_loss(self, loss, loss_dict, name):
        if name in loss_dict and loss_dict[name] is not None:
            loss += loss_dict[name].mean()
        return loss

    def compute_loss(self, loss_dict):
        loss = 0
        for key in ['photo_loss', 'smooth_loss', 'census_loss']:
            loss = self.fetch_loss(loss, loss_dict, key)
        return loss


def compute_aee(flow_pred, flow_gt):
    """
    计算平均终点误差 (AEE)

    Args:
        flow_pred (torch.Tensor): 预测的位移场 (B, 2, H, W)
        flow_gt (torch.Tensor): 真实的位移场 (B, 2, H, W)

    Returns:
        float: 平均终点误差 (AEE)
    """
    # 确保预测和GT的尺寸一致
    assert flow_pred.shape == flow_gt.shape, "预测和GT尺寸不一致！"

    # 计算终点误差
    error = torch.sqrt(torch.sum((flow_pred - flow_gt) ** 2, dim=1))  # (B, H, W)
    aee = torch.mean(error)  # 取均值
    # 计算 MAE (平均绝对误差)
    mae = torch.mean(torch.abs(flow_pred - flow_gt))

    # 计算 MSE (均方误差)
    mse = torch.mean((flow_pred - flow_gt) ** 2)

    return aee.item(), mae.item(), mse.item()


def save_checkpoint(state, is_best, save_path, flag):
    model_path = os.path.join(save_path, "model")
    os.makedirs(model_path, exist_ok=True)
    torch.save(state, os.path.join(model_path, f'{flag}_checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(model_path, f'{flag}_checkpoint.pth.tar'),
                        os.path.join(save_path, f'{flag}_model_best.pth.tar'))


def dic(batch, is_aug):
    keys = ['Ref', 'Def', 'Ref_croped', 'Def_croped', 'start','Dispx','Dispy']
    return {k: torch.cat(batch[k], 0) if is_aug else batch[k] for k in keys}

testlog_dir = "/home/dell/DATA/wh/UnDICnet/result/test_log1/"
test_writer=SummaryWriter(testlog_dir)
os.makedirs(testlog_dir, exist_ok=True)
def test(test_loader, model, batch_size, is_aug):
    loss_manager = LossManager()
    losses = AverageMeter()
    AEEs = AverageMeter()
    MAEs=AverageMeter()
    MSEs=AverageMeter()
    model.eval()
    start_time = time.time()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            model_batch = {
                'Ref': batch['Ref'],
                'Def': batch['Def'],
                'Ref_croped': batch['Ref_croped'],
                'Def_croped': batch['Def_croped'],
                'start': batch['start']
            }

            output = model(model_batch, True)
            loss = loss_manager.compute_loss(output)
            losses.update(loss.item(), batch['Ref'].size(0))

            # 获取预测的位移场 (B, 2, H, W)
            flow_pred = output['flow_f_out'].to(device)

            # 获取真实位移场并拼接成 (B, 2, H, W)
            flow_gt_x = batch['Dispx'].to(device)
            flow_gt_y = batch['Dispy'].to(device)
            flow_gt = torch.stack([flow_gt_x, flow_gt_y], dim=1)

            # 确保尺寸一致
            assert flow_pred.shape == flow_gt.shape, f"Shape mismatch: pred {flow_pred.shape} vs gt {flow_gt.shape}"

            # 计算整 batch 的 AEE、MAE、MSE（矢量化）
            aee, mae, mse = compute_aee(flow_pred, flow_gt)
            AEEs.update(aee, flow_pred.size(0))
            MAEs.update(mae, flow_pred.size(0))
            MSEs.update(mse, flow_pred.size(0))

            # 写入 TensorBoard
            test_writer.add_scalar("Test Loss", losses.avg, i)
            test_writer.add_scalar("Test AEE", AEEs.avg, i)

            print(f'Batch[{i + 1}/{len(test_loader)}] - Loss: {losses.avg:.3f}  AEE: {AEEs.avg:.3f}')

    flops, params = profile(model, inputs=(model_batch,), verbose=False)
    print(f"Model Complexity: FLOPs={flops / 1e9:.2f} G\t Params={params / 1e6:.2f} M")
    total_time = (time.time() - start_time) * 1000

    return {'Loss_avg': losses.avg,'AEE_avg':AEEs.avg, 'MSE_avg':MSEs.avg, 'MAE_avg':MAEs.avg, 'total_time': total_time}


def main():
    batch_size =1
    num_workers = 1
    save_path = '/home/dell/DATA/wh/UnDICnet/result/'
    transform = transforms.Compose([Augmentation(is_Aug=False)])

    test_data = dicDataset(
        csv_file="/home/dell/DATA/wh/DATASET/Test_annotations1.csv", root_dir="/home/dell/DATA/wh/DATASET/Test1/",
        transform=transform)
    print(f'Number of test samples: {len(test_data)}')
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)

    model = UnDICnet_D(args=arg).to(device)
    network_data = torch.load("/home/dell/DATA/wh/UnDICnet/result/model1/model_best.pth.tar", map_location=device)
    model.load_state_dict(network_data['state_dict'])

    print("Loaded model weights from pretrained model")

    results = test(test_loader, model, batch_size, is_aug=False)
    avg_time=results['total_time']/len(test_data)
    print(f"Total Inference Time: {results['total_time']:.2f}ms, Average Inference Time: {avg_time:.2f}ms")
    print(f"Average Loss: {results['Loss_avg']:.3f}, Average AEE: {results['AEE_avg']:.3f}, Average MSE: {results['MSE_avg']:.3f},  Average MAE: {results['MAE_avg']:.3f}")

    cpu_usage = psutil.cpu_percent()
    gpu_memory = torch.cuda.memory_allocated(device) / 1024 ** 2 if device.type == 'cuda' else 0

    print(f"CPU Usage: {cpu_usage:.2f}%\nGPU Memory: {gpu_memory:.2f} MB")


if __name__ == '__main__':
    main()
