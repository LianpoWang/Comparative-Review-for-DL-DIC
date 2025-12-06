import argparse
import os
import time
import psutil
import torch
import pandas as pd
import numpy as np
from thop import profile
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from models.StrainNetF import StrainNet_f
from util import AverageMeter
from multiscaleloss import realEPE, multiscaleEPE
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import models
# 定义模型和测试相关参数
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))
parser = argparse.ArgumentParser(description='StrainNet Testing on speckle dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--arch', default='StrainNet_f', choices=['StrainNet_f', 'StrainNet_h','StrainNet_l'],
                    help='network f or h or l')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size')

parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--pretrained', dest='pretrained', default="/home/dell/DATA/wh/result/StrainNet_f/1/Epoch300.pth",
                    help='path to pre-trained model')
parser.add_argument('--multiscale-weights', '-w', default=[0.005], type=float, nargs=5,
                    help='training weight for each scale, from highest resolution (flow2) to lowest (flow6)',
                    metavar=('W2', 'W3', 'W4', 'W5', 'W6'))
parser.add_argument('--sparse', action='store_true',
                    help='look for NaNs in target flow when computing EPE, avoid if flow is garantied to be dense,'
                         ' ')
parser.add_argument('--div-flow', default=2,
                    help='value by which flow will be divided. Original value is 2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义SpecklesDataset
class SpecklesDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.Speckles_frame = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.Speckles_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        Ref_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 0])
        Def_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 1])
        Dispx_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 2])
        Dispy_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 3])

        Ref = np.genfromtxt(Ref_name, delimiter=',')
        Def = np.genfromtxt(Def_name, delimiter=',')
        Dispx = np.genfromtxt(Dispx_name, delimiter=',')
        Dispy = np.genfromtxt(Dispy_name, delimiter=',')

        Ref = Ref[np.newaxis, ...]
        Def = Def[np.newaxis, ...]
        Dispx = Dispx[np.newaxis, ...]
        Dispy = Dispy[np.newaxis, ...]

        sample = {'Ref': Ref, 'Def': Def, 'Dispx': Dispx, 'Dispy': Dispy}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Normalization(object):
    def __call__(self, sample):
        Ref, Def, Dispx, Dispy = sample['Ref'], sample['Def'], sample['Dispx'], sample['Dispy']
        self.mean = 0.0
        self.std = 255.0
        self.mean1 = -1.0
        self.std1 = 2.0
        return {'Ref': torch.from_numpy((Ref - self.mean) / self.std).float(),
                'Def': torch.from_numpy((Def - self.mean) / self.std).float(),
                'Dispx': torch.from_numpy((Dispx - self.mean1) / self.std1).float(),
                'Dispy': torch.from_numpy((Dispy - self.mean1) / self.std1).float()}


def main():
    global args
    args = parser.parse_args()

    # 数据加载部分
    transform = transforms.Compose([Normalization()])
    test_set = SpecklesDataset(csv_file='/home/dell/DATA/wh/DATASET/Test_annotations1.csv',
                               root_dir='/home/dell/DATA/wh/DATASET/Test1/', transform=transform)
    print('{} test samples '.format(len(test_set)))
    if args.arch == 'StrainNet_l':
        test_loader = DataLoader(test_set, batch_size=48, num_workers=8, pin_memory=True, shuffle=False)
    else:
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=False)

    # 加载模型
    # create model
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        print('=> using pre-trained model'+args.pretrained)
    else:
        network_data = None
        print('creating model')
    print(torch.cuda.is_available())
    model = models.__dict__[args.arch](network_data).cuda()
    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True


    # 进行测试
    with torch.no_grad():  # 禁用梯度计算
        total_time = 0
        start = time.time()
        test(test_loader, model)

        # 测量时间
        end = time.time()
        total_time = (end - start) * 1000
    # 获取系统资源占用情况
    cpu_usage = psutil.cpu_percent()
    if device.type == 'cuda':
        gpu_memory = torch.cuda.memory_allocated(device) / 1024 ** 2  # MB
    else:
        gpu_memory = 0

    print(f'Total Inference Time: {total_time:.2f}ms\nAverage Inference Time: {total_time / len(test_set):.2f} ms\n'
          f'CPU Usage: {cpu_usage:.2f}%\nGPU Memory: {gpu_memory:.2f} MB')

def test(test_loader, model):
    losses=AverageMeter()
    flow2_EPEs = AverageMeter()
    mae_meter = AverageMeter()
    mse_meter = AverageMeter()

    model.eval()

    test_logdir = os.path.join("/home/dell/DATA/wh/result/StrainNet_f/1/", 'test_log')
    if not os.path.exists(test_logdir):
        os.makedirs(test_logdir)
    test_writer = SummaryWriter(test_logdir)
    n_iter=0
    for i, batch in enumerate(test_loader):
        # 准备输入和目标数据，target_x 和 target_y 分别是光流图的 x 和 y 分量，将它们拼接成一个目标张量 target。
        target_x = batch['Dispx'].to(device)
        target_y = batch['Dispy'].to(device)
        target = torch.cat([target_x, target_y], 1).to(device)

        in_ref = batch['Ref'].float().to(device)
        in_ref = torch.cat([in_ref, in_ref, in_ref], 1).to(device)

        in_def = batch['Def'].float().to(device)
        in_def = torch.cat([in_def, in_def, in_def], 1).to(device)

        input = torch.cat([in_ref, in_def], 1).to(device)
        # 如果是 StrainNet_l，则取绿色通道，并拼成 2 通道输入
        if args.arch == 'StrainNet_l':
            # 取绿色通道（索引 1），保持维度 [B, 1, H, W]
            in_ref = in_ref[:, 1:2, :, :]
            in_def = in_def[:, 1:2, :, :]

            # 拼接参考图和变形图，得到 [B, 2, H, W]
            input = torch.cat([in_ref, in_def], 1).to(device)
            input = torch.nn.functional.interpolate(input, size=(7, 7), mode='bilinear', align_corners=False)
            output = model(input)
            output =torch.nn.functional.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)
            loss = multiscaleEPE(output, target, weights=[0.005], sparse=args.sparse)
        else:
            output = model(input)
            loss = multiscaleEPE(output, target, weights=args.multiscale_weights, sparse=args.sparse)
        # 计算多尺度的终点误差 (EPE) 作为损失
        flow2_EPE = args.div_flow * realEPE(output, target)
        mae = torch.mean(torch.abs(output - target))
        mse = torch.mean((output - target) ** 2)
        #更新
        losses.update(loss.item(), target.size(0))
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))
        mae_meter.update(mae.item(), target.size(0))
        mse_meter.update(mse.item(), target.size(0))

        test_writer.add_scalar('Test Loss', loss.item(), n_iter)
        test_writer.add_scalar('Test AEE', flow2_EPE.item(), n_iter)
        n_iter+=1

        if i % args.print_freq == 0:
            print(f'Test: [{i}/{len(test_loader)}]\t Loss:{loss.item():.3f}\tAEE:{flow2_EPE.item():.3f}\t MAE:{mae.item():.3f}\t MSE:{mse.item():.3f}')
    print(f'AEE {flow2_EPEs.avg:.3f}\t MAE: {mae_meter.avg:.3f} \t MSE: {mse_meter.avg:.3f}')
    if isinstance(model, torch.nn.DataParallel):
        model_to_profile = model.module.to(device)  # 把实际模型移到GPU
    else:
        model_to_profile = model.to(device)

    flops, params = profile(model_to_profile, inputs=(input.cuda(),), verbose=False)

if __name__ == '__main__':
    main()
