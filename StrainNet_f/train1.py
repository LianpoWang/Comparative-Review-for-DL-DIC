import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import models
import pandas as pd
import numpy as np
from multiscaleloss import multiscaleEPE, realEPE
import datetime

from util import AverageMeter, save_checkpoint
'''已经调试好了'''
# 这个文件通常用于模型的训练过程。它可能包括数据加载、模型初始化、训练循环、损失计算、优化器步骤、模型保存等代码逻辑。
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='StrainNet Training on speckle dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--arch', default='StrainNet_f',choices=['StrainNet_f','StrainNet_h','StrainNet_l'],
                    help='network f or h or l')
#parser.add_argument('--arch', default='StrainNet_f', choices=['StrainNet_f', 'StrainNet_h'],
                    #help='network f or h')
parser.add_argument('--solver', default='adam', choices=['adam', 'sgd'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')
parser.add_argument('--multiscale-weights', '-w', default=[0.005, 0.01, 0.02, 0.08, 0.32], type=float, nargs=5,
                    help='training weight for each scale, from highest resolution (flow2) to lowest (flow6)',
                    metavar=('W2', 'W3', 'W4', 'W5', 'W6'))
parser.add_argument('--sparse', action='store_true',
                    help='look for NaNs in target flow when computing EPE, avoid if flow is garantied to be dense,'
                         ' ')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')
parser.add_argument('--div-flow', default=2,
                    help='value by which flow will be divided. Original value is 2')
parser.add_argument('--milestones', default=[40, 80, 120, 160, 200, 240], metavar='N', nargs='*',
                    help='epochs at which learning rate is divided by 2')

best_EPE = -1
n_iter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.version.cuda)  # PyTorch 使用的 CUDA 版本



class SpecklesDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):  # 初始化

        self.Speckles_frame = pd.read_csv(csv_file,header=None)#从 CSV 文件读取数据
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):  # 返回数据集中样本的数量
        return len(self.Speckles_frame)

    def __getitem__(self, idx):  # 获取指定索引的数据，索引是idx
        if torch.is_tensor(idx):
            idx = idx.tolist()  # 如果 idx 是张量，则转换为列表
        # ref和tar是图像转化为
        # 构建文件路径
        Ref_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 0])
        Def_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 1])
        Dispx_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 2])
        Dispy_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 3])
        # 从csv文件中读取数据，并存在四个数组中
        Ref = np.genfromtxt(Ref_name, delimiter=',')
        Def = np.genfromtxt(Def_name, delimiter=',')
        Dispx = np.genfromtxt(Dispx_name, delimiter=',')
        Dispy = np.genfromtxt(Dispy_name, delimiter=',')
        # 将数组的维度扩展至 (1, H, W) 的形式
        Ref = Ref
        Def = Def
        Dispx = Dispx
        Dispy = Dispy
        # 构建样本字典，通过 np.newaxis 为每个数组增加一个维度，使其形状从 (H, W) 变为 (1, H, W)
        Ref = Ref[np.newaxis, ...]
        Def = Def[np.newaxis, ...]
        Dispx = Dispx[np.newaxis, ...]
        Dispy = Dispy[np.newaxis, ...]

        # 创建字典sample，里面有很多键值对
        sample = {'Ref': Ref, 'Def': Def, 'Dispx': Dispx, 'Dispy': Dispy, 'Def_name': Def_name, 'Ref_name': Ref_name}
        # 如果有定义好的transform类型，见init函数，则为sample应用已定义转化，
        #if self.transform:
            #sample = self.transform(sample)

        return sample


# 将NumPy数组转换为 PyTorch张量的功能，并同时执行了归一化操作
class Normalization(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):  # call使得实例化后的对象可以像函数一样被调用
        Ref, Def, Dispx, Dispy = sample['Ref'], sample['Def'], sample['Dispx'], sample['Dispy']
        # 归一化公式 x'=(x-mean)/std
        self.mean = 0.0
        self.std = 255.0  # 这意味着图像中的每个像素值（通常在 0 到 255 之间）将被缩放到 0 到 1 之间
        self.mean1 = -1.0
        self.std1 = 2.0
        # 这段代码对提取的数据进行归一化处理，并将 NumPy 数组转换为 PyTorch 张量。
        return {'Ref': torch.from_numpy((Ref - self.mean) / self.std).float(),
                'Def': torch.from_numpy((Def - self.mean) / self.std).float(),
                'Dispx': torch.from_numpy((Dispx - self.mean1) / self.std1).float(),
                'Dispy': torch.from_numpy((Dispy - self.mean1) / self.std1).float()}


def main():
    global args, best_EPE,save_path,model_path
    args = parser.parse_args()

    save_path = '{},{},{}epochs{},b{},lr{}'.format(  # 生成一个字符串 save_path，用来表示保存文件的路径
        args.arch,  # 模型架构
        args.solver,  # 求解器/优化算法类型如adam、sgd
        args.epochs,  # 训练轮数
        ',epochSize' + str(args.epoch_size) if args.epoch_size > 0 else '',  # 每个epoch大小若>0,则包含在路径中
        args.batch_size,  # 批量大小
        args.lr)  # 学习率

    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_path = os.path.join(save_path, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    model_path= os.path.join(save_path, 'model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    train_writer = SummaryWriter(log_path)

    # Data loading code
    transform = transforms.Compose([Normalization()])
    train_set = SpecklesDataset(csv_file="/home/dell/DATA/wh/DATASET/Train_annotations1.csv",
                                root_dir='/home/dell/DATA/wh/DATASET/Train1', transform=transform)

    print(' {} train samples '.format(len(train_set)))
    if args.arch == 'StrainNet_l':
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=48,
            num_workers=args.workers, pin_memory=True, shuffle=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True, shuffle=True)

    # create model
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        print('=> using pre-trained model')
    else:
        network_data = None
        print('creating model')
    print(torch.cuda.is_available())
    model = models.__dict__[args.arch](network_data).cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # 优化器设置
    assert (args.solver in ['adam', 'sgd'])  # 使用 assert 确保 args.solver 的值只能是 'adam' 或 'sgd'
    print('=> setting {} solver'.format(args.solver))
    # 定义两个参数组，一个用于模型的偏置参数，另一个用于权重参数。weight_decay 用于控制权重衰减（L2正则化）
    param_groups = [{'params': model.module.bias_parameters(), 'weight_decay': args.bias_decay},
                    {'params': model.module.weight_parameters(), 'weight_decay': args.weight_decay}]
    if args.solver == 'adam':  # 根据选择的优化器类型（Adam或SGD）初始化相应的优化器，从前面的参数组中获取梯度更新的参数。
        optimizer = torch.optim.Adam(param_groups, args.lr,
                                     betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr,
                                    momentum=args.momentum)
    # 利用 MultiStepLR 来调整学习率。在指定的 milestones 时间点将学习率衰减到原来的 gamma倍
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)
    # 循环训练
    for epoch in range(args.start_epoch, args.epochs):
        # 训练并获取当前 epoch 的平均损失和 EPE
        mean_loss, mean_epe = train(train_loader, model, optimizer, epoch, train_writer, scheduler)

        # 记录损失和 EPE 到 TensorBoard
        train_writer.add_scalar('Loss per Epoch', mean_loss, epoch)
        train_writer.add_scalar('AEE per Epoch', mean_epe, epoch)

    # 打印当前 epoch 的损失和 EPE
        print(f"Epoch [{epoch+1}] - Mean Loss: {mean_loss:.3f} - Mean AEE: {mean_epe:.4f}")
        if epoch % 10 == 9:  # 每 10 个 epoch 保存一次
            model_save_path = os.path.join(model_path, f'model_epoch_{epoch+1}.pth')
            print(f"=> Saving model to {model_save_path}")
            torch.save({
                'epoch': epoch+1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'div_flow': args.div_flow,
                'optimizer': optimizer.state_dict(),
            },model_save_path)
    print("Training finished!")

def train(train_loader, model, optimizer, epoch, train_writer, scheduler):
    global n_iter, args,save_path
    ave_loss = AverageMeter()
    ave_epe = AverageMeter()

    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    # switch to train mode
    model.train()

    for i, batch in enumerate(train_loader):

        # 准备输入和目标数据，target_x 和 target_y 分别是光流图的 x 和 y 分量，将它们拼接成一个目标张量 target。
        target_x = batch['Dispx'].to(device)
        target_y = batch['Dispy'].to(device)
        target = torch.cat([target_x, target_y], 1).to(device)

        in_ref = batch['Ref'].float().to(device)
        in_ref = torch.cat([in_ref, in_ref, in_ref], 1).to(device)

        in_def = batch['Def'].float().to(device)
        in_def = torch.cat([in_def, in_def, in_def], 1).to(device)
        input = torch.cat([in_ref, in_def], 1).to(device)
        if args.arch == 'StrainNet_l':
            # 取绿色通道（索引 1），保持维度 [B, 1, H, W]
            in_ref = in_ref[:, 1:2, :, :]
            in_def = in_def[:, 1:2, :, :]

            # 拼接参考图和变形图，得到 [B, 2, H, W]
            input = torch.cat([in_ref, in_def], 1).to(device)
            input = torch.nn.functional.interpolate(input, size=(7, 7), mode='bilinear', align_corners=False)
            output = model(input)
            loss=multiscaleEPE(output, target, weights=[0.005, 0.01, 0.02], sparse=args.sparse)

        else:
            output = model(input)
            loss = multiscaleEPE(output, target, weights=args.multiscale_weights, sparse=args.sparse)
        # 计算多尺度的终点误差 (EPE) 作为损失

        epe =  args.div_flow * realEPE(output[0], target, sparse=args.sparse)
        # record loss and EPE
        ave_loss.update(loss.item(),target.size(0))
        ave_epe.update(epe.item(), target.size(0))

        train_writer.add_scalar('Loss per Batch', loss.item(), n_iter)
        train_writer.add_scalar('AEE per Batch', epe.item(), n_iter)

        # compute gradient and do optimization step
        optimizer.zero_grad()  # 清空上一次的梯度
        loss.backward()  # 计算损失对模型参数的梯度
        optimizer.step()  # 根据梯度更新模型参数
        scheduler.step()  # 调整学习率
        print(f"Epoch [{epoch+1}] [{i+1}/{epoch_size}] - Loss: {loss.item():.3f}  AEE: {epe.item():.3f}")
        n_iter += 1

    return ave_loss.avg, ave_epe.avg

if __name__ == '__main__':
    main()
