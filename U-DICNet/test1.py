
import argparse
import os
import time

import psutil
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from thop import profile

from multiscaleloss import multiscaleEPE, realEPE,MAE,MSE
import models
import datasets
import numpy as np
import shutil
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

dataset_names = sorted(name for name in datasets.__all__)
best_EPE = -1
n_iter = 0
world_size = torch.cuda.device_count()
parser = argparse.ArgumentParser(description='U-DICNet Training on speckle dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--arch', default='U_DICNet', choices=['StrainNet_f', 'U_DICNet', 'U_StrainNet_f'],
                    help='network selection')
parser.add_argument('--train_dataset_root', '-trr', metavar='DIR',
                    default='/home/dell/DATA/wh/DATASET/Train1',
                    help='path to training dataset')
parser.add_argument('--val_dataset_root', '-var', metavar='DIR',default='/home/dell/DATA/wh/DATASET/Train1/',
                    help='path to validation dataset')
parser.add_argument('--test_dataset_root', '-ter', metavar='DIR',
                    default='/home/dell/DATA/wh/DATASET/Test1',
                    help='path to training dataset')
parser.add_argument('--pretrained', dest='pretrained', default="/home/dell/DATA/wh/U_DICNet-main/U_DICNet-main/1/model_best.pth.tar",
                    help='path to pre-trained model')
parser.add_argument('--solver', default='sgd', choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--milestones', default=[40, 80, 120, 160, 200, 240],
                    metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
# parser.add_argument('--multiscale-weights', '-w', default=[0.12, 0.04, 0.08, 0.01, 0.02], type=float, nargs=5,
#                     help='training weight for each scale, from highest resolution (flow2) to lowest (flow6)',
#                     metavar=('W2', 'W3', 'W4', 'W5', 'W6'))

parser.add_argument('--multiscale-weights', '-w', default=[0.08, 0.02, 0.02, 0.05, 0.24], type=float, nargs=5,
                    help='training weight for each scale, from highest resolution (flow2) to lowest (flow6)',
                    metavar=('W2', 'W3', 'W4', 'W5', 'W6'))  #0.01, 0.02, 0.05, 0.08, 0.24

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class AverageMeter(object):
    """Computes and stores the average and current value"""

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

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)



def main():

    global args, best_EPE,n_iter,test_writer

    args = parser.parse_args()

    save_path = './{}_network_data/{}epochs,b{},lr{}'.format(
        args.arch,
        args.epochs,
        args.batch_size,
        args.lr)

    testlog_path = os.path.join(save_path, 'test_log1')
    if not os.path.exists(testlog_path):
        os.makedirs(testlog_path)
    test_writer = SummaryWriter(testlog_path)
    train_set, val_set, test_set = datasets.__dict__['speckle_dataset'](
        args.train_dataset_root,
        args.val_dataset_root,
        args.test_dataset_root,
        args.arch  # 网络结构参数
    )
    # dataset sampler
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=int(args.batch_size / world_size),
        num_workers=args.workers, pin_memory=True)

    # create model
    if args.pretrained:
        # using pre-trained model
        network_data = torch.load(args.pretrained ,map_location=device, weights_only=False)
        print("=> using pre-trained model ")
        best_EPE = network_data['best_EPE']
    else:
        network_data = None
        print("=> creating new model ")




    # choose the model
    print(torch.cuda.is_available())
    with (torch.no_grad()):
        model = models.__dict__['U_DICNet'](network_data).to(device)  # , drop=False)
        model = model.to(device)
        cudnn.benchmark = True
    EPE,MAE,MSE=test(test_loader, model)
    print(f'AEE {EPE:.3f}\t MAE: {MAE:.3f} \t MSE: {MSE:.3f}')
def test(test_loader, model):
    global args
    batch_time = AverageMeter()
    losses = AverageMeter()
    flow2_EPEs = AverageMeter()
    MAEs = AverageMeter()
    MSEs = AverageMeter()
    total_time=0
    # switch to evaluate mode
    model.eval()
    for i, (input_img, target) in enumerate(test_loader):
        start = time.time()
        target = target.to(device)
        input_img = input_img.to(device)

        # compute output
        output = model(input_img)
        # record

        loss = multiscaleEPE(output, target,rank=0,weights=[0.08])
        flow2_EPE = realEPE(output, target,rank=0)
        mae = MAE(output, target,rank=0)
        mse = MSE(output, target,rank=0)

        losses.update(loss.item(), input_img.size(0))
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))
        MAEs.update(mae.item(), target.size(0))
        MSEs.update(mse.item(), target.size(0))

        test_writer.add_scalar('Test Loss', loss.item(), i)
        test_writer.add_scalar('Test AEE', flow2_EPE.item(), i)


        batch_time.update(time.time() - start)
        print(f'Test1: [{i}/{len(test_loader)}]\t Loss:{loss:.3f}\tAEE:{flow2_EPE:.3f}\t MAE:{mae:.3f}\t MSE:{mse:.3f}')

    # print(' * EPE {:.3f}'.format(flow2_EPEs.avg))
    total_time=batch_time.sum * 1000
    cpu_usage = psutil.cpu_percent()
    if device.type == 'cuda':
        gpu_memory = torch.cuda.memory_allocated(device) / 1024 ** 2  # MB
    else:
        gpu_memory = 0
    total_images = len(test_loader.dataset)
    flops, params = profile(model, inputs=(input_img,), verbose=False)
    print(f"Model Complexity: FLOPs={flops / 1e9:.2f} G\t Params={params / 1e6:.2f} M")
    print(f'Total Inference Time: {total_time:.2f}ms\nAverage Inference Time per Image: {total_time / total_images:.2f} ms\n'
          f'CPU Usage: {cpu_usage:.2f}%\nGPU Memory: {gpu_memory:.2f} MB')

    return flow2_EPEs.avg,MAEs.avg,MSEs.avg

if __name__ == '__main__':
    main()
