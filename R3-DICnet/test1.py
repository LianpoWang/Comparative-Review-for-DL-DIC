# This is a sample Python script.
import os

import psutil
import torch
import time

from thop import profile
from torch.utils.tensorboard import SummaryWriter

from model.R3Dicnet import DIC
from Util.util import save_checkpoint, AverageMeter

import torchvision.transforms as transforms
from tqdm import tqdm
from losses import  sequence_loss
import sys
from Dataset.R3DicDataset import R3DicDataset,Normalization
import torch.nn.functional as F
best_EPE = -1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def realEPE(output, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(output, (h,w), mode='bilinear', align_corners=False)
    return EPE(upsampled_output, target, sparse, mean=True)

def EPE(input_flow, target_flow, sparse=False, mean=True):
    EPE_map = torch.norm(target_flow-input_flow,2,1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)
        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size
def lossfun(output, target):
    return EPE(output, target),realEPE(output, target)

save_path = '/home/dell/DATA/wh/R3DICnet/R3DICnet/R3DICnet-master/result/test_log1'
if not os.path.exists(save_path):
    os.makedirs(save_path)
test_writer = SummaryWriter(save_path)

def test(val_loader, model):
    losslist = []
    epelist = []
    infodic = {}
    batch_time = AverageMeter()
    losses= AverageMeter()
    EPEs = AverageMeter()
    MAEs = AverageMeter()
    MSEs = AverageMeter()
    # switch to evaluate mode
    model.eval()
    epoch_size = len(val_loader)
    end = time.time()

    data_loader = tqdm(val_loader, file=sys.stdout,ncols=200)
    n_iter=0
    total_time=0
    start=time.time()
    for i, batch in enumerate(data_loader):

        target_x = batch['Dispx'].to(device)
        target_y = batch['Dispy'].to(device)
        target= torch.cat([target_x, target_y], 1).to(device)
        Ref = batch['Ref'].float().to(device)
        Def = batch['Def'].float().to(device)
        input = torch.cat([Ref, Def], 1).to(device)  # torch.Size([16, 6, 256, 256])
        # compute output
        output = model(Ref,Def,8)

        loss, aee, mae, mse = sequence_loss(output, target)


        losslist.append(loss.item())
        epelist.append(aee.item())

        losses.update(loss.item(), target_x.size(0))
        EPEs.update(aee.item(), target_x.size(0))
        MAEs.update(mae, target_x.size(0))
        MSEs.update(mse, target_x.size(0))
        test_writer.add_scalar('Test Loss', loss.item(), n_iter)
        test_writer.add_scalar('Test AEE', aee.item(), n_iter)
        batch_time.update(time.time() - end)

        end = time.time()
        inference_time = (end - start) * 1000
        total_time += inference_time
        n_iter+=1

        # 获取系统资源占用情况
    cpu_usage = psutil.cpu_percent()
    if device.type == 'cuda':
        gpu_memory = torch.cuda.memory_allocated(device) / 1024 ** 2  # MB
    else:
        gpu_memory = 0

    print(f'AEE {EPEs.avg:.3f}\t MAE: {MAEs.avg:.3f} \t MSE: {MSEs.avg:.3f}')
    flops, params = profile(model, inputs=(input,), verbose=False)
    print(f"Model Complexity: FLOPs={flops / 1e9:.2f} G\t Params={params / 1e6:.2f} M")
    print(
        f'Total Inference Time: {total_time:.2f}ms\nAverage Inference Time: {total_time / len(data_loader):.2f} ms\n'
        f'CPU Usage: {cpu_usage:.2f}%\nGPU Memory: {gpu_memory:.2f} MB')

    return infodic, EPEs.avg

def main():
    batch_size =1
    num_workers = 1

    transform = transforms.Compose([Normalization()])
    test_data = R3DicDataset(csv_file='/home/dell/DATA/wh/DATASET/Test_annotations1.csv',
                               root_dir='/home/dell/DATA/wh/DATASET/Test1/', transform=transform)

    print(' {} test samples '.format(len(test_data)))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,num_workers=num_workers, pin_memory=True, shuffle=True)
    torch.cuda.empty_cache()
    network_data = torch.load("/home/dell/DATA/wh/R3DICnet/R3DICnet/R3DICnet-master/result/model/model_epoch240.pth_checkpoint.pth.tar")
    model = DIC(max_disp=4).to(device)
    # 如果是多卡训练保存的，需要去掉 'module.'
    model.load_state_dict(network_data['state_dict'])
    # 切换到评估模式
    model.eval()

    # 调用 validate 函数执行测试
    with torch.no_grad():
        infodic, avg_epe = test(test_loader, model)


if __name__ == '__main__':
    main()



