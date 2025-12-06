# This is a sample Python script.
import os
import torch
import time
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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
EPOCHs=300
save_path = '/home/dell/DATA/wh/R3DICnet/R3DICnet/R3DICnet-master/result/'
trainlog_path=os.path.join(save_path, 'train_log1')
vallog_path=os.path.join(save_path, 'val_log1')
model_path = os.path.join(save_path, 'model1')
os.makedirs(trainlog_path, exist_ok=True)
os.makedirs(vallog_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)
train_writer=SummaryWriter(trainlog_path)
val_writer=SummaryWriter(vallog_path)
def train(train_loader, model, optimizer, epoch, scheduler):
    losslist=[]
    epelist=[]
    model.train()
    # model.training = True
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    EPEs = AverageMeter()
    epoch_size = len(train_loader)
    # switch to train mode
    end = time.time()

    for i, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target_x = batch['Dispx'].to(device)
        target_y = batch['Dispy'].to(device)
        target = torch.cat([target_x, target_y], 1).to(device)
        Ref = batch['Ref'].float().to(device)
        Def = batch['Def'].float().to(device)
        # Ref = torch.cat([Ref, Ref, Ref], 1).to(device)  # torch.Size([16, 3, 256, 256])
        # Ref = torch.cat([Ref, Ref, Ref], 1).to(device)  # torch.Size([16, 3, 256, 256])
        # Def = torch.cat([Def, Def, Def], 1).to(device)  # torch.Size([16, 3, 256, 256])
        input = torch.cat([Ref , Def], 1).to(device)  # torch.Size([16, 6, 256, 256])

        # compute output
        output = model(Ref,Def,8)
        # 字典中值的长度为2 分别对应光流，size都为torch.Size([2, 2, 384, 512])

        # 字典中值的长度为2 分别对应光流，size都为torch.Size([2, 2, 384, 512])
        loss, aee, mae, mse = sequence_loss(output, target)
        losslist.append(loss.item())
        epelist.append(aee.item())
        losses.update(loss.item(), target_x.size(0))
        # losses.update(loss.item(), target_x.size(0))
        # EPE_ = realEPE(output[0], target)
        EPEs.update(aee.item(), target_x.size(0))
        train_writer.add_scalar('Loss per Batch', loss.item(), i)
        train_writer.add_scalar('AEE per Batch', aee.item(), i)
        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()
        scheduler.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        print(f'Epoch: [{epoch+1}/{EPOCHs}] Batch[{i+1}/{epoch_size}] - Loss: {loss.item():.3f}  AEE: {aee.item():.3f}')
    infodic={"loss":losslist,"epe":epelist,"lossavg":losses.avg,"epeavg":EPEs.avg}
    # return losslist,epelist, losses.avg, EPEs.avg
    return infodic


def main():
    best_EPE = -1
    batch_size =16
    num_workers = 1

    transform = transforms.Compose([Normalization()])
    train_data =R3DicDataset(csv_file='/home/dell/DATA/wh/DATASET/Train_annotations1.csv',
                           root_dir='/home/dell/DATA/wh/DATASET/Train1', transform=transform)


    print(' {} train samples '.format(len(train_data)))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,num_workers=num_workers, pin_memory=True, shuffle=True)


    model=DIC(max_disp=4)
    model.training = True

    model = torch.nn.DataParallel(model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=.00005, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.0002,  # 最高学习率
        epochs=300,  # 训练 300 轮
        steps_per_epoch=len(train_loader),  # 每个 epoch 的 step 数
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy='linear')
    infodict = {}
    traindict = {}
    # # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80,120,160,200,240], gamma=0.5)
    for epoch in range(0, EPOCHs):
        # train for one epoch
        dict1 = train(train_loader, model, optimizer, epoch, scheduler)
        index = "Epoch" + str(epoch)
        traindict[index] = dict1
        # evaluate on test dataset
        train_writer.add_scalar('Loss per Epoch', dict1["lossavg"], epoch)
        train_writer.add_scalar('AEE per Epoch', dict1["epeavg"], epoch)

        # 每个 epoch 保存一次模型
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'R3DICnet',
            'state_dict': model.module.state_dict(),
            'best_EPE': best_EPE,
        }, infodict, False, model_path, f'model_epoch{epoch + 1}.pth')



if __name__ == '__main__':
    main()



