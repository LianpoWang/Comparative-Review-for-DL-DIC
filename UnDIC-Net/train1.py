# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys
import scipy.io as scio
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import torch.nn.functional as F  # 在文件头部添加


from torch.utils.tensorboard import SummaryWriter

from datasets.datasets import  dicDataset
from model.Un_DICnet import UnDICnet_D
import os
import shutil
from tqdm import tqdm
import torch
import cv2 as cv
import numpy as np
import pandas as pd

arg = {
            'smooth_level': 'final',  # final or 1/4
            'smooth_type': 'edge',  # edge or delta
            'smooth_order_1_weight': 0.5,
            'smooth_order_2_weight':0,
            'photo_loss_type': 'Pach_ZNSSD',  # abs_robust, charbonnier,L1, SSIM,Pach_ZNSSD,ZNSSD
            'photo_loss_delta': 0.4,
            'photo_loss_census_weight': 0.5,
        }
best_EPE = -1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0 ,1,2,3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
class Loss_manager():
    def __init__(self):
        pass
    def fetch_loss(self, loss, loss_dict, name,  short_name=None):
        if name not in loss_dict.keys():
            pass
        elif loss_dict[name] is None:
            pass
        else:
            this_loss = loss_dict[name].mean()
            loss = loss + this_loss
        return loss
    def compute_loss(self, loss_dict):
        loss = 0
        loss = self.fetch_loss(loss=loss, loss_dict=loss_dict, name='photo_loss', short_name='ph')
        loss = self.fetch_loss(loss=loss, loss_dict=loss_dict, name='smooth_loss', short_name='sm')
        loss = self.fetch_loss(loss=loss, loss_dict=loss_dict, name='census_loss', short_name='cen')

        return loss
def _to_bchw2(x, device):
    """
    将 numpy 或 torch 张量统一成 (B, 2或1, H, W)，保留 dtype=float32
    """
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x).float().to(device)
    else:
        t = x.float().to(device)

    if t.dim() == 4 and t.shape[1] in (1, 2):  # (B,C,H,W)
        return t
    if t.dim() == 4 and t.shape[-1] in (1, 2):  # (B,H,W,C)
        return t.permute(0, 3, 1, 2).contiguous()
    if t.dim() == 3 and t.shape[0] in (1, 2):   # (C,H,W)
        return t.unsqueeze(0)
    if t.dim() == 3 and t.shape[-1] in (1, 2):  # (H,W,C)
        return t.permute(2, 0, 1).unsqueeze(0).contiguous()
    if t.dim() == 2:                             # (H,W)
        return t.unsqueeze(0).unsqueeze(0)
    if t.dim() == 1:                             # (N,) 不支持位移图
        raise ValueError("Unexpected 1D tensor for flow/disp.")
    return t  # 兜底

def _extract_pred_uv(output, device):
    cand = ['flow_f_out', 'uv', 'flow', 'disp', 'pred', 'prediction', 'pred_uv', 'out']
    pred = None
    for k in cand:
        if isinstance(output, dict) and k in output:
            v = output[k]
            if isinstance(v, (list, tuple)): v = v[-1]
            pred = v
            break
    if pred is None: return None
    pred = _to_bchw2(pred, device)
    if pred.shape[1] != 2: return None
    return pred
def _extract_gt_uv(batch, device):
    """
    用 dispx/dispy 组装 GT 位移 (B,2,H,W)。不再处理 *_croped。
    """
    if 'dispx' not in batch or 'dispy' not in batch:
        return None
    dx = batch['dispx']
    dy = batch['dispy']

    # 统一到 (B,1,H,W)
    def to_b1hw(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(device=device, dtype=torch.float32)
        if x.dim() == 2:              # (H,W)
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:            # (B,H,W)
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.size(1) == 1:
            pass
        else:
            # 其他情况按需扩/permute，这里假定数据经过前面的 dic() 已经标准化
            pass
        return x

    dx = to_b1hw(dx)
    dy = to_b1hw(dy)
    return torch.cat([dx, dy], dim=1)  # (B,2,H,W)
def _resize_like(pred_bchw, gt_bchw):
    """
    将预测位移 pred 插值到 GT 尺寸，并按比例缩放分量：
    u *= (W_gt / W_pred), v *= (H_gt / H_pred)
    """
    H0, W0 = pred_bchw.shape[-2:]
    H1, W1 = gt_bchw.shape[-2:]
    if (H0, W0) != (H1, W1):
        pred_bchw = F.interpolate(pred_bchw, size=(H1, W1),
                                  mode='bilinear', align_corners=True)
        sx = W1 / float(W0)
        sy = H1 / float(H0)
        pred_bchw[:, 0, :, :] *= sx  # u 分量按宽比例缩放
        pred_bchw[:, 1, :, :] *= sy  # v 分量按高比例缩放
    return pred_bchw
def _torch_aee(pred_bchw, gt_bchw):
    """
    AEE：mean( sqrt( (du)^2 + (dv)^2 ) )，自动忽略非有限值。
    输入均为 (B,2,H,W)
    """
    # 有些数据可能包含 NaN/Inf；仅在有效像素上求平均
    valid = torch.isfinite(gt_bchw).all(dim=1, keepdim=False) & \
            torch.isfinite(pred_bchw).all(dim=1, keepdim=False)  # (B,H,W)

    diff = pred_bchw - gt_bchw
    epe = torch.linalg.norm(diff, dim=1)  # (B,H,W)

    if valid.any():
        return epe[valid].mean().item()
    else:
        return float('nan')

start_epoch = 0
num_epochs=300
best_loss = -1
batch_size = 6   #6
num_workers = 1
n_iter=0
n_iter_val=0
save_path = '/data/wlp/wh/undicnet/result/'
model_path=os.path.join(save_path,'model1')
trainlog_path=os.path.join(save_path, 'train_log1')
vallog_path=os.path.join(save_path, 'val_log1')
os.makedirs(trainlog_path, exist_ok=True)
os.makedirs(vallog_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)
train_writer = SummaryWriter(trainlog_path)
val_writer = SummaryWriter(vallog_path)

train_data = dicDataset(csv_file="/home/wlp/DATA/wh/DATASET/Train_annotations_1.csv",
                        root_dir="/data/wlp/wh/DATASET/Train1/")
val_data = dicDataset(csv_file="/home/wlp/DATA/wh/DATASET/Val_annotations_1.csv",
                        root_dir="/data/wlp/wh/DATASET/Train1/")

print('{} samples found, {} train samples and {} test samples '.format(len(train_data) + len(val_data),
                                                                           len(train_data),
                                                                           len(val_data)))
train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
epoch_size = len(train_loader)

def save_checkpoint(state, is_best, save_path, filename):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))


import torch

def dic(batch):
    import torch
    def to_tensor(x):
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def maybe_cat(x):
        if isinstance(x, (list, tuple)):
            return torch.cat([to_tensor(t) for t in x], dim=0)
        return to_tensor(x)
    def ensure_b1hw(x: torch.Tensor):
        if x.dim() == 2: x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3: x = x.unsqueeze(1)
        return x.contiguous().float()

    out = {}
    for k in ['Ref', 'Def', 'start', 'dispx', 'dispy']:
        if k in batch:
            out[k] = ensure_b1hw(maybe_cat(batch[k]))
    return out


def train(train_loader, model, optimizer, epoch,batchsize, scheduler):
    losslist=[]
    aeelist=[]
    model.train()
    # model.training = True
    losses = AverageMeter()
    loss_manager = Loss_manager()
    AEEs = AverageMeter()
    for i, batch in enumerate(train_loader):
        global n_iter
        batch = dic(batch)
        output = model(batch, True)

        optimizer.zero_grad()
        loss = loss_manager.compute_loss(loss_dict=output)

        # 假设 ref 是真实位移场，def_img 是预测位移场
        #aee,mae,mse = calculate_epe(def, ref)

        # aeelist.append(aee)
        # AEEs.update(aee)
        losslist.append(loss.item())
        losses.update(loss.item(), batchsize * 4)
        loss.backward()
        optimizer.step()
        scheduler.step()

        aee_val = None
        pred_uv = _extract_pred_uv(output, device)  # (B,2,H,W) or None
        gt_uv = _extract_gt_uv(batch, device)  # (B,2,H,W) or None
        if (pred_uv is not None) and (gt_uv is not None):
            pred_uv = _resize_like(pred_uv, gt_uv)
            try:
                aee_val = _torch_aee(pred_uv, gt_uv)
                aeelist.append(aee_val)
                AEEs.update(aee_val, batchsize * 4)
                train_writer.add_scalar("AEE1 per Batch", aee_val, n_iter)
            except Exception as e:
                print(f"[Warn] train AEE failed: {e}")
        train_writer.add_scalar("Loss1 per Batch", loss.item(),n_iter)
        n_iter+=1
        print(f'UnDIC1=>Epoch[{epoch + 1}/{num_epochs}] Batch[{i + 1}/{epoch_size}] '
              f'- Loss:{loss.item():.3f}  AEE:{aee_val}')
    infodic={"loss":losslist,"lossavg":losses.avg,"AEE": aeelist, "AEEavg": (AEEs.avg if aeelist else None)}
    # return losslist,epelist, losses.avg, EPEs.avg
    return infodic


def validate(val_loader, model, epoch, batchsize ):
    global n_iter_val
    losslist, aeelist = [], []
    batch_time = AverageMeter()
    loss_manager, losses, AEEs = Loss_manager(), AverageMeter(), AverageMeter()

    model.eval()
    epoch_size = len(val_loader)
    end = time.time()

    for i, batch in enumerate(val_loader):
        batch = dic(batch )
        with torch.no_grad():
            output = model(batch, True)
            loss = loss_manager.compute_loss(loss_dict=output)

        losslist.append(loss.item())
        losses.update(loss.item(), batchsize * 4)

        aee_val = None
        pred_uv = _extract_pred_uv(output, device)
        gt_uv   = _extract_gt_uv(batch, device)
        if (pred_uv is not None) and (gt_uv is not None):
            pred_uv = _resize_like(pred_uv, gt_uv)
            try:
                aee_val = _torch_aee(pred_uv, gt_uv)
                aeelist.append(aee_val)
                AEEs.update(aee_val, batchsize * 4)
                val_writer.add_scalar("AEE1 per Batch", aee_val, n_iter_val)
            except Exception as e:
                print(f"[Warn] val AEE failed: {e}")

        batch_time.update(time.time() - end)
        end = time.time()

        msg_aee = f"{aee_val:.4f}" if aee_val is not None else "n/a"
        print(f'UnDIC1=>Epoch[{epoch + 1}/{num_epochs}] Batch[{i + 1}/{epoch_size}] '
              f'- Loss:{loss.item():.3f}  AEE:{msg_aee}')

        val_writer.add_scalar("Loss1 per Batch", loss.item(), n_iter_val)
        n_iter_val += 1
    infodic={"loss":losslist,"lossavg":losses.avg,"AEE": aeelist, "AEEavg": (AEEs.avg if aeelist else None)}
    # return losslist,epelist, losses.avg, EPEs.avg
    return infodic
def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cuda"):
    print(f"=> Loading checkpoint {path}")
    checkpoint = torch.load(path, map_location=device)

    # 处理 DataParallel 情况
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)

    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    start_epoch = checkpoint.get("epoch", 0)  # 不额外 +1
    best_loss = checkpoint.get("best_EPE", float("inf"))

    print(f"=> Loaded checkpoint (epoch {start_epoch}, best_EPE {best_loss:.4f})")
    return start_epoch, best_loss



def main():
    best_loss=-1
    model = UnDICnet_D(args=arg).to(device)
    model = torch.nn.DataParallel(model, device_ids=[0]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=.00005, eps=1e-8)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.0002, 500000,pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.0002,
        epochs=300,
        steps_per_epoch=len(train_loader),
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy='linear'
    )
    traindict = {}
    vaildict = {}
    resume_path = None  # 你要加载的模型路径
    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        start_epoch = load_checkpoint(resume_path, model, optimizer, scheduler, device)
    else:
        print("=> No checkpoint found, training from scratch")

    for epoch in range(start_epoch,num_epochs ):
        # train for one epoch
        dict1 = train(train_loader, model, optimizer, epoch,batch_size, scheduler)
        index = "Epoch" + str(epoch)
        traindict[index] = dict1
        # evaluate on test dataset
        with torch.no_grad():
            dict2= validate(val_loader, model, epoch,batch_size)
        vaildict[index] = dict2
        # 训练完一轮后的记录（你已有的基础上加两行）
        train_writer.add_scalar("Loss1 per Epoch", dict1["lossavg"], epoch)
        val_writer.add_scalar("Loss1 per Epoch", dict2["lossavg"], epoch)
        train_writer.add_scalar("AEE1 per Epoch", dict1["AEEavg"], epoch)
        val_writer.add_scalar("AEE1 per Epoch", dict2["AEEavg"], epoch)

        if best_loss < 0:
            best_loss = dict2["lossavg"]
        is_best = dict2["lossavg"] < best_loss
        best_loss = min(dict2["lossavg"], best_loss)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'best_EPE': best_loss,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, is_best, model_path, f"Epoch{epoch}.pth")


if __name__ == '__main__':
    main()

