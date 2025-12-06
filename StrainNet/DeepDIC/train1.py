import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torch.nn import init
from torch.utils.tensorboard import SummaryWriter

from torchvision.models.resnet import BasicBlock, ResNet
from torchvision.transforms import ToTensor
import io
from torchvision import models, transforms
import torch.utils.data as data_utils
from PIL import Image
import os
import sys
from collections import OrderedDict

from datetime import datetime

from datasets.dataset import MyDataset
import cv2
import matplotlib.pyplot as plt

import torch.optim as optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable


# ------------------------------
# 工具：保存/加载 Checkpoint（支持 last/best/epoch 存档、pretrained_net、RNG）
# ------------------------------
def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model

def _strip_module_prefix(state_dict):
    new_state = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v
    return new_state

def save_checkpoint(
    save_dir,
    epoch,
    model,
    optimizer,
    scheduler=None,
    scaler=None,
    best_metric=float("inf"),
    n_iter=0,
    metric_name="val_aee",
    filename_prefix="deepdic_1",
    extra_states: dict = None,
):
    os.makedirs(save_dir, exist_ok=True)
    model_to_save = _unwrap_model(model)
    state = {
        "epoch": epoch,
        "model_state": model_to_save.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_metric": float(best_metric),
        "metric_name": metric_name,
        "n_iter": int(n_iter),
        "rng_cpu": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if hasattr(model_to_save, "pretrained_net"):
        state["pretrained_net_state"] = model_to_save.pretrained_net.state_dict()
    if extra_states:
        state.update(extra_states)

    # 1) 按 epoch 存档
    epoch_ckpt = os.path.join(save_dir, f"{filename_prefix}_epoch{epoch}.ckpt")
    torch.save(state, epoch_ckpt)
    # 2) 覆盖式 last
    last_ckpt = os.path.join(save_dir, f"{filename_prefix}_last.ckpt")
    torch.save(state, last_ckpt)
    return epoch_ckpt, last_ckpt

def save_best_checkpoint(save_dir, state, filename_prefix="deepdic_1"):
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt = os.path.join(save_dir, f"{filename_prefix}_best.ckpt")
    torch.save(state, best_ckpt)
    return best_ckpt

def load_checkpoint(
    ckpt_path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location=None,
    strict=False
):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    map_location = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=map_location)

    model_to_load = _unwrap_model(model)

    model_state = checkpoint.get("model_state", {})
    model_state = _strip_module_prefix(model_state)
    missing, unexpected = model_to_load.load_state_dict(model_state, strict=strict)
    if missing or unexpected:
        print(f"[load_checkpoint] missing keys: {missing}")
        print(f"[load_checkpoint] unexpected keys: {unexpected}")

    if hasattr(model_to_load, "pretrained_net") and "pretrained_net_state" in checkpoint:
        try:
            pn_state = _strip_module_prefix(checkpoint["pretrained_net_state"])
            model_to_load.pretrained_net.load_state_dict(pn_state, strict=False)
        except Exception as e:
            print(f"[load_checkpoint] WARN: load pretrained_net_state failed: {e}")

    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        except Exception as e:
            print(f"[load_checkpoint] WARN: load scheduler_state failed: {e}")
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        try:
            scaler.load_state_dict(checkpoint["scaler_state"])
        except Exception as e:
            print(f"[load_checkpoint] WARN: load scaler_state failed: {e}")

    try:
        if checkpoint.get("rng_cpu") is not None:
            torch.set_rng_state(checkpoint["rng_cpu"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
    except Exception as e:
        print(f"[load_checkpoint] WARN: restore RNG failed: {e}")

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    n_iter = int(checkpoint.get("n_iter", 0))
    best_metric = float(checkpoint.get("best_metric", float("inf")))
    metric_name = checkpoint.get("metric_name", "val_aee")

    print(f"[load_checkpoint] Loaded '{ckpt_path}' (epoch {start_epoch-1}), "
          f"{metric_name} best={best_metric:.6f}, n_iter={n_iter}")
    return start_epoch, n_iter, best_metric, metric_name, checkpoint


# ------------------------------
# 你的模型与数据
# ------------------------------
def default_loader(path):
    return Image.open(path)

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, bias=True, transposed=False):
    if transposed:
        layer = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1,
                                   output_padding=1, dilation=dilation, bias=bias)
    else:
        padding = (kernel_size + 2 * (dilation - 1)) // 2
        layer = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                          dilation=dilation, bias=bias)
    if bias and layer.bias is not None:
        init.constant_(layer.bias, 0)  # 修复：使用 constant_
    return layer

def bn(planes):
    layer = nn.BatchNorm2d(planes)
    init.constant_(layer.weight, 1)
    init.constant_(layer.bias, 0)
    return layer


class FeatureResNet(ResNet):
    def __init__(self):
        super().__init__(BasicBlock, [3, 14, 16, 3], 1000)
        self.conv_f = conv(2, 64, kernel_size=3, stride=1)
        self.ReLu_1 = nn.ReLU(inplace=True)
        self.conv_pre = conv(512, 1024, stride=2, transposed=False)
        self.bn_pre = bn(1024)

    def forward(self, x):
        x1 = self.conv_f(x)
        x = self.bn1(x1)
        x = self.relu(x)
        x2 = self.maxpool(x)
        x = self.layer1(x2)
        x3 = self.layer2(x)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.ReLu_1(self.bn_pre(self.conv_pre(x5)))
        return x1, x2, x3, x4, x5, x6

class SegResNet(nn.Module):
    def __init__(self, num_classes, pretrained_net):
        super().__init__()
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.conv3_2 = conv(1024, 512, stride=1, transposed=False)
        self.bn3_2 = bn(512)
        self.conv4 = conv(512, 512, stride=2, transposed=True)
        self.bn4 = bn(512)
        self.conv5 = conv(512, 256, stride=2, transposed=True)
        self.bn5 = bn(256)
        self.conv6 = conv(256, 128, stride=2, transposed=True)
        self.bn6 = bn(128)
        self.conv7 = conv(128, 64, stride=2, transposed=True)
        self.bn7 = bn(64)
        self.conv8 = conv(64, 64, stride=2, transposed=True)
        self.bn8 = bn(64)
        self.conv9 = conv(64, 32, stride=2, transposed=True)
        self.bn9 = bn(32)
        self.convadd = conv(32, 16, stride=1, transposed=False)
        self.bnadd = bn(16)
        self.conv10 = conv(16, num_classes, stride=2, kernel_size=5)
        init.constant_(self.conv10.weight, 0)  # 修复：使用 constant_

    def forward(self, x):
        x1, x2, x3, x4, x5, x6 = self.pretrained_net(x)
        x = self.relu(self.bn3_2(self.conv3_2(x6)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x + x4)))
        x = self.relu(self.bn7(self.conv7(x + x3)))
        x = self.relu(self.bn8(self.conv8(x + x2)))
        x = self.relu(self.bn9(self.conv9(x + x1)))
        x = self.relu(self.bnadd(self.convadd(x)))
        x = self.conv10(x)
        return x

fnet = FeatureResNet()
fcn = SegResNet(2, fnet)
fcn = fcn.cuda()


def adjust_learning_rate(optimizer, epoch, initial_lr):
    """ 在 100 轮后将学习率降低至原来的 1/100 """
    if epoch >= 100:
        new_lr = initial_lr / 100
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


# ------------------------------
# 训练超参 & 数据
# ------------------------------
EPOCH = 300
BATCH_SIZE = 12
print('BATCH_SIZE = ', BATCH_SIZE)
INITIAL_LR = 0.001
NUM_WORKERS = 0

optimizer = optim.Adam(fcn.parameters(), lr=INITIAL_LR, betas=(0.9, 0.999))
loss_func = nn.MSELoss()

train_dataset = MyDataset("/home/dell/DATA/wh/DATASET/Train_annotations_1.csv", '/home/dell/DATA/wh/DATASET/Train1/', 34560)
val_dataset   = MyDataset("/home/dell/DATA/wh/DATASET/Val_annotations_1.csv",   '/home/dell/DATA/wh/DATASET/Train1/', 8640)
print(f'Number of training samples:{len(train_dataset)},validation samples:{len(val_dataset)}')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

dataString = datetime.now().strftime("%Y%m%d")

root_result   = '/home/dell/DATA/wh/DeepDIC/DeepDIC/result/'
model_result  = os.path.join(root_result, 'model1')
trainlog_dir  = os.path.join(root_result, 'train_log1')
vallog_dir    = os.path.join(root_result, 'val_log1')
os.makedirs(root_result, exist_ok=True)
os.makedirs(model_result, exist_ok=True)
os.makedirs(trainlog_dir, exist_ok=True)
os.makedirs(vallog_dir, exist_ok=True)

train_writer = SummaryWriter(log_dir=trainlog_dir)
val_writer   = SummaryWriter(log_dir=vallog_dir)


# ------------------------------
# 断点续训设置
# ------------------------------
metric_name = "val_aee"   # 以验证 AEE 更小为更好
best_metric = float("inf")
n_iter = 0
start_epoch = 0

# 手动指定 ckpt 路径续训（可选）
resume_path = None  # 例如：os.path.join(model_result, "deepdic_1_last.ckpt")

# 或自动从 last 续训（若存在）
AUTO_RESUME_LAST = False
auto_last_path = os.path.join(model_result, "deepdic_1_last.ckpt")

if resume_path is None and AUTO_RESUME_LAST and os.path.exists(auto_last_path):
    resume_path = auto_last_path

if resume_path is not None:
    try:
        start_epoch, n_iter, best_metric, metric_name, _ = load_checkpoint(
            resume_path, fcn, optimizer=optimizer, scheduler=None, scaler=None, strict=False
        )
    except Exception as e:
        print(f"[resume] WARN: failed to load checkpoint: {e}")
        start_epoch = 0


# ------------------------------
# 训练循环
# ------------------------------
for epoch in range(start_epoch, EPOCH):
    adjust_learning_rate(optimizer, epoch, INITIAL_LR)

    fcn.train()
    total_loss = 0.0
    total_aee = 0.0
    num_batches = len(train_loader)

    for i, (img, gt) in enumerate(train_loader):
        img = Variable(img).cuda()
        gt = Variable(gt.float()).cuda()

        output = fcn(img)
        loss = loss_func(output, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # AEE: (B, C, H, W) -> 按通道范数再整体均值
        aee = torch.norm(output - gt, dim=1).mean().item()
        total_aee += aee

        train_writer.add_scalar('Loss1 per Batch', loss.item(), n_iter)
        train_writer.add_scalar('AEE1 per Batch',  aee,         n_iter)
        print(f"DeepDIC_1=>Epoch[{epoch + 1}/{EPOCH}] Batch[{i + 1}/{len(train_loader)}] - Loss: {loss.item():.3f}  AEE: {aee:.3f}")
        n_iter += 1

    avg_loss = total_loss / num_batches
    avg_aee  = total_aee  / num_batches
    print(f"DeepDIC_1=>Epoch {epoch+1} finished, Average Loss: {avg_loss:.3f}, Average AEE: {avg_aee:.3f}")

    train_writer.add_scalar('Loss1 per Epoch', avg_loss, epoch)
    train_writer.add_scalar('AEE1 per Epoch',  avg_aee,  epoch)

    # ---------- 验证 ----------
    fcn.eval()
    val_total_loss = 0.0
    val_total_aee  = 0.0
    with torch.no_grad():
        for step, (img, gt) in enumerate(val_loader):
            img = Variable(img).cuda()
            gt  = Variable(gt.float()).cuda()
            output = fcn(img)
            val_total_loss += loss_func(output, gt).item()
            val_total_aee  += torch.norm(output - gt, dim=1).mean().item()

    avg_val_loss = val_total_loss / len(val_loader)
    avg_val_aee  = val_total_aee  / len(val_loader)
    val_writer.add_scalar("Val Loss1 per Epoch", avg_val_loss, epoch)
    val_writer.add_scalar("Val AEE1 per Epoch",  avg_val_aee,  epoch)
    print(f"Validation Loss1 at Epoch {epoch+1}: {avg_val_loss:.3f} AEE: {avg_val_aee:.3f}")

    # ---------- 保存 ckpt ----------
    # 每个 epoch 都保存：epoch 存档 + last 覆盖
    epoch_ckpt, last_ckpt = save_checkpoint(
        save_dir=model_result,
        epoch=epoch+1,
        model=fcn,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        best_metric=best_metric,
        n_iter=n_iter,
        metric_name=metric_name,
        filename_prefix="deepdic_1",
        extra_states={"avg_val_loss": avg_val_loss, "avg_val_aee": avg_val_aee}
    )
    print(f"[checkpoint] saved epoch: {epoch_ckpt} (also updated last)")

    # 根据验证 AEE 刷新 best
    current_metric = avg_val_aee
    if current_metric < best_metric:
        best_metric = current_metric
        best_path = save_best_checkpoint(model_result, torch.load(last_ckpt), filename_prefix="deepdic_1")
        print(f"[checkpoint] NEW BEST {metric_name}={best_metric:.6f} @ epoch {epoch+1} -> {best_path}")

    # （保留你的早停逻辑）
    if epoch >= 200 and avg_val_aee < 0.01:
        print(f"Early stopping at Epoch {epoch+1}, as validation error stabilized below 0.01")
        break

# 关闭 SummaryWriter
train_writer.close()
val_writer.close()
#从最后一次训练继续：把 resume_path = os.path.join(model_result, "deepdic_1_last.ckpt")

#从最佳模型继续或微调：resume_path = os.path.join(model_result, "deepdic_1_best.ckpt")

#想自动续训：把 AUTO_RESUME_LAST = True 并确保 deepdic_1_last.ckpt 存在