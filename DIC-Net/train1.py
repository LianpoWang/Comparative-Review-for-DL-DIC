from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from DICNet.DICNet import DICNet_d
from datasets.dataset import MyDataset

# ======================
# 配置
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHs = 300
BATCH_SIZE = 16
LR = 1e-4
NUM_WORKERS = 4
AUTO_RESUME = True  # ✅ 设置 True 时，自动从 last.ckpt 恢复
pretrained_path = "/home/dell/DATA/wh/DIC-Net/DIC-Net-main/result/model10/last_checkpoint.pth"

# ======================
# 模型 & 优化器
# ======================
model = DICNet_d(2, 2).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # 例子：每 50 epoch lr 减半
loss_func = nn.MSELoss()

# ======================
# 数据
# ======================
train_dataset = MyDataset('/home/dell/DATA/wh/DATASET/Train1', 40800)
print('Number of training images:', len(train_dataset))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# ======================
# 结果目录
# ======================
root_result = '/home/dell/DATA/wh/DIC-Net/DIC-Net-main/result/'
model_result = os.path.join(root_result, 'model1')
log_result = os.path.join(root_result, 'log1')
os.makedirs(model_result, exist_ok=True)
os.makedirs(log_result, exist_ok=True)

writer = SummaryWriter(log_dir=log_result)

# ======================
# 工具类
# ======================
class AverageMeter(object):
    """计算和存储均值"""
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

# ======================
# 加载预训练模型
# ======================
start_epoch = 0
if AUTO_RESUME and os.path.exists(pretrained_path):
    print(f"=> Loading checkpoint from {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch']
    print(f"=> Resumed training from epoch {start_epoch}")
else:
    print("=> No checkpoint found, training from scratch.")

# ======================
# 训练循环
# ======================
for epoch in range(start_epoch, EPOCHs):
    model.train()
    MSEs = AverageMeter()
    AEEs = AverageMeter()

    for step, (img, gt) in enumerate(train_loader):
        img = img.to(device)
        gt = gt.float().to(device)

        output = model(img)
        mseloss = loss_func(output, gt)
        aeeloss = torch.norm(output - gt, dim=1).mean()

        optimizer.zero_grad()
        mseloss.backward()
        optimizer.step()

        MSEs.update(mseloss.item(), BATCH_SIZE)
        AEEs.update(aeeloss.item(), BATCH_SIZE)

        global_step = epoch * len(train_loader) + step
        writer.add_scalar('Train/MSELoss_per_Batch', mseloss.item(), global_step)
        writer.add_scalar('Train/AEE_per_Batch', aeeloss.item(), global_step)

        print(f"DICNet1 => Epoch[{epoch+1}/{EPOCHs}] "
              f"Batch[{step+1}/{len(train_loader)}] - "
              f"MSELoss:{mseloss.item():.4f} AEELoss:{aeeloss.item():.4f}")

    # 每个 epoch 结束后
    scheduler.step()
    writer.add_scalar('Train/MSELoss_per_Epoch', MSEs.avg, epoch)
    writer.add_scalar('Train/AEE_per_Epoch', AEEs.avg, epoch)

    print(f"Epoch {epoch+1} finished => Avg MSE: {MSEs.avg:.4f}, Avg AEE: {AEEs.avg:.4f}")

    # ✅ 保存 checkpoint（last + 每 10 轮一次）
    last_ckpt = os.path.join(model_result, "last_checkpoint.pth")
    torch.save({
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, last_ckpt)

    if epoch % 10 == 9:
        PATH = os.path.join(model_result, f"DICNet1_epoch{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, PATH)
        print(f"=> Saved checkpoint at {PATH}")

writer.close()
