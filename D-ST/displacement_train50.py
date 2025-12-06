import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import timm
import scipy.io as sio
import scipy
from timm.models.swin_transformer import SwinTransformer
import random
import csv  # For saving loss history to CSV
#from Swin_Unet_disp_22222 import SwinTransformerSys
#from displacement_model_channel6 import SwinTransformerSys
from displacement_model import SwinTransformerSys
import pandas as pd
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 设置全局随机种子以确保完全可重复性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


class DeformationDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.pair=pd.read_csv(csv_file,header=None)
        self.csv_file=csv_file
        self.root_dir = root_dir  # 只保留一个 root_dir 参数
        self.transform = transform
    def __len__(self):
        return len(self.pair)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        Ref_name = os.path.join(self.root_dir, self.pair.iloc[idx, 0])
        Def_name = os.path.join(self.root_dir, self.pair.iloc[idx, 1])
        Dispx_name = os.path.join(self.root_dir, self.pair.iloc[idx, 2])
        Dispy_name = os.path.join(self.root_dir, self.pair.iloc[idx, 3])

        # 用 numpy 读
        ref_img = np.genfromtxt(Ref_name, delimiter=',')
        tar_img = np.genfromtxt(Def_name, delimiter=',')
        disp_x = np.genfromtxt(Dispx_name, delimiter=',')
        disp_y = np.genfromtxt(Dispy_name, delimiter=',')

        h, w = ref_img.shape
        assert tar_img.shape == (h, w)
        assert disp_x.shape == (h, w)
        assert disp_y.shape == (h, w)

        # 转成 float32 Tensor，加 channel 维度
        ref_img = torch.tensor(ref_img, dtype=torch.float32).unsqueeze(0)  # [1,H,W]
        tar_img = torch.tensor(tar_img, dtype=torch.float32).unsqueeze(0)  # [1,H,W]

        displacement = torch.stack([
            torch.tensor(disp_x, dtype=torch.float32),
            torch.tensor(disp_y, dtype=torch.float32)
        ], dim=0)  # [2,H,W]

        return ref_img, tar_img, displacement


# Enhanced transform with data augmentation (optional)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
# Initialize the full dataset
full_dataset = DeformationDataset(
    root_dir="/home/dell/DATA/wh/DATASET/Train50/",
    csv_file="/home/dell/DATA/wh/DATASET/Train_annotations10_30_50.csv",
    transform=transform
)


# Calculate split sizes
train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size

# 设置固定的随机种子以确保数据集划分的一致性
seed = 42
generator = torch.Generator().manual_seed(seed)

# Split the dataset with the fixed random seed
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)

# Data loaders
batch_size = 8
num_workers = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

save_dir ='/home/dell/DATA/wh/D-ST_and_S-ST-main/D-ST_and_S-ST-main/result50/'
os.makedirs(save_dir, exist_ok=True)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


# 新增：分布式清理函数
def cleanup():
    dist.destroy_process_group()


# 修改训练函数以支持分布式
def calculate_aee(pred, target):
    """计算平均端点误差(Average Endpoint Error)"""
    diff = pred - target
    squared_diff = torch.pow(diff, 2)
    sum_squared_diff = torch.sum(squared_diff, dim=1)  # 在通道维度求和
    epe = torch.sqrt(sum_squared_diff)  # 欧式距离
    return torch.mean(epe).item()


def calculate_mae(pred, target):
    """计算平均绝对误差(Mean Absolute Error)"""
    diff = torch.abs(pred - target)
    return torch.mean(diff).item()


def train_with_scheduler(rank, world_size, model, train_dataset, test_dataset, criterion, optimizer, scheduler,
                         num_epochs=15):
    setup(rank, world_size)

    # 初始化最佳损失为无穷大
    best_val_loss = float('inf')

    # 模型移到当前GPU并包装为DDP
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # 使用DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        sampler=train_sampler,
        num_workers=16,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        sampler=test_sampler,
        num_workers=16,
        pin_memory=True
    )

    # 只在主进程保存模型和日志
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        loss_history_path = os.path.join(save_dir, 'metrics_history.csv')
        if not os.path.exists(loss_history_path):
            with open(loss_history_path, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['epoch', 'train_loss', 'train_aee', 'train_mae',
                                 'val_loss', 'val_aee', 'val_mae'])

    for epoch in range(num_epochs):
        # 设置sampler的epoch
        train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        total_aee = 0.0
        total_mae = 0.0

        for i, (original, transformed, displacement) in enumerate(train_loader):
            original = original.to(rank, non_blocking=True)
            transformed = transformed.to(rank, non_blocking=True)
            displacement = displacement.to(rank, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(original, transformed)
            loss = criterion(outputs, displacement)
            loss.backward()
            optimizer.step()

            # 计算AEE和MAE
            aee = calculate_aee(outputs, displacement)
            mae = calculate_mae(outputs, displacement)

            total_loss += loss.item()
            total_aee += aee
            total_mae += mae

            if i % 100 == 0 and rank == 0:
                print(f'Rank {rank}, Epoch [{epoch + 1}/{num_epochs}], '
                      f'Batch [{i}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, AEE: {aee:.4f}, MAE: {mae:.4f}')

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_aee = 0.0
        val_mae = 0.0

        with torch.no_grad():
            for original, transformed, displacement in test_loader:
                original = original.to(rank)
                transformed = transformed.to(rank)
                displacement = displacement.to(rank)
                outputs = model(original, transformed)

                val_loss += criterion(outputs, displacement).item()
                val_aee += calculate_aee(outputs, displacement)
                val_mae += calculate_mae(outputs, displacement)

        # 收集所有进程的指标
        metrics = {
            'train_loss': torch.tensor(total_loss / len(train_loader)).to(rank),
            'train_aee': torch.tensor(total_aee / len(train_loader)).to(rank),
            'train_mae': torch.tensor(total_mae / len(train_loader)).to(rank),
            'val_loss': torch.tensor(val_loss / len(test_loader)).to(rank),
            'val_aee': torch.tensor(val_aee / len(test_loader)).to(rank),
            'val_mae': torch.tensor(val_mae / len(test_loader)).to(rank)
        }

        # 对所有指标进行reduce操作
        for key in metrics:
            dist.reduce(metrics[key], dst=0, op=dist.ReduceOp.SUM)
            if rank == 0:
                metrics[key] = metrics[key].item() / world_size

        # 主进程记录结果
        if rank == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Train Loss: {metrics["train_loss"]:.4f}, AEE: {metrics["train_aee"]:.4f}, MAE: {metrics["train_mae"]:.4f} | '
                  f'Val Loss: {metrics["val_loss"]:.4f}, AEE: {metrics["val_aee"]:.4f}, MAE: {metrics["val_mae"]:.4f}')

            # 保存指标到CSV
            with open(loss_history_path, mode='a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    epoch + 1,
                    metrics["train_loss"],
                    metrics["train_aee"],
                    metrics["train_mae"],
                    metrics["val_loss"],
                    metrics["val_aee"],
                    metrics["val_mae"]
                ])

            # 保存最佳模型
            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'val_aee': metrics["val_aee"],
                    'val_mae': metrics["val_mae"]
                }, os.path.join(save_dir, f'best_model.pth'))

        # 保存检查点
        if rank == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'val_aee': metrics.get("val_aee", 0),
                'val_mae': metrics.get("val_mae", 0)
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

        scheduler.step()
        dist.barrier()

    cleanup()


# 主函数
# ... (前面的代码保持不变)

def main():
    # 初始化模型和数据集
    model = SwinTransformerSys()
    full_dataset = DeformationDataset(
        root_dir="/home/dell/DATA/wh/DATASET/Train50/",
        csv_file="/home/dell/DATA/wh/DATASET/Train_annotations10_30_50.csv",
        transform=transform
    )


    # 划分数据集
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size],
                                               generator=torch.Generator().manual_seed(42))

    # 初始化优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    # 分布式训练参数
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs")

    # 使用多进程启动训练
    mp.spawn(
        train_with_scheduler,
        args=(world_size, model, train_dataset, test_dataset,
              nn.MSELoss(),
              optimizer,
              scheduler,
              300),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
