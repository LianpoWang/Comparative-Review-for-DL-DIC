import time
import glob
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from displacement_model import SwinTransformerSys
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import psutil  # 获取 CPU/GPU 资源占用
from thop import profile
# 定义测试数据集类（与训练时相同）
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

        Ref_name = os.path.join(self.root_dir, self.pair.iloc[idx, 0].strip())
        Def_name = os.path.join(self.root_dir, self.pair.iloc[idx, 1].strip())
        Dispx_name = os.path.join(self.root_dir, self.pair.iloc[idx, 2].strip())
        Dispy_name = os.path.join(self.root_dir, self.pair.iloc[idx, 3].strip())

        try:
            # 读 jpg
            ref_img = Image.open(Ref_name).convert("L")
            tar_img = Image.open(Def_name).convert("L")

            if self.transform:
                ref_img = self.transform(ref_img)
                tar_img = self.transform(tar_img)
            else:
                ref_img = transforms.ToTensor()(ref_img)
                tar_img = transforms.ToTensor()(tar_img)

            # 读 csv displacement
            disp_x = np.genfromtxt(Dispx_name, delimiter=',')
            disp_y = np.genfromtxt(Dispy_name, delimiter=',')

        except Exception as e:
            print(f"\n❌ Error at index {idx}")
            print(f"Ref: {Ref_name}")
            print(f"Tar: {Def_name}")
            print(f"DispX: {Dispx_name}")
            print(f"DispY: {Dispy_name}")
            raise e  # 继续抛出异常，方便 DataLoader 报错

        displacement = torch.stack([
            torch.tensor(disp_x, dtype=torch.float32),
            torch.tensor(disp_y, dtype=torch.float32)
        ], dim=0)

        return ref_img, tar_img, displacement

# 定义评估指标计算函数
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


def calculate_mse(pred, target):
    """计算均方误差(Mean Squared Error)"""
    diff = pred - target
    squared_diff = torch.pow(diff, 2)
    return torch.mean(squared_diff).item()


# 测试函数
def test_model(model, test_loader, device):
    model.eval()
    total_aee = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_samples = 0
    total_inference_time = 0.0  # 总推理时间

    with torch.no_grad():
        for original, transformed, displacement in test_loader:
            original = original.to(device)
            transformed = transformed.to(device)
            displacement = displacement.to(device)

            # ⏱ 推理计时
            start_time = time.time()
            outputs = model(original, transformed)
            end_time = time.time()
            total_inference_time += (end_time - start_time)

            batch_size = original.size(0)
            total_samples += batch_size

            total_aee += calculate_aee(outputs, displacement) * batch_size
            total_mae += calculate_mae(outputs, displacement) * batch_size
            total_mse += calculate_mse(outputs, displacement) * batch_size

    avg_aee = total_aee / total_samples
    avg_mae = total_mae / total_samples
    avg_mse = total_mse / total_samples
    avg_inference_time = total_inference_time / total_samples  # 每对图像平均推理时间

    # 模型复杂度统计
    cpu_usage = psutil.cpu_percent()
    if device.type == 'cuda':
        gpu_memory = torch.cuda.memory_allocated(device) / 1024 ** 2  # MB
    else:
        gpu_memory = 0
    flops, params = profile(model, inputs=(original, transformed), verbose=False)
    print(f"Model Complexity: FLOPs={flops / 1e9:.2f} G\t Params={params / 1e6:.2f} M")

    return avg_aee, avg_mae, avg_mse, avg_inference_time

# 主函数（已修正数据集初始化）
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化模型
    model = SwinTransformerSys().to(device)

    # 加载预训练模型
    checkpoint_path = ""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded pretrained model from {checkpoint_path}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 创建测试数据集（修改：使用新的初始化方式）
    test_dataset = DeformationDataset(
        root_dir="/home/dell/DATA/wh/DATASET/Test10/",  # 修改：只传一个根目录
        csv_file="/home/dell/DATA/wh/DATASET/Test_annotations10_30_50png.csv",
        transform=transform
    )

    # 创建测试数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 运行测试
    aee, mae, mse, avg_time = test_model(model, test_loader, device)

    # 打印结果
    print("\nTest Results:")
    print(f"Average Endpoint Error (AEE): {aee:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Average Inference Time per Image Pair: {avg_time:.6f} s")

if __name__ == '__main__':
    main()