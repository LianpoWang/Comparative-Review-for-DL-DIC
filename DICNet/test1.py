import argparse
import os
import time
import psutil
import torch
import torch.nn as nn
from thop import profile
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from DICNet import DICNet
from SpeckleDataset.datasets import build_test_dataset1

# 初始化路径和日志
save_path = "/home/dell/DATA/wh/DICNet-Large-deformation/result/"
testlog_path = os.path.join(save_path, "testlog1")
os.makedirs(testlog_path, exist_ok=True)
test_writer = SummaryWriter(testlog_path)
n_iter = 0


class DisplacementMetrics:
    """用于计算位移场评估指标的类"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_samples = 0
        self.total_pixels = 0
        self.total_mae = 0.0
        self.total_mse = 0.0
        self.total_epe = 0.0
        self.total_loss = 0.0

    def update(self, pred_disp, true_disp, loss):
        """更新指标统计"""
        batch_size = pred_disp.size(0)
        num_pixels = pred_disp.numel()

        # 确保位移场是[B,2,H,W]格式
        if pred_disp.dim() != 4 or pred_disp.shape[1] != 2:
            raise ValueError(f"位移场应为[B,2,H,W]格式，实际得到{pred_disp.shape}")

        # 计算MAE和MSE
        self.total_mae += nn.L1Loss(reduction='sum')(pred_disp, true_disp).item()
        self.total_mse += nn.MSELoss(reduction='sum')(pred_disp, true_disp).item()

        # 计算端点误差(EPE)
        epe = torch.norm(pred_disp - true_disp, p=2, dim=1)  # [B,H,W]
        self.total_epe += epe.sum().item()

        self.total_loss += loss.item() * batch_size
        self.total_samples += batch_size
        self.total_pixels += num_pixels

    def compute(self):
        """计算所有指标的平均值"""
        metrics = {
            'loss': self.total_loss / self.total_samples,
            'mae': self.total_mae / self.total_pixels,
            'mse': self.total_mse / self.total_pixels,
            'aee': self.total_epe / (self.total_pixels // 2)  # 因为每个像素有x,y两个分量
        }
        return metrics


def shape_loss(pred_disp, true_disp):
    """形状保持损失"""
    d_max, d_min = true_disp.max(), true_disp.min()
    d_hat_max, d_hat_min = pred_disp.max(), pred_disp.min()
    return torch.mean(
        ((pred_disp - d_hat_min) / (d_hat_max - d_hat_min) -
         (true_disp - d_min) / (d_max - d_min)) ** 2
    )


def absolute_loss(pred_disp, true_disp):
    """绝对位移误差损失"""
    return nn.MSELoss()(pred_disp, true_disp)


def test_model(model, test_loader, device):
    """测试模型主函数"""
    global n_iter
    model.eval()
    metrics = DisplacementMetrics()

    with torch.no_grad():
        start_time = time.time()

        for i, (re_img, tar_img, true_disp) in enumerate(test_loader):
            re_img = re_img.to(device)
            tar_img = tar_img.to(device)
            true_disp = true_disp.to(device)

            # 前向传播
            input_img = torch.cat((re_img, tar_img), dim=1)
            pred_disp = model(input_img)

            # 计算损失
            loss = shape_loss(pred_disp, true_disp) + absolute_loss(pred_disp, true_disp)

            # 更新指标
            metrics.update(pred_disp, true_disp, loss)

            # 记录到TensorBoard
            batch_metrics = metrics.compute()
            test_writer.add_scalar("Test Loss", batch_metrics['loss'], n_iter)
            test_writer.add_scalar("Test AEE", batch_metrics['aee'], n_iter)
            n_iter += 1

            print(f"[Batch {i + 1}/{len(test_loader)}] Loss: {loss.item():.4f} AEE: {batch_metrics['aee']:.4f}")

        # 计算最终指标
        final_metrics = metrics.compute()
        total_time = (time.time() - start_time) * 1000  # 毫秒

        # 打印结果
        print("\n✅ Test Results:")
        print(f"Avg Loss: {final_metrics['loss']:.6f}")
        print(f"Avg AEE: {final_metrics['aee']:.6f}")
        print(f"Avg MAE: {final_metrics['mae']:.6f}")
        print(f"Avg MSE: {final_metrics['mse']:.6f}")

        # 模型复杂度分析
        dummy_input = torch.randn(1, 2, 256, 256).to(device)  # 假设输入尺寸
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        print(f"FLOPs: {flops / 1e9:.2f} G | Params: {params / 1e6:.2f} M")

        # 资源使用情况
        cpu_usage = psutil.cpu_percent()
        gpu_memory = torch.cuda.memory_allocated(device) / 1024 ** 2 if device.type == 'cuda' else 0
        print(f"Total Inference Time: {total_time:.2f} ms")
        print(f"CPU Usage: {cpu_usage:.2f}% | GPU Memory: {gpu_memory:.2f} MB")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='DICNet Testing')
    parser.add_argument('--pretrained', type=str,
                        default="/home/dell/DATA/wh/Large-deformation/pre-trained-models.pth",
                        help='Path to the pre-trained model')
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    args = parser.parse_args()

    # 初始化模型
    model = DICNet().to(device)
    if args.pretrained:
        print(f"=> Loading pre-trained model from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(checkpoint['model'])

    # 准备测试数据
    test_dataset = build_test_dataset1(args)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=not args.distributed,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    print(f"=> Number of test samples: {len(test_dataset)}")

    # 运行测试
    test_model(model, test_loader, device)


if __name__ == "__main__":
    main()