import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from DICNet import DICNet
from SpeckleDataset.datasets import SpeckleDataset

save_path = "/home/dell/DATA/wh/DICNet-Large-deformation/result/"
log_path = os.path.join(save_path, "log1")
if not os.path.exists(log_path):
    os.makedirs(log_path)
model_path = os.path.join(save_path, "model1")
if not os.path.exists(model_path):
    os.makedirs(model_path)


# ================== 定义损失函数 ==================
def shape_loss(pred_disp, true_disp):
    d_max, d_min = true_disp.max(), true_disp.min()
    d_hat_max, d_hat_min = pred_disp.max(), pred_disp.min()
    loss = torch.mean(
        ((pred_disp - d_hat_min) / (d_hat_max - d_hat_min + 1e-8) -
         (true_disp - d_min) / (d_max - d_min + 1e-8)) ** 2
    )
    return loss


def absolute_loss(pred_disp, true_disp):
    return nn.MSELoss()(pred_disp, true_disp)


# ================== 主程序入口 ==================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='DICNet Training')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume checkpoint')
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    args = parser.parse_args()

    # 初始化模型
    model = DICNet().to(device)

    # 构建训练集 & 验证集
    train_dataset = SpeckleDataset(csv_file="/home/dell/DATA/wh/DATASET/Train_annotations_1.csv",root_dir="/home/dell/DATA/wh/DATASET/Train1/")
    val_dataset = SpeckleDataset(csv_file="/home/dell/DATA/wh/DATASET/Val_annotations_1.csv",root_dir="/home/dell/DATA/wh/DATASET/Train1/")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=not args.distributed, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False)

    print(f"=> Number of training samples: {len(train_dataset)}")
    print(f"=> Number of validation samples: {len(val_dataset)}")

    # ========== 初始化优化器 & 调度器 ==========
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    start_epoch = 0
    best_val_loss = float("inf")

    # ========== 如果有断点，加载 ==========
    if args.resume:
        print(f"=> Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float("inf"))
        print(f"=> Resumed at epoch {start_epoch}, best val loss {best_val_loss:.4f}")

    # 训练模型
    train_model(model, train_loader, val_loader, device,
                optimizer, scheduler,
                start_epoch, args.epochs, best_val_loss)


# ================== 训练 & 验证 ==================
def train_model(model, train_loader, val_loader, device,
                optimizer, scheduler, start_epoch, epochs, best_val_loss):

    train_writer = SummaryWriter(log_path)

    for epoch in range(start_epoch, epochs):
        # ---------- Train ----------
        model.train()
        epoch_loss, epoch_epe = 0, 0
        for i, (re_img, tar_img, true_disp) in enumerate(train_loader):
            re_img, tar_img, true_disp = re_img.to(device), tar_img.to(device), true_disp.to(device)

            optimizer.zero_grad()
            input_img = torch.cat((re_img, tar_img), dim=1)
            pred_disp = model(input_img)

            loss_shape = shape_loss(pred_disp, true_disp)
            loss_error = absolute_loss(pred_disp, true_disp)
            loss_total = loss_shape + loss_error
            epe = torch.norm(pred_disp - true_disp, p=2, dim=1).mean().item()

            loss_total.backward()
            optimizer.step()

            epoch_loss += loss_total.item()
            epoch_epe += epe

            train_writer.add_scalar("Train/Loss_iter", loss_total.item(), epoch * len(train_loader) + i)
            train_writer.add_scalar("Train/AEE_iter", epe, epoch * len(train_loader) + i)

            print(f"[Train] Epoch [{epoch + 1}/{epochs}] "
                  f"Batch [{i + 1}/{len(train_loader)}] "
                  f"Loss: {loss_total.item():.4f}  AEE: {epe:.4f}")
        avg_loss = epoch_loss / len(train_loader)
        avg_epe = epoch_epe / len(train_loader)

        train_writer.add_scalar("Train/Loss_epoch", avg_loss, epoch)
        train_writer.add_scalar("Train/AEE_epoch", avg_epe, epoch)
        print(f"[Train] Epoch {epoch+1}/{epochs} Avg Loss: {avg_loss:.4f} Avg EPE: {avg_epe:.4f}")

        # ---------- Validation ----------
        model.eval()
        val_loss, val_epe = 0, 0
        with torch.no_grad():
            for re_img, tar_img, true_disp in val_loader:
                re_img, tar_img, true_disp = re_img.to(device), tar_img.to(device), true_disp.to(device)
                input_img = torch.cat((re_img, tar_img), dim=1)
                pred_disp = model(input_img)

                loss_shape = shape_loss(pred_disp, true_disp)
                loss_error = absolute_loss(pred_disp, true_disp)
                loss_total = loss_shape + loss_error
                epe = torch.norm(pred_disp - true_disp, p=2, dim=1).mean().item()

                val_loss += loss_total.item()
                val_epe += epe

        avg_val_loss = val_loss / len(val_loader)
        avg_val_epe = val_epe / len(val_loader)

        train_writer.add_scalar("Val/Loss_epoch", avg_val_loss, epoch)
        train_writer.add_scalar("Val/AEE_epoch", avg_val_epe, epoch)
        print(f"[Val]   Epoch {epoch+1}/{epochs} Avg Loss: {avg_val_loss:.4f} Avg EPE: {avg_val_epe:.4f}")

        # ---------- Save checkpoint (每个 epoch) ----------
        checkpoint_path = os.path.join(model_path, f"DICNet1_epoch{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # ---------- Save best ----------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(model_path, "DICNet1_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model (Val Loss={best_val_loss:.4f})")

        scheduler.step()


if __name__ == "__main__":
    main()
