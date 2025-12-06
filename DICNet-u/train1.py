import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from Net.DICNet_corr import DICNet  # 导入自定义DICNet网络
from dataset1.dataset1 import DICDataset  # 导入自定义数据集类
from torch.optim.lr_scheduler import StepLR
from util import AverageMeter  # 假设用于统计指标的工具类


# -------------------------- 1. 核心损失函数与评估指标 --------------------------
def pearson_corr_loss(y_true, y_pred):
    """皮尔逊相关系数损失，添加1e-8防止除零"""
    x = y_true.flatten()
    y = y_pred.flatten()
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    rho = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(vx ** 2) + 1e-8) * torch.sqrt(torch.sum(vy ** 2) + 1e-8)
    )
    return 1 - rho


def calculate_aee(pred_disp, gt_disp):
    """计算平均端点误差(Average Endpoint Error)"""
    return torch.mean(torch.norm(pred_disp - gt_disp, p=2, dim=1))  # 按通道计算L2范数，再取平均


# -------------------------- 2. 模型保存/恢复工具 --------------------------
def save_model(model, epoch, optimizer, scheduler, save_dir, model_name, is_best=False):
    os.makedirs(save_dir, exist_ok=True)
    if is_best:
        save_path = os.path.join(save_dir, "best_" + model_name)
    else:
        save_path = os.path.join(save_dir, f"epoch_{epoch}_" + model_name)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }, save_path)
    print(f"✅ Model saved to {save_path}")


def load_model(model, optimizer, scheduler, load_path):
    best_val_loss = float("inf")
    start_epoch = 0

    if os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location=next(model.parameters()).device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"🔄 Loaded model from {load_path}, start at epoch {start_epoch}")
    else:
        print(f"⚠️ Model path {load_path} not found, training from scratch")

    return start_epoch, best_val_loss


# -------------------------- 3. 图像 warp 函数 --------------------------
def get_predicted_reference_image(def_img, displacements):
    batch_size, _, height, width = displacements.shape
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=def_img.device),
        torch.linspace(-1, 1, width, device=def_img.device),
        indexing="ij"
    )
    grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    flow = displacements.permute(0, 2, 3, 1)
    flow[..., 0] /= (width / 2)
    flow[..., 1] /= (height / 2)

    warped_grid = grid - flow
    predicted_ref_img = F.grid_sample(
        def_img,
        warped_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True
    )
    return predicted_ref_img


# -------------------------- 4. 验证流程 --------------------------
def validate_supervised(model, val_loader, criterion, device, val_writer, epoch):
    model.eval()
    val_loss_meter = AverageMeter()
    val_aee_meter = AverageMeter()

    with torch.no_grad():
        for ref_img, def_img, gt_disp_x, gt_disp_y in val_loader:
            ref_img = ref_img.float().to(device)
            def_img = def_img.float().to(device)
            gt_disp = torch.cat((gt_disp_x, gt_disp_y), dim=1).float().to(device)

            pred_disp = model(torch.cat((ref_img, def_img), dim=1))
            loss = criterion(pred_disp, gt_disp)
            aee = calculate_aee(pred_disp, gt_disp)

            val_loss_meter.update(loss.item(), ref_img.size(0))
            val_aee_meter.update(aee.item(), ref_img.size(0))

    # 记录验证集指标
    val_writer.add_scalar("Loss", val_loss_meter.avg, epoch)
    val_writer.add_scalar("AEE", val_aee_meter.avg, epoch)

    model.train()
    return val_loss_meter.avg, val_aee_meter.avg


def validate_unsupervised(model, val_loader, device, val_writer, epoch):
    model.eval()
    val_loss_meter = AverageMeter()
    val_aee_meter = AverageMeter()
    mse_criterion = nn.MSELoss()

    with torch.no_grad():
        for ref_img, def_img, gt_disp_x, gt_disp_y in val_loader:  # 无监督验证仍需真实位移计算AEE
            ref_img = ref_img.float().to(device)
            def_img = def_img.float().to(device)
            gt_disp = torch.cat((gt_disp_x, gt_disp_y), dim=1).float().to(device)

            pred_disp = model(torch.cat((ref_img, def_img), dim=1))
            predicted_ref_img = get_predicted_reference_image(def_img, pred_disp)

            mse_loss = mse_criterion(predicted_ref_img, ref_img)
            corr_loss = pearson_corr_loss(predicted_ref_img, ref_img)
            total_loss = mse_loss + corr_loss
            aee = calculate_aee(pred_disp, gt_disp)

            val_loss_meter.update(total_loss.item(), ref_img.size(0))
            val_aee_meter.update(aee.item(), ref_img.size(0))

    # 记录验证集指标
    val_writer.add_scalar("Loss", val_loss_meter.avg, epoch)
    val_writer.add_scalar("AEE", val_aee_meter.avg, epoch)

    model.train()
    return val_loss_meter.avg, val_aee_meter.avg


# -------------------------- 5. 训练流程 --------------------------
def pretrain_supervised(model, train_loader, val_loader, optimizer, scheduler, criterion,
                        device, save_dir, start_epoch=0, total_epochs=30):
    best_val_loss = float("inf")
    # 初始化训练和验证日志写入器
    train_writer = SummaryWriter(os.path.join(save_dir, "pretrain_log1"))
    val_writer = SummaryWriter(os.path.join(save_dir, "pretrain_val_log1"))

    for epoch in range(start_epoch, total_epochs):
        model.train()
        train_loss_meter = AverageMeter()
        train_aee_meter = AverageMeter()

        # 训练迭代
        for batch_idx, (ref_img, def_img, gt_disp_x, gt_disp_y) in enumerate(train_loader):
            ref_img = ref_img.float().to(device)
            def_img = def_img.float().to(device)
            gt_disp = torch.cat((gt_disp_x, gt_disp_y), dim=1).float().to(device)

            optimizer.zero_grad()
            pred_disp = model(torch.cat((ref_img, def_img), dim=1))
            loss = criterion(pred_disp, gt_disp)
            aee = calculate_aee(pred_disp, gt_disp)

            loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item(), ref_img.size(0))
            train_aee_meter.update(aee.item(), ref_img.size(0))

            print(f"📌 Pretrain Epoch[{epoch + 1}/{total_epochs}] Batch[{batch_idx + 1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.6f}, AEE: {aee.item():.6f}")

        # 学习率调度
        scheduler.step()

        # 记录训练集指标
        train_writer.add_scalar("Loss", train_loss_meter.avg, epoch)
        train_writer.add_scalar("AEE", train_aee_meter.avg, epoch)
        train_writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch)

        # 验证迭代
        val_loss, val_aee = validate_supervised(model, val_loader, criterion, device, val_writer, epoch)

        # 打印 epoch 总结
        print(f"📊 Pretrain Epoch[{epoch + 1}/{total_epochs}] "
              f"Train Loss: {train_loss_meter.avg:.6f}, Train AEE: {train_aee_meter.avg:.6f} | "
              f"Val Loss: {val_loss:.6f}, Val AEE: {val_aee:.6f}")

        # 预训练阶段模型保存逻辑（修改后）
        # 1. 保存当前epoch的模型（每个epoch都保存，带epoch编号）
        save_model(
            model=model,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            save_dir=os.path.join(save_dir, "model1"),
            model_name=f"pretrain_dicnet1_epoch_{epoch}.pth",  # 每个epoch单独命名
            is_best=False  # 非最佳模型标记
        )

        # 2. 若当前验证损失更优，保存为最佳模型（固定名称）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(
                model=model,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                save_dir=os.path.join(save_dir, "model1"),
                model_name="pretrain_dicnet1_best.pth",  # 固定最佳模型名称
                is_best=True  # 最佳模型标记
            )
            print(f"🌟 New best val loss: {best_val_loss:.6f} (saved as best model)")

    # 关闭写入器
    train_writer.close()
    val_writer.close()
    print("✅ Pretrain completed!")
    return best_val_loss


def train_unsupervised(model, train_loader, val_loader, optimizer, scheduler, device,
                       save_dir, start_epoch=0, total_epochs=270):
    best_val_loss = float("inf")
    # 初始化训练和验证日志写入器
    train_writer = SummaryWriter(os.path.join(save_dir, "train_log1"))
    val_writer = SummaryWriter(os.path.join(save_dir, "val_log1"))
    mse_criterion = nn.MSELoss()

    for epoch in range(start_epoch, total_epochs):
        model.train()
        train_loss_meter = AverageMeter()
        train_mse_meter = AverageMeter()
        train_corr_meter = AverageMeter()
        train_aee_meter = AverageMeter()  # 训练集也计算AEE（需要标注）

        # 训练迭代
        for batch_idx, (ref_img, def_img, gt_disp_x, gt_disp_y) in enumerate(train_loader):
            ref_img = ref_img.float().to(device)
            def_img = def_img.float().to(device)
            gt_disp = torch.cat((gt_disp_x, gt_disp_y), dim=1).float().to(device)

            optimizer.zero_grad()
            pred_disp = model(torch.cat((ref_img, def_img), dim=1))
            predicted_ref_img = get_predicted_reference_image(def_img, pred_disp)

            # 计算损失
            mse_loss = mse_criterion(predicted_ref_img, ref_img)
            corr_loss = pearson_corr_loss(predicted_ref_img, ref_img)
            total_loss = mse_loss + corr_loss
            # 计算AEE（评估指标）
            aee = calculate_aee(pred_disp, gt_disp)

            total_loss.backward()
            optimizer.step()

            # 更新统计
            train_loss_meter.update(total_loss.item(), ref_img.size(0))
            train_mse_meter.update(mse_loss.item(), ref_img.size(0))
            train_corr_meter.update(corr_loss.item(), ref_img.size(0))
            train_aee_meter.update(aee.item(), ref_img.size(0))

            print(f"📌 Unsupervised Epoch[{epoch + 1}/{total_epochs}] Batch[{batch_idx + 1}/{len(train_loader)}] "
                  f"Total Loss: {total_loss.item():.6f}, AEE: {aee.item():.6f}")

        # 学习率调度
        scheduler.step()

        # 记录训练集指标
        train_writer.add_scalar("TotalLoss", train_loss_meter.avg, epoch)
        train_writer.add_scalar("MSELoss", train_mse_meter.avg, epoch)
        train_writer.add_scalar("CorrLoss", train_corr_meter.avg, epoch)
        train_writer.add_scalar("AEE", train_aee_meter.avg, epoch)
        train_writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch)

        # 验证迭代
        val_loss, val_aee = validate_unsupervised(model, val_loader, device, val_writer, epoch)

        # 打印 epoch 总结
        print(f"📊 Unsupervised Epoch[{epoch + 1}/{total_epochs}] "
              f"Train Loss: {train_loss_meter.avg:.6f}, Train AEE: {train_aee_meter.avg:.6f} | "
              f"Val Loss: {val_loss:.6f}, Val AEE: {val_aee:.6f}")

        # 模型保存（修改后）
        # 1. 保存当前epoch的模型（每个epoch都保存）
        save_model(
            model=model,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            save_dir=os.path.join(save_dir, "model1"),
            model_name=f"dicnet-u1_epoch_{epoch}.pth",  # 每个epoch用编号区分
            is_best=False  # 非最佳模型
        )

        # 2. 若当前验证损失更优，保存为最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(
                model=model,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                save_dir=os.path.join(save_dir, "model1"),
                model_name="dicnet-u1_best.pth",  # 固定名称，覆盖更新
                is_best=True  # 标记为最佳模型
            )
            print(f"🌟 New best val loss: {best_val_loss:.6f} (saved as best model)")
    # 关闭写入器
    train_writer.close()
    val_writer.close()
    print("✅ Unsupervised training completed!")
    return best_val_loss


# -------------------------- 6. 主函数 --------------------------
def main():
    # 基础配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Using device: {device}")

    # 路径配置
    project_root = "/home/dell/DATA/wh/DICNet_Unsupervised/unDICNet_coor"
    save_root = '/home/dell/DATA/wh/DICNet_Unsupervised/unDICNet_coor/result/'
    data_root = '/home/dell/DATA/wh/DATASET/'
    pretrainned=True
    if not pretrainned:
    # 数据集加载
        pretrain_dataset = DICDataset(
            root_dir=os.path.join(data_root, "Train1"),
           csv_file=os.path.join(data_root, "Train_annotations_1.csv"),
            is_pretrain=True
        )
        train_dataset = DICDataset(
            root_dir=os.path.join(data_root, "Train1"),
            csv_file=os.path.join(data_root, "Train_annotations_1.csv"),  # 无监督训练仍需标注计算AEE
            is_pretrain=False
        )
        val_dataset = DICDataset(
            root_dir=os.path.join(data_root, "Train1"),
            csv_file=os.path.join(data_root, "Val_annotations_1.csv"),
            is_pretrain=False
        )

    # 数据加载器
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=4, shuffle=True, num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    print(f"📥 Dataset sizes: "
          f"Pretrain: {len(pretrain_dataset)}, "
          f"Train: {len(train_dataset)}, "
          f"Val: {len(val_dataset)}")

    # 模型与优化器初始化
    model = DICNet().to(device)
    pre_optimizer = optim.Adam(model.parameters(), lr=0.00008, betas=(0.9, 0.999))
    pre_scheduler = StepLR(pre_optimizer, step_size=20, gamma=0.8)
    un_optimizer = optim.Adam(model.parameters(), lr=0.0008, betas=(0.9, 0.999))
    un_scheduler = StepLR(un_optimizer, step_size=20, gamma=0.8)

    # 监督预训练
    pre_model_load_path = os.path.join(save_root, "model1", "best_pretrain_dicnet.pth")
    pre_start_epoch, _ = load_model(model, pre_optimizer, pre_scheduler, pre_model_load_path)
    pretrain_supervised(
        model=model,
        train_loader=pretrain_loader,
        val_loader=val_loader,
        optimizer=pre_optimizer,
        scheduler=pre_scheduler,
        criterion=nn.MSELoss(),
        device=device,
        save_dir=save_root,
        start_epoch=pre_start_epoch,
        total_epochs=30
    )

    # 无监督训练
    un_model_load_path = os.path.join(save_root, "model1", "best_unsupervised_dicnet.pth")
    un_start_epoch, _ = load_model(model, un_optimizer, un_scheduler, un_model_load_path)
    train_unsupervised(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=un_optimizer,
        scheduler=un_scheduler,
        device=device,
        save_dir=save_root,
        start_epoch=un_start_epoch,
        total_epochs=270
    )


if __name__ == "__main__":
    main()
