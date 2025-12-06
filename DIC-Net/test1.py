import os

import psutil
import time
import torch.nn as nn
from torch.autograd import Variable
from thop import profile
import torch
from torch.utils.tensorboard import SummaryWriter

from datasets.dataset import MyDataset
from DICNet.DICNet import DICNet_d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DICNet_d(2,2).to(device)
ck = torch.load("/home/dell/DATA/wh/DIC-Net/DIC-Net-main/result/model300/DICNet1_epoch300.pth")# file path of model parameters
model.load_state_dict(ck['model'])


test_dataset = MyDataset('/home/dell/DATA/wh/DATASET/Test1', 40800)
test_loader = torch.utils.data.DataLoader(test_dataset)
loss_func = nn.MSELoss()
print('Number of test samples: {} '.format(test_dataset.__len__()))
start_time = time.time()
total_aee = 0.0
total_mae = 0.0
total_mse = 0.0
test_logdir=os.path.join('/home/dell/DATA/wh/DIC-Net/DIC-Net-main/result/','test_log1')
if not os.path.exists(test_logdir):
    os.makedirs(test_logdir)
test_writer=SummaryWriter(test_logdir)
n_iter=0
# 遍历文件夹中的图像
with torch.no_grad():
    for step, (img, gt) in enumerate(test_loader):
        img = Variable(img).cuda()
        gt = Variable(gt).cuda()

        output = model(img)
        loss = loss_func(output, gt)
        # 计算 AEE (欧几里得距离均值)
        aee = torch.norm(output - gt, dim=1).mean().item()
        total_aee += aee

        # 计算 MSE（均方误差）
        mse = torch.mean((output - gt) ** 2).item()
        total_mse += mse

        # 计算 MAE（平均绝对误差）
        mae = torch.mean(torch.abs(output - gt)).item()
        total_mae += mae
        print(f"{n_iter}/{test_dataset.__len__()}\t AEE: {aee} \t Loss: {loss.item()}")
        test_writer.add_scalar('Test Loss', loss.item(), n_iter)
        test_writer.add_scalar('Test AEE', aee, n_iter)
        n_iter += 1
# 计算总推理时间
end_time = time.time()
total_inference_time = (end_time - start_time)*1000
average_inference_time = total_inference_time / len(test_dataset)  # 计算平均推理时间
# 计算 AEE、MAE、MSE 的均值
avg_aee = total_aee / len(test_dataset)
avg_mae = total_mae / len(test_dataset)
avg_mse = total_mse / len(test_dataset)
print(f"Test dataset Average AEE : {avg_aee:.3f}\t MAE: {avg_mae:.3f}\t MSE: {avg_mse:.3f}")
print(f'Total Inferenve Time:{total_inference_time:.2f}ms \nAverage Inference Time: {average_inference_time:.2f} ms')
# 计算最终的平均 AEE（全局平均 EPE）
dummy_input = torch.randn(1, 2, 256, 256).cuda()
flops, params = profile(model, inputs=(dummy_input,), verbose=False)
print(f"Model Complexity: FLOPs={flops / 1e9:.2f} G\t Params={params / 1e6:.2f} M")
print('Number of training images:', len(test_dataset))
print(f'GPU Memory Usage: {torch.cuda.memory_allocated() / 1e6:.2f} MB')
print(f'CPU Memory Usage: {psutil.virtual_memory().used / 1e9:.2f} GB')
