from datetime import datetime

import psutil
import time

from torch.utils.tensorboard import SummaryWriter

from DeepDIC import DeepDIC
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
from torchvision.models.resnet import BasicBlock, ResNet
from datasets.dataset import MyDataset
import os
from thop import profile
def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, bias=False, transposed=False):
    if transposed:
        layer = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1,
                                   output_padding=1,
                                   dilation=dilation, bias=bias)
    else:
        padding = (kernel_size + 2 * (dilation - 1)) // 2
        layer = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                          dilation=dilation, bias=bias)
    if bias:
        init.constant(layer.bias, 0)
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


# 定义 SegResNet 模型（确保与你的训练代码中的模型结构一致）
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
fcn = fcn.cuda().eval()
ck=torch.load("/data/wh/DeepDIC/result/deepdic_epoch300_step674")
fcn.load_state_dict(ck['model'], strict=False)
loss_func = nn.MSELoss()
test_dataset = MyDataset("/data/wh/DATASET/Test_annotations1.csv",'/data/wh/DATASET/Test1/', 40800)

test_loader = torch.utils.data.DataLoader(test_dataset)
print('Number of test samples: {} '.format(test_dataset.__len__()))
test_logdir=os.path.join('/data/wh/DeepDIC/result/model1/','test_log')
if not os.path.exists(test_logdir):
    os.makedirs(test_logdir)
test_writer=SummaryWriter(test_logdir)
# 计算测试误差
start_time = time.time()
total_aee = 0.0
total_mae = 0.0
total_mse = 0.0
start_time = time.time()
with torch.no_grad():
    for step, (img, gt) in enumerate(test_loader):
        img = Variable(img).cuda()
        gt = Variable(gt).cuda()

        output = fcn(img)
        loss = loss_func(output, gt)  # 计算损失
        # 计算 AEE (欧几里得距离均值)
        aee = torch.norm(output - gt, dim=1).mean().item()
        total_aee += aee

        # 计算 MSE（均方误差）
        mse = torch.mean((output - gt) ** 2).item()
        total_mse += mse

        # 计算 MAE（平均绝对误差）
        mae = torch.mean(torch.abs(output - gt)).item()
        total_mae += mae

        test_writer.add_scalar('Test Loss', loss.item(), step)
        test_writer.add_scalar('Test AEE', aee, step)
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
flops, params = profile(fcn, inputs=(dummy_input,), verbose=False)
print(f"Model Complexity: FLOPs={flops / 1e9:.2f} G\t Params={params / 1e6:.2f} M")
print('Number of training images:', len(test_dataset))
print(f'GPU Memory Usage: {torch.cuda.memory_allocated() / 1e6:.2f} MB')
print(f'CPU Memory Usage: {psutil.virtual_memory().used / 1e9:.2f} GB')





