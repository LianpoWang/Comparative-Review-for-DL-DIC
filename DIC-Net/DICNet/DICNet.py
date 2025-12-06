import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


# 构建了一个包含两个卷积层和两个 ReLU的模块
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


# 实现了一个常见的残差网络（ResNet）中的瓶颈（Bottleneck）模块
class BottleNeck(nn.Module):
    expansion = 2  # 输出通道数是输入通道数的 2 倍。

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            # 残差功能组成：三个“卷积+归一化+RELU激活”
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),  # 批量归一化层，帮助稳定训练过程和加速收敛。
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()  # 实现残差连接的快捷方式

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class DICNet(nn.Module):

    def __init__(self, in_channel, out_channel, block, num_block):
        super().__init__()

        self.in_channels = 64
        # 相对位置提取器RPE
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 下采样
        self.conv1_x = self._make_layer(block, 64, num_block[5], 2)
        self.conv2_x = self._make_layer(block, 128, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 256, num_block[3], 2)
        self.conv6_x = self._make_layer(block, 512, num_block[4], 2)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 上采样
        self.dconv_up4 = double_conv(1024 + 512, 512)
        self.dconv_up3 = double_conv(512 + 512, 256)
        self.dconv_up2 = double_conv(256 + 256, 128)
        self.dconv_up1 = double_conv(128 + 256, 64)

        self.dconv_last = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
            nn.Conv2d(64, out_channel, 1)
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        conv1 = self.conv1(x)
        temp = self.conv1_x(conv1)
        conv2 = self.conv2_x(temp)
        conv3 = self.conv3_x(conv2)
        conv4 = self.conv4_x(conv3)
        conv5 = self.conv5_x(conv4)
        bottle = self.conv6_x(conv5)

        x = self.upsample(bottle)  # 1上采样

        x = torch.cat([x, conv5], dim=1)
        # 2拼接特征图，将conv5产生的特征图和特征图x拼接
        # dim=0/1/2/3,分别代表在批次/通道/高度/宽度上拼接
        x = self.dconv_up4(x)  # 3根据上一步结特征图继续double卷积
        x = self.upsample(x)  # 上采样，特征图大小增大

        x = torch.cat([x, conv4], dim=1)  # 拼接

        x = self.dconv_up3(x)
        x = self.upsample(x)

        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)

        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up1(x)
        x = self.upsample(x)

        x = torch.cat([x, conv1], dim=1)
        out = self.dconv_last(x)
        return out


def DICNet_d(in_channel, out_channel):
    model = DICNet(in_channel, out_channel, BottleNeck, [5, 6, 8, 5, 3, 1])

    return model
