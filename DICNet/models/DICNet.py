import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.init import kaiming_normal_, constant_

class ChannelAttention(nn.Module):
    def __init__(self, in_planes=2, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class DepthwiseConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super(DepthwiseConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=in_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=kernel_size//2,
                                    groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class DICNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super(DICNet, self).__init__()
        self.activation_function = nn.LeakyReLU()

        self.conv1_1 = DepthwiseConv(in_channels, 32, kernel_size=7)
        self.conv1_2 = DepthwiseConv(32, 32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = DepthwiseConv(32, 64, kernel_size=3)
        self.conv2_2 = DepthwiseConv(64, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = DepthwiseConv(64, 128, kernel_size=3)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = DepthwiseConv(128, 256, kernel_size=3)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = DepthwiseConv(256, 512, kernel_size=3)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.cbam = cbam_block(channel=512)

        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv10_1 = nn.Conv2d(32, out_channels, kernel_size=1, stride=1)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):

        conv1 = self.activation_function(self.conv1_1(x))
        conv1 = self.activation_function(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)

        conv2 = self.activation_function(self.conv2_1(pool1))
        conv2 = self.activation_function(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)

        conv3 = self.activation_function(self.conv3_1(pool2))
        conv3 = self.activation_function(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)

        conv4 = self.activation_function(self.conv4_1(pool3))
        conv4 = self.activation_function(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)

        conv5 = self.activation_function(self.conv5_1(pool4))
        conv5 = self.activation_function(self.conv5_2(conv5))   
        conv5 = self.cbam(conv5)

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.activation_function(self.conv6_1(up6))
        conv6 = self.activation_function(self.conv6_2(conv6))

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.activation_function(self.conv7_1(up7))
        conv7 = self.activation_function(self.conv7_2(conv7))

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.activation_function(self.conv8_1(up8))
        conv8 = self.activation_function(self.conv8_2(conv8))

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.activation_function(self.conv9_1(up9))
        conv9 = self.activation_function(self.conv9_2(conv9))

        conv10 = self.conv10_1(conv9)

        return conv10


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DICNet().to(device)
    

    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))


    net.eval()
    with torch.no_grad():
        input_x = torch.zeros((1, 2, 480, 480)).to(device)
        
        import time
        time0 = time.time()
        for _ in range(100):
            y = net(input_x)
        print( (time.time() - time0)/100)

        print(input_x.shape, y.shape)


