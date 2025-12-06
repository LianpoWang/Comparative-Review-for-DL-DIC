import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .util import conv, predict_flow, deconv, crop_like
import torch.nn.parallel


__all__ = ['StrainNet_f', 'U_StrainNet_f', 'U_DICNet']


class StrainNetF(nn.Module):
    """
        args :
        batchNorm : if true, batchNorm was applied

    """

    def __init__(self, batchNorm=True):
        super(StrainNetF, self).__init__()

        self.batchNorm = batchNorm

        # convolutional layer
        self.conv1 = conv(self.batchNorm, 6, 64, kernel_size=7, stride=1)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=1)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        # deconvolution layer (transposed convolution with upsampling)
        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        # predict the flow (convolution layer)
        self.predict_flow6 = predict_flow(1024, 2)
        self.predict_flow5 = predict_flow(1026, 2)
        self.predict_flow4 = predict_flow(770, 2)
        self.predict_flow3 = predict_flow(386, 2)
        self.predict_flow2 = predict_flow(194, 2)

        # upsampling (transposed convolution)
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        # initialize the model parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        """
            args :
                x : the input |reference image target image|reference image target image|reference image target image|
            Returns:
                training : predicted flow of different scale: flow6, flow5, flow4, flow3, flow2
                evaluate : the final predicted flow: flow2
        """
        # multi convolutional layer for feature extraction
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        # predicted the flow6
        flow6 = self.predict_flow6(out_conv6)
        flow6_up = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        # predicted the flow5
        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        # predicted the flow4
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        # predicted the flow3
        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        # predicted the flow2
        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            # return the flow of different scale
            return flow2, flow3, flow4, flow5, flow6
        else:
            # return the final predicted flow
            return flow2

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def StrainNet_f(data=None):

    model = StrainNetF(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


class U_StrainNet_f_model(nn.Module):
    """
        args :
        batchNorm : if true, batchNorm was applied

    """

    def __init__(self, batchNorm=True):
        super(U_StrainNet_f_model, self).__init__()

        self.batchNorm = batchNorm

        # convolutional layer
        self.conv1 = conv(self.batchNorm, 2, 64, kernel_size=7, stride=1) # the input channel change
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=1)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        # deconvolution layer (transposed convolution with upsampling)
        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1025, 256)
        self.deconv3 = deconv(769, 128)
        self.deconv2 = deconv(385, 64)

        # predict the flow (convolution layer)
        self.predict_flow6 = predict_flow(1024, 1)
        self.predict_flow5 = predict_flow(1025, 1)
        self.predict_flow4 = predict_flow(769, 1)
        self.predict_flow3 = predict_flow(385, 1)
        self.predict_flow2 = predict_flow(193, 1)

        # upsampling (transposed convolution)
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)

        # initialize the model parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        """
            args :
                x : the input |reference image target image|
            Returns:
                training : predicted flow of different scale: flow6, flow5, flow4, flow3, flow2
                evaluate : the final predicted flow: flow2
        """
        # multi convolutional layer for feature extraction
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        # predict the flow
        flow6 = self.predict_flow6(out_conv6)  # 1层
        flow6_up = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)  # 1层
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)  # 512层

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)  # 1025层
        flow5 = self.predict_flow5(concat5)  # 1层
        flow5_up = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)  # 1层
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)  # 256层

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)  # 512+256+1=769层
        flow4 = self.predict_flow4(concat4)  # 1层
        flow4_up = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)  # 1层
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)  # 128层

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)  # 256+128+1=385层
        flow3 = self.predict_flow3(concat3)  # 1层
        flow3_up = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)  # 1层
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)  # 64层

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)  # 128+64+1=193层

        flow2 = self.predict_flow2(concat2)  # 1层

        if self.training:
            return flow2, flow3, flow4, flow5, flow6  # 用于组建损失函数，并在后面加上权重
        else:
            return flow2

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def U_StrainNet_f(data=None):
    model = U_StrainNet_f_model(batchNorm=True)
    # weight_init(model)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


class U_DICNet_model(nn.Module):
    """
        args :
            U_StrainNet_f : the network archtecture of U_StrainNet_f
            batchNorm : if true, batchNorm was applied
            drop: if true, dropout was used to avoid overfitting (default: true)
    """

    def __init__(self, U_StrainNet_f_model,  batchNorm=True, drop=False):
        super(U_DICNet_model, self).__init__()
        self.model = U_StrainNet_f_model(batchNorm=batchNorm)
        self.batchNorm = batchNorm

        # fine tuning of the layer in U-StrainNet-f
        conv1 = conv(self.batchNorm, 2, 128, kernel_size=5, stride=1)
        conv2 = conv(self.batchNorm, 128, 128, kernel_size=3, stride=1)
        conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=1)
        conv5 = conv(self.batchNorm, 512, 512, stride=1)
        conv4_1 = conv(self.batchNorm, 512, 512, drop=drop)
        conv5_1 = conv(self.batchNorm, 512, 512, drop=drop)
        conv6_1 = conv(self.batchNorm, 1024, 1024, drop=drop)
        deconv4 = deconv(1025, 256, stride=1, drop=drop)
        deconv2 = deconv(385, 64, stride=1, drop=drop)
        upsampled_flow5_to_4 = nn.ConvTranspose2d(1, 1, 4, 1, 1, bias=False)
        upsampled_flow3_to_2 = nn.ConvTranspose2d(1, 1, 4, 1, 1, bias=False)
        self.model.conv1 = conv1
        self.model.conv2 = conv2
        self.model.conv3 = conv3
        self.model.conv5 = conv5
        self.model.conv4_1 = conv4_1
        self.model.conv5_1 = conv5_1
        self.model.conv6_1 = conv6_1
        self.model.deconv4 = deconv4
        self.model.deconv2 = deconv2
        self.model.upsampled_flow5_to_4 = upsampled_flow5_to_4
        self.model.upsampled_flow3_to_2 = upsampled_flow3_to_2

    def forward(self, x):
        """
            args :
                x : the input |reference image target image|
            Returns:
                if training : predicted flow of different scale: flow6, flow5, flow4, flow3, flow2
                else:  the final predicted flow: flow2
        """
        x = self.model(x)
        return x

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def U_DICNet(data=None, batchNorm=True, drop=False):
    model = U_DICNet_model(U_StrainNet_f_model, batchNorm=batchNorm, drop=drop)
    if data is not None:
        # load the pre_trained model parameters
        model.load_state_dict(data['state_dict'])
        # model.load_state_dict({k.replace('module.', ''): v for k, v in data['state_dict'].items()})
    return model
