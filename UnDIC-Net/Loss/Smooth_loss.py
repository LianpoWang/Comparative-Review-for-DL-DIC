import torch
import numpy as np
import torch.nn.functional as F
import cv2 as cv

import cv2 as cv

def gradient_x(img, stride=1):
    gx = img[:, :, :-stride, :] - img[:, :, stride:, :]
    return gx

def gradient_y(img, stride=1):
    gy = img[:, :, :, :-stride] - img[:, :, :, stride:]
    return gy

def edge_aware_smoothness_order1( img, pred):

    pred_gradients_x = gradient_x(pred,1)
    pred_gradients_y = gradient_y(pred,1)

    image_gradients_x = gradient_x(img,1)
    image_gradients_y = gradient_y(img,1)

    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

    smoothness_x = torch.abs(pred_gradients_x) * weights_x
    smoothness_y = torch.abs(pred_gradients_y) * weights_y
    return torch.mean(smoothness_x) + torch.mean(smoothness_y)


def edge_aware_smoothness_order2(img, pred):
    pred_gradients_x = gradient_x(pred,1)
    pred_gradients_xx = gradient_x(pred_gradients_x,1)
    pred_gradients_y = gradient_y(pred,1)
    pred_gradients_yy = gradient_y(pred_gradients_y,1)
    image_gradients_x = gradient_x(img, stride=2)
    image_gradients_y = gradient_y(img, stride=2)
    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))
    smoothness_x = torch.abs(pred_gradients_xx) * weights_x
    smoothness_y = torch.abs(pred_gradients_yy) * weights_y
    return torch.mean(smoothness_x) + torch.mean(smoothness_y)



if __name__ == '__main__':

    img= torch.randn(1, 1, 196, 196)  # 裁剪后的图


    flow = torch.randn(1, 2, 196, 196)  # 扭曲后的的图
    gx = gradient_x(img)
    gy = gradient_y(img)


    grad_flow = F.pad(F.avg_pool2d(img, kernel_size=2, stride=1), [0, 1, 0, 1], mode='replicate') - img

    grad_y = F.pad(img[:, :, :, :-1], (1, 0, 0, 0), mode='replicate') - F.pad(img[:, :, :, 1:], (0, 1, 0, 0),
                                                                              mode='replicate')

    grad_x = F.pad(img[:, :, :-1, :], (0, 0, 1, 0), mode='replicate') - F.pad(img[:, :, 1:, :], (0, 0, 0, 1),
                                                                              mode='replicate')



    # p1 = edge_aware_smoothness_1d(  im1_croped, flow)
    # print(p1)
    # p2 = edge_aware_smoothness_2d(im1_croped, flow)
    # print(p2)