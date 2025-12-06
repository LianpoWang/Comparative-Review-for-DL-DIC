import torch
import numpy as np
import torch.nn.functional as F
import cv2 as cv

import cv2 as cv
import cv2 as cv
def ZNSSD_loss(f,g):
    f_ = torch.mean(f, dim=(2, 3), keepdim=True)
    dis_f=f-f_
    g_ = torch.mean(g, dim=(2, 3), keepdim=True)

    dis_g= g - g_
    delta_g = torch.sqrt(torch.sum(torch.square(dis_g + 1e-8), dim=(2, 3), keepdim=True))
    delta_f = torch.sqrt(torch.sum(torch.square(dis_f + 1e-8), dim=(2, 3), keepdim=True))

    # delta_g = torch.sum(torch.sqrt(torch.square(dis_g)+1e-8), dim=(2, 3), keepdim=True)
    # delta_f = torch.sum(torch.sqrt(torch.square(dis_f)+1e-8), dim=(2, 3), keepdim=True)
    diss=torch.div(dis_g, delta_g) - torch.div(dis_f, delta_f)
    loss = torch.sum(torch.square(diss),dim=(2,3),keepdim=True)

    return loss


def Pach_ZNSSD_loss(img,imgwarped,window_size = (40, 40),stride = (4, 4)):
    def ZNSSD(f, g, flag="mean"):
        f_ = torch.mean(f, dim=2, keepdim=True)
        dis_f = f - f_
        g_ = torch.mean(g, dim=2, keepdim=True)
        dis_g = g - g_
        delta_g = torch.sqrt(torch.sum(torch.square(dis_g+ 1e-8), dim=2, keepdim=True) )
        delta_f = torch.sqrt(torch.sum(torch.square(dis_f+ 1e-8), dim=2, keepdim=True) )
        # delta_g = torch.sum(torch.sqrt(torch.square(dis_g+1e-8)), dim=2, keepdim=True)
        # delta_f = torch.sum(torch.sqrt(torch.square(dis_f+1e-8)), dim=2, keepdim=True)
        diss = torch.div(dis_g, delta_g) - torch.div(dis_f, delta_f)

        loss = torch.sum(torch.square(diss), dim=2, keepdim=True)
        # print(loss)
        if flag == "mean":
            loss=torch.mean(loss,dim=1,keepdim=True)
        elif flag == "sum":
            loss = torch.sum(loss,dim=1,keepdim=True)
        return loss
    # 定义 padding 大小
    padding = (window_size[0] // 2, window_size[1] // 2)
    # 对输入数据进行 padding
    padded_input = torch.nn.functional.pad(img, (padding[1], padding[1], padding[0], padding[0]))
    # 对 padding 后的输入数据进行 unfold 操作
    unfolded_input = torch.nn.functional.unfold(padded_input, kernel_size=window_size, stride=stride)
    unfolded_input=unfolded_input.permute(0,2,1)
    # 对输入数据进行 padding
    padded_imgwarped = torch.nn.functional.pad(imgwarped, (padding[1], padding[1], padding[0], padding[0]))
    # 对 padding 后的输入数据进行 unfold 操作
    unfolded_imgwarped = torch.nn.functional.unfold(padded_imgwarped , kernel_size=window_size, stride=stride)
    unfolded_imgwarped=unfolded_imgwarped.permute(0,2,1)

    loss = ZNSSD(unfolded_input, unfolded_imgwarped)
    return loss


def warp(img, flo, start):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B1, C1, H1, W1 = img.shape
        B, C, H, W = flo.shape
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().to(device=img.device)
        start = start.view(start.shape[0], 2, 1, 1)  # 变成 (8, 2, 1, 1)
        start = start.expand(-1, -1, grid.shape[2], grid.shape[3])  # 扩展到 (8, 2, 256, 256)
        grid = grid.to(flo.device)
        start=start.to(flo.device)
        img = img.to(flo.device)
        grid[:, :2, :, :] = grid[:, :2, :, :] + start
        vgrid = grid + flo  # B,2,H,W
        # 图二的每个像素坐标加上它的光流即为该像素点对应在图一的坐标
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W1 - 1, 1) - 1.0
        # 取出光流v这个维度，原来范围是0~W-1，再除以W-1，范围是0~1，再乘以2，范围是0~2，再-1，范围是-1~1
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H1 - 1, 1) - 1.0  # 取出光流u这个维度，同上
        vgrid = vgrid.permute(0, 2, 3, 1)  # from B,2,H,W -> B,H,W,2，为什么要这么变呢？是因为要配合grid_sample这个函数的使用
        output = F.grid_sample(img, vgrid, align_corners=True)
        return output


def photo_loss_multi_type( x, y,  photo_loss_type='abs_robust',  # abs_robust, charbonnier,L1, SSIM
                              photo_loss_delta=0.4, photo_loss_use_occ=False,
                              ):



        x = x.to(y.device)
        if photo_loss_type == 'abs_robust':
            photo_diff = x - y
            loss_diff = (torch.abs(photo_diff) + 0.01).pow(photo_loss_delta)
        elif photo_loss_type == 'charbonnier':
            photo_diff = x - y
            loss_diff = ((photo_diff) ** 2 + 1e-6).pow(photo_loss_delta)
        elif photo_loss_type == 'L1':
            photo_diff = x - y
            loss_diff = torch.abs(photo_diff + 1e-6)

        elif photo_loss_type == 'Pach_ZNSSD':
            loss_diff = Pach_ZNSSD_loss(x, y)

        elif photo_loss_type == 'ZNSSD':
            loss_diff = ZNSSD_loss(x, y)

        else:
            raise ValueError('wrong photo_loss type: %s' % photo_loss_type)


        photo_loss = torch.mean(loss_diff)
        return photo_loss
'''
计算加权结构化图像相似性度量
'''
def weighted_ssim(x, y, weight, c1=float('inf'), c2=9e-6, weight_epsilon=0.01):
    """Computes a weighted structured image similarity measure. 计算加权结构化图像相似性度量。
    Args:
      x: a batch of images, of shape [B, C, H, W].   一批图像，形状[B，C，H，W]。
      y:  a batch of images, of shape [B, C, H, W].  一批图像，形状[B，C，H，W]。
      weight: shape [B, 1, H, W], representing the weight of each
        pixel in both images when we come to calculate moments (means and
        correlations). values are in [0,1]  weight：形状[B，1，H，W]，表示当我们计算矩（均值和相关性）时，两幅图像中每个像素的权重。值在[0,1]之间
      c1: A floating point number, regularizes division by zero of the means. 一个浮点数，用零整除均值。
      c2: A floating point number, regularizes division by zero of the second 一个浮点数，将第二个数除以零正则化
        moments.
      weight_epsilon: A floating point number, used to regularize division by the 一个浮点数，用于正则化重量
        weight.

    Returns:
      A tuple of two pytorch Tensors. First, of shape [B, C, H-2, W-2], is scalar
      similarity loss per pixel per channel, and the second, of shape
      [B, 1, H-2. W-2], is the average pooled `weight`. It is needed so that we
      know how much to weigh each pixel in the first tensor. For example, if
      `'weight` was very small in some area of the images, the first tensor will
      still assign a loss to these pixels, but we shouldn't take the result too
      seriously.
      两个pytorch张量的元组。首先，形状[B，C，H-2，W-2]是标量每个通道每个像素的相似性损失，以及形状的第二个[B，1，H-2。W-2]是平均合并“权重”。我们需要它
      知道第一个张量中每个像素的权重。例如，如果`“权重”在图像的某些区域非常小，第一个张量将仍然将损失分配给这些像素，但我们不应该也计算结果认真地
    """

    def _avg_pool3x3(x):
        # tf kernel [b,h,w,c]
        return F.avg_pool2d(x, (3, 3), (1, 1))
        # return tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 1, 1, 1], 'VALID')

    if c1 == float('inf') and c2 == float('inf'):
        raise ValueError('Both c1 and c2 are infinite, SSIM loss is zero. This is '
                         '这可能是意外'
                         'likely unintended.')
    average_pooled_weight = _avg_pool3x3(weight)
    weight_plus_epsilon = weight + weight_epsilon
    inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)

    def weighted_avg_pool3x3(z):
        wighted_avg = _avg_pool3x3(z * weight_plus_epsilon)
        return wighted_avg * inverse_average_pooled_weight

    mu_x = weighted_avg_pool3x3(x)
    mu_y = weighted_avg_pool3x3(y)
    sigma_x = weighted_avg_pool3x3(x ** 2) - mu_x ** 2
    sigma_y = weighted_avg_pool3x3(y ** 2) - mu_y ** 2
    sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y
    if c1 == float('inf'):
        ssim_n = (2 * sigma_xy + c2)
        ssim_d = (sigma_x + sigma_y + c2)
    elif c2 == float('inf'):
        ssim_n = 2 * mu_x * mu_y + c1
        ssim_d = mu_x ** 2 + mu_y ** 2 + c1
    else:
        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    result = ssim_n / ssim_d
    return torch.clamp((1 - result) / 2, 0, 1), average_pooled_weight


if __name__ == '__main__':
    im1_ori = torch.randn(5, 1, 256, 256)  # 裁剪前的图
    im1_croped= torch.randn(5, 1, 196, 196)  # 裁剪后的图
    flow = torch.randn(5, 2, 196, 196)  # 扭曲后的的图
    occ_fw = torch.randn(5, 1, 196, 196)  # 掩码
    # flow = torch.randn(1,2, 10, 10)  # 掩码
    start = torch.tensor([[[[0]], [[0]]]])
    im2_warp = warp( im1_ori, flow, start)

    photo_loss = photo_loss_multi_type(im1_croped, im2_warp, occ_fw , photo_loss_type='abs_robust',photo_loss_delta=0.4, photo_loss_use_occ=False)
    print(photo_loss)
    photo_loss = photo_loss_multi_type(im1_croped, im2_warp, occ_fw, photo_loss_type='charbonnier', photo_loss_delta=0.4,photo_loss_use_occ=False)
    print(photo_loss)
    photo_loss = photo_loss_multi_type(im1_croped, im2_warp, occ_fw, photo_loss_type='L1', photo_loss_delta=0.4, photo_loss_use_occ=False)
    print(photo_loss)
    photo_loss = photo_loss_multi_type(im1_croped, im2_warp, occ_fw, photo_loss_type='SSIM', photo_loss_delta=0.4,photo_loss_use_occ=False)
    print(photo_loss)