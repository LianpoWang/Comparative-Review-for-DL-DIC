import torch
import numpy as np
import torch.nn.functional as F



def photo_loss_function( diff, q, charbonnier_or_abs_robust, averge=True):
    if charbonnier_or_abs_robust:
        p = ((diff) ** 2 + 1e-8).pow(q)
        if averge:
            p = p.mean()
        else:
            p = p.sum()
        return p
    else:
        diff = (torch.abs(diff) + 0.01).pow(q)
        if averge:
            loss_mean = diff.mean()
        else:
            loss_mean = diff.sum()
    return loss_mean



def census_loss_torch( img1, img1_warp, q, charbonnier_or_abs_robust, averge=True,
                      max_distance=3):
    patch_size = 2 * max_distance + 1

    def _ternary_transform_torch(image):

        intensities_torch = image
        out_channels = patch_size * patch_size
        w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))  # h,w,1,out_c (7, 7, 1, 49)
        w_ = np.transpose(w, (3, 2, 0, 1))  # 1,out_c,h,w (1 49, 7, 7)
        weight = torch.from_numpy(w_).float()
        if image.is_cuda:
            weight = weight.to(image.device)
        patches_torch = torch.conv2d(input=intensities_torch, weight=weight, bias=None, stride=[1, 1],
                                     padding=[max_distance, max_distance])  # torch.Size([1, 49, 10, 10])
        transf_torch = patches_torch - intensities_torch  # torch.Size([1, 49, 10, 10])
        transf_norm_torch = transf_torch / torch.sqrt(0.81 + transf_torch ** 2)  # torch.Size([1, 49, 10, 10])
        return transf_norm_torch

    def _hamming_distance_torch(t1, t2):
        t2 = t2.to(t1.device)
        dist = (t1 - t2) ** 2
        dist = torch.sum(dist / (0.1 + dist), 1, keepdim=True)
        return dist

    def create_mask_torch(tensor, paddings):
        shape = tensor.shape  # N,c, H,W 1 1 8 8
        inner_width = shape[2] - (paddings[0][0] + paddings[0][1])
        inner_height = shape[3] - (paddings[1][0] + paddings[1][1])
        inner_torch = torch.ones([shape[0], shape[1], inner_width, inner_height]).float()
        if tensor.is_cuda:
            inner_torch = inner_torch.cuda()
        mask2d = F.pad(inner_torch, [paddings[0][0], paddings[0][1], paddings[1][0], paddings[1][1]])
        return mask2d

    img1 = _ternary_transform_torch(img1)  # torch.Size([1, 49, 10, 10])
    img1_warp = _ternary_transform_torch(img1_warp)  # torch.Size([1, 49, 10, 10])
    dist = _hamming_distance_torch(img1, img1_warp)  # torch.Size([1, 1, 10, 10])

    census_loss = photo_loss_function(diff=dist,  q=q,
                                          charbonnier_or_abs_robust=charbonnier_or_abs_robust,
                                          averge=averge)
    return census_loss


if __name__ == '__main__':
    im1_ori = torch.randn(1, 1, 10, 10)  # 扭曲后的的图
    im1_warp = torch.randn(1, 1, 10, 10)  # 扭曲后的的图
    occ_fw = torch.randn(1, 1, 10, 10)  # 掩码
    photo_loss_delta = 0.4
    p = census_loss_torch(img1=im1_ori, img1_warp=im1_warp,occ_fw = torch.randn(1, 1, 10, 10), q=photo_loss_delta,
                                      charbonnier_or_abs_robust=False, if_use_occ=False,
                                      averge=True)
    print(p)
    # p1 = census_loss(im1_ori,im1_warp)
    # print(p1)
    # p2 = census_loss_with_mask(im1_ori,im1_warp,occ_fw)
    # print(p2)
