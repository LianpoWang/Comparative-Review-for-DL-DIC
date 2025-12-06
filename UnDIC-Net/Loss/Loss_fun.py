# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


'''
s_im1 s_im2裁剪后的图片，
im1_s, im2_s 裁剪前的图片
start 左上坐标
s_flow_f 前向光流
s_flow_b 反向光流
occ_fw 前向遮挡
occ_bw 后向遮挡
flows 金字塔光流
arg：参数
'''



import torch.nn.functional as F

from Loss.Census_loss import census_loss_torch
from Loss.Photo_loss import photo_loss_multi_type, warp
from Loss.Smooth_loss import edge_aware_smoothness_order1, edge_aware_smoothness_order2


def upsample_flow(inputs, target_size=None, target_flow=None, mode="bilinear"):
    if target_size is not None:
        h, w = target_size
    elif target_flow is not None:
        _, _, h, w = target_flow.size()
    else:
        raise ValueError('wrong input')
    _, _, h_, w_ = inputs.size()
    res = F.interpolate(inputs, [h, w], mode=mode, align_corners=True)
    res[:, 0, :, :] *= (w / w_)
    res[:, 1, :, :] *= (h / h_)
    return res

def lossfun(ori_img1,ori_img2,crop_im1,crop_im2,start,flow_f,flow_b, arg):
    smooth_loss = 0
    output_dict = {}
    #======== 1 order smooth loss
    if arg["smooth_order_1_weight"] > 0:
        smooth_loss += arg["smooth_order_1_weight"] * edge_aware_smoothness_order1(img=crop_im1,pred=flow_f)
        smooth_loss += arg["smooth_order_1_weight"] * edge_aware_smoothness_order1(img=crop_im2,pred=flow_b)
    if arg["smooth_order_2_weight"] > 0:
        smooth_loss += arg["smooth_order_2_weight"] * edge_aware_smoothness_order2(img=crop_im1,pred=flow_f)
        smooth_loss += arg["smooth_order_2_weight"] * edge_aware_smoothness_order2(img=crop_im2,pred=flow_b)
    output_dict['smooth_loss'] = smooth_loss

    #======photo loss
    im1_warp = warp(ori_img2, flow_f, start)
    im2_warp = warp(ori_img1, flow_b, start)

    photo_loss= photo_loss_multi_type(crop_im1, im1_warp,  photo_loss_type=arg["photo_loss_type"],
                                                   photo_loss_delta=arg["photo_loss_delta"],
                                                 )
    photo_loss += photo_loss_multi_type(crop_im2, im2_warp,  photo_loss_type=arg["photo_loss_type"],
                                                    photo_loss_delta=arg["photo_loss_delta"],
                                                   )
    output_dict['photo_loss'] = photo_loss


    # === census loss
    if arg["photo_loss_census_weight"] > 0:
        census_loss = census_loss_torch(img1=crop_im1, img1_warp=im1_warp,
                                                     q=arg["photo_loss_delta"],
                                                     charbonnier_or_abs_robust=False,
                                                      averge=True) + \
                      census_loss_torch(img1=crop_im2, img1_warp=im2_warp,
                                                     q=arg["photo_loss_delta"],
                                                     charbonnier_or_abs_robust=False,
                                                      averge=True)
        census_loss *= arg["photo_loss_census_weight"]
    else:
        census_loss = None

    output_dict['census_loss'] = census_loss

    return output_dict


def lossfun1(ori_img2, crop_im1, start, flow_f, arg):
    smooth_loss = 0
    output_dict = {}

    if arg["smooth_order_1_weight"] > 0:
        smooth_loss += arg["smooth_order_1_weight"] * edge_aware_smoothness_order1(img=crop_im1, pred=flow_f)
    if arg["smooth_order_2_weight"] > 0:
        smooth_loss += arg["smooth_order_2_weight"] * edge_aware_smoothness_order2(img=crop_im1, pred=flow_f)

    output_dict['smooth_loss'] = smooth_loss
    im1_warp = warp(ori_img2, flow_f, start)

    photo_loss = photo_loss_multi_type(crop_im1, im1_warp, photo_loss_type=arg["photo_loss_type"],
                                       photo_loss_delta=arg["photo_loss_delta"],
                                       )
    output_dict['photo_loss'] = photo_loss

    if arg["photo_loss_census_weight"] > 0:
        census_loss = census_loss_torch(img1=crop_im1, img1_warp=im1_warp,
                                        q=arg["photo_loss_delta"],
                                        charbonnier_or_abs_robust=False, averge=True)

        census_loss *= arg["photo_loss_census_weight"]
    else:
        census_loss = None

    output_dict['census_loss'] = census_loss

    return output_dict
