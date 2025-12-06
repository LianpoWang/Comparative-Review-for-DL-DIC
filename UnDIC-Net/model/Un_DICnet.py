import torch
import torch.nn as nn
from Loss.Loss_fun import lossfun  # 维持你原来的导入
from model.UnDICnet import UnDICnet_s

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class UnDICnet_S(nn.Module):
    def __init__(self, args):
        super(UnDICnet_S, self).__init__()
        self.DICnet = UnDICnet_s()
        self.args = args

    def forward(self, input_dict: dict, if_loss=True):
        # 只取 Ref / Def
        im1 = input_dict['Ref'].to(device)  # (B,1,H,W)
        im2 = input_dict['Def'].to(device)  # (B,1,H,W)

        # 前向预测
        flow_f_out, flows_f = self.DICnet(im1, im2)

        output_dict = {}
        if if_loss:
            # 和原签名保持一致：im1_s, im2_s, im1_ori, im2_ori
            # 既然不裁剪，就都传同一对图像
            start = input_dict.get(
                'start',
                torch.zeros(im1.size(0), 2, 1, 1, device=im1.device, dtype=im1.dtype)
            )
            output_dict = lossfun(im1, im2, im1, im2, start, flow_f_out, self.args)

        output_dict['flow_f_out'] = flow_f_out
        return output_dict
import torch
import torch.nn as nn
from Loss.Loss_fun import lossfun  # D 版的 lossfun 你原来传了两条流，保持不变
from model.UnDICnet import UnDICnet_d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class UnDICnet_D(nn.Module):
    def __init__(self, args):
        super(UnDICnet_D, self).__init__()
        self.DICnet = UnDICnet_d()
        self.args = args

    def forward(self, input_dict: dict, if_loss=True):
        im1 = input_dict['Ref'].to(device)
        im2 = input_dict['Def'].to(device)

        output = self.DICnet(im1, im2)  # forward estimation
        # 兼容 dict / tuple 两种返回
        if isinstance(output, dict):
            flow_f_out = output.get('flow_f_out', None)
            flow_b_out = output.get('flow_b_out', None)
        else:
            # 若模型返回 (flow_f, flow_b, ...) 这样的 tuple
            flow_f_out = output[0]
            flow_b_out = output[1] if len(output) > 1 else None

        output_dict = {}
        if if_loss:
            start = input_dict.get(
                'start',
                torch.zeros(im1.size(0), 2, 1, 1, device=im1.device, dtype=im1.dtype)
            )
            # 不裁剪，前后图像参数都传 im1/im2；有反向流就一起传
            if flow_b_out is not None:
                output_dict = lossfun(im1, im2, im1, im2, start, flow_f_out, flow_b_out, self.args)
            else:
                output_dict = lossfun(im1, im2, im1, im2, start, flow_f_out, self.args)

        output_dict['flow_f_out'] = flow_f_out
        if flow_b_out is not None:
            output_dict['flow_b_out'] = flow_b_out
        return output_dict
if __name__ == '__main__':
    model = UnDICnet_d().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params)

    img1 = torch.randn(1, 1, 256, 256, device=device)
    img2 = torch.randn(1, 1, 256, 256, device=device)
    start_s = torch.zeros(1, 2, 1, 1, device=device)  # 可省略

    input_dict = {'Ref': img1, 'Def': img2, 'start': start_s}
    # 如果要测包装类：
    # wrap = UnDICnet_D(args=arg).to(device)
    # out = wrap(input_dict, if_loss=True)
