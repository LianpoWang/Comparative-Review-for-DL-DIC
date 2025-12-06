import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(inplace=False, negative_slope=0.1)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)
def get_grid(x):
    grid_H = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    grid_V = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([grid_H, grid_V], 1)

    grids_cuda = grid.float().requires_grad_(False).to(x.device)
    return grids_cuda

class WarpingLayer(nn.Module):
    def __init__(self):
        super(WarpingLayer, self).__init__()
  #div_flow=0.05

    def forward(self,img, flo):
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
        vgrid = grid + flo  # B,2,H,W
        # 图二的每个像素坐标加上它的光流即为该像素点对应在图一的坐标
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :]/ max(W1 - 1, 1) - 1.0
        # 取出光流v这个维度，原来范围是0~W-1，再除以W-1，范围是0~1，再乘以2，范围是0~2，再-1，范围是-1~1
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H1 - 1, 1) - 1.0  # 取出光流u这个维度，同上
        vgrid = vgrid.permute(0, 2, 3, 1)  # from B,2,H,W -> B,H,W,2，为什么要这么变呢？是因为要配合grid_sample这个函数的使用

        x_warp = F.grid_sample(img, vgrid, align_corners=True)
        mask = torch.ones(img.size(), requires_grad=False).to(img.device)

        mask = F.grid_sample(mask, vgrid, align_corners=True)

        mask = (mask >= 1.0).float()

        return x_warp * mask
        # mask = torch.ones(img.size(), requires_grad=False).to(img.device)
        # mask = F.grid_sample(mask, grid, align_corners=True)
        # mask = (mask >= 1.0).float()

        # return output

    # def forward(self, x, flow, height_im, width_im, div_flow):
    #     #torch.Size([2, 128, 8, 8])，torch.Size([2, 2, 8, 8])
    #
    #     flo_list = []
    #     flo_w = flow[:, 0]
    #     flo_h = flow[:, 1]
    #     flo_list.append(flo_w)
    #     flo_list.append(flo_h)
    #
    #     flow_for_grid = torch.stack(flo_list).transpose(0, 1)
    #
    #     grid = torch.add(get_grid(x), flow_for_grid).transpose(1, 2).transpose(2, 3)
    #
    #     x_warp = F.grid_sample(x, grid, align_corners=True)
    #
    #     mask = torch.ones(x.size(), requires_grad=False).to(x.device)
    #
    #     mask = F.grid_sample(mask, grid, align_corners=True)
    #
    #     mask = (mask >= 1.0).float()
    #
    #     return x_warp * mask
# class ConvGRU(nn.Module):
#     def __init__(self, input_dim=2+81,hidden_dim=2):
#         super(ConvGRU, self).__init__()
#
#         self.convz1 = nn.Conv2d(input_dim, hidden_dim, (1, 5), padding=(0, 2))
#         self.convr1 = nn.Conv2d( input_dim, hidden_dim, (1, 5), padding=(0, 2))
#         self.convq1 = nn.Conv2d(input_dim, hidden_dim, (1, 5), padding=(0, 2))
#
#         self.convz2 = nn.Conv2d(input_dim, hidden_dim, (5, 1), padding=(2, 0))
#         self.convr2 = nn.Conv2d( input_dim, hidden_dim, (5, 1), padding=(2, 0))
#         self.convq2 = nn.Conv2d(input_dim, hidden_dim, (5, 1), padding=(2, 0))
#
#         self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
#         self.conv2 = nn.Conv2d(32, 2, 3, padding=1)
#     def forward(self, h, x):
#
#
#         # vertical
#         hx = torch.cat([h, x], dim=1)
#         z = torch.sigmoid(self.convz1(hx))
#         r = torch.sigmoid(self.convr1(hx))
#         q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
#         h = (1 - z) * h + z * q
#
#         # vertical
#         hx = torch.cat([h, x], dim=1)
#         z = torch.sigmoid(self.convz2(hx))
#         r = torch.sigmoid(self.convr2(hx))
#         q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
#         h = (1 - z) * h + z * q
#
#         h = self.conv2(self.conv1(h))
#         return h

class FlowHead(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, input_dim=96,hidden_dim=2):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(input_dim+input_dim+64, input_dim, 3, padding=1)
        self.convr = nn.Conv2d(input_dim+input_dim+64, input_dim, 3, padding=1)
        self.convq = nn.Conv2d(input_dim+input_dim+64, input_dim, 3, padding=1)


    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        return h

class BasicUpdateBlock(nn.Module):
    def __init__(self,  input_dim=96,hidden_dim=2,scales=2):
        super(BasicUpdateBlock, self).__init__()


        self.flowgru = ConvGRU(input_dim,hidden_dim)

        self.flow_head = FlowHead(input_dim, hidden_dim)
        self.mask =MaskBlock(input_dim,scales)

    def forward(self, net, inp, motion_features):

        inp = torch.cat([inp, motion_features], dim=1)
        net = self.flowgru(net, inp)


        # motion_features1 = self.encoder(corr, x1, y1, )
        # net=self.flowgru(net,motion_features)
        delta_flow = self.flow_head(net)
        mask = .25 * self.mask(net)
        # scale mask to balence gradients
        return net, mask, delta_flow


class BasicEncoder(nn.Module):
    def __init__(self, input_dim=3 ,norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)


        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        # self.conv1 = nn.Conv2d(1,16, kernel_size=7, stride=1, padding=3)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3)

        # self.layer1 = self._make_layer(64, stride=1)
        # self.layer2 = self._make_layer(96, stride=2)
        # self.layer3 = self._make_layer(128, stride=2)

        self.relu1 =nn.LeakyReLU(inplace=False, negative_slope=0.1)

        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(16, 16,kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True))
        # self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2)
        # self.relu2 = nn.ReLU(inplace=True)
        self.in_planes =64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
        self.layer4 = self._make_layer(256, stride=2)
        # self.layer5 = self._make_layer(128, stride=2)
        # self.layer6 = self._make_layer(196, stride=2)

        # output convolution
        # self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        # x= self.conv2(x)
        # x = self.conv2(x)
        # x = self.norm1(x)
        # x = self.relu2(x)
        disp1 = self.layer1(x)

        disp2 = self.layer2(disp1)

        disp3 = self.layer3(disp2)

        disp4 = self.layer4(disp3)

        # disp5 = self.layer5(disp4)
        #
        # disp6 = self.layer6(disp5)

        return  disp4, disp3, disp2,

class MaskBlock(nn.Module):
    def __init__(self,in_planes,scales):
        super().__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(in_planes, 256, 3, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(256, scales*scales*9, 1, padding=0))
    def forward(self, x):
        return 0.25*self.mask(x)

class BasicMotionEncoder(nn.Module):
    def __init__(self, cor_planes,dis_planes):
        super(BasicMotionEncoder, self).__init__()
        # cor_planes = args['corr_levels'] * (2*args['corr_radius'] + 1)**2

        self.convc1 = nn.Conv2d(cor_planes, 32, 1, padding=0)
        self.convc2 = nn.Conv2d(32, 32, 3, padding=1)


        self.convf1 = nn.Conv2d(2,32, 3, padding=1)
        self.convf2 = nn.Conv2d(32, 32, 3, padding=1)

        self.convx1 = nn.Conv2d(dis_planes, 32, 3, padding=1)
        self.convx2 = nn.Conv2d(dis_planes, 32, 3, padding=1)

        self.convdis = nn.Conv2d(dis_planes, 32, 3, padding=1)


        self.conv = nn.Conv2d(32+32+32+32+32, 64, 3, padding=1)
    def forward(self, corr,flow,x1,x2,dis):

        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))

        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        x1 = F.relu(self.convx1(x1))

        x2=  F.relu(self.convx2(x2))
        dis = F.relu(self.convdis(dis))

        cor_flo = torch.cat([cor, x1, x2,dis,flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return out
class SmallMotionEncoder(nn.Module):
    def __init__(self, cor_planes,dis_planes):
        super(SmallMotionEncoder, self).__init__()
        # cor_planes = args['corr_levels'] * (2*args['corr_radius'] + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 32, 1, padding=0)
        self.convc2 = nn.Conv2d(32,32, 3, padding=1)
        self.convx1 = nn.Conv2d(dis_planes, 64, 3, padding=1)
        self.convx2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(32+64, 64, 3, padding=1)
    def forward(self,corr,x1):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        x1 = F.relu(self.convx1(x1))
        x1=  F.relu(self.convx2(x1))
        cor_flo = torch.cat([cor, x1], dim=1)
        out = F.relu(self.conv(cor_flo))
        return out
class DIC(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self,max_disp):
        super().__init__( )
        self.backbone = BasicEncoder(input_dim=3) #2819232
        self.mode = 2
        self.context  =  BasicEncoder(input_dim=6)
        self.warping_layer = WarpingLayer()
        self._div_flow=1
        self.planes= (max_disp*2+1)*(max_disp*2+1)
        self.max_disp = max_disp
        # self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)
        #
        self.updata_layers = nn.ModuleList()
        self.mask_layers = nn.ModuleList()
        self.in_planes = [256,128,64]
        self.Rate = [2, 2, 4]
        # self.CBAM = nn.ModuleList()
        self.MotionEncoder =nn.ModuleList()

        for i_layer in range(3):
            GRU = BasicUpdateBlock(self.in_planes[i_layer]//2,64,self.Rate[i_layer])
            # CBAM = cbam_block(self.planes)
            if i_layer == 0:
                Motion = SmallMotionEncoder(self.planes, self.in_planes[i_layer])
            else:
                Motion = BasicMotionEncoder(self.planes, self.in_planes[i_layer])

            self.updata_layers.append(GRU)
            # self.CBAM.append(CBAM)
            self.MotionEncoder.append(Motion)
        #     # 注意这里构建的stage和论文图中有些差异
        #     # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
        #     layers = MaskBlock(in_planes=in_dim[i_layer],hide_planes=128,out_planes=out_dim[i_layer])
        #     self.mask_layers.append(layers)
        #
        # self.mask_layer1 = MaskBlock(256,2)
        # self.mask_layer1 = MaskBlock(128, 2)
        # self.mask_layer1 = MaskBlock(64, 2)
        # self.mask_layer1 = MaskBlock(32, 2)
        # self.mask_layer1 = MaskBlock(16, 2)
        # self.mask_layer1 = MaskBlock(512, 128, 32)
        # self.mask_layer1 = MaskBlock(512, 128, 32)
        #
        # self.mask_layer1 = MaskBlock(512, 128, 32)
        # self.mask_layer1 = MaskBlock(512, 128, 32)
    def upsample2d_as(self,inputs, mode="bilinear"):
        _, _, h, w = inputs.size()
        return F.interpolate(inputs, [h*2, w*2], mode=mode, align_corners=True)
    def upsample_as(self,inputs,outputs, mode="bilinear"):
        _, _, h, w =outputs.size()
        return F.interpolate(inputs, [h, w], mode=mode, align_corners=True)
    def compute_cost_volume(self,feat1, feat2, max_disp=4):
        """
        only implemented for:
            kernel_size = 1
            stride1 = 1
            stride2 = 1
        """
        _, _, height, width = feat1.size()
        num_shifts = 2 * max_disp + 1
        feat2_padded = F.pad(feat2, (max_disp, max_disp, max_disp, max_disp), "constant", 0)
        cost_list = []
        for i in range(num_shifts):
            for j in range(num_shifts):
                corr = torch.mean(feat1 * feat2_padded[:, :, i:(height + i), j:(width + j)], axis=1, keepdims=True)
                cost_list.append(corr)
        cost_volume = torch.cat(cost_list, axis=1)
        return cost_volume


    def upsample_flow(self, flow, mask,scale):
        # mask.shape= torch.Size([1, 576, 32, 32])
        # flow = torch.Size([1, 2, 32, 32])
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, scale, scale, H, W) #(2,1,1,2,2,4,4)
        mask = torch.softmax(mask, dim=2) #torch.Size([2, 18, 16]
        up_flow = F.unfold(8* flow, [3, 3], padding=1)  # torch.Size([1, 18, 1024])
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)#(2,2,9,1,1,4,4)  # (2,2,9,2,2,4,4
        # )
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, scale * H, scale * W)

    def forward(self, img0, img1,iters):
        B,C,H,W=img0.shape
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        pyramid=self.backbone(concat)
        #init
        b_size, _, h_x1, w_x1, = pyramid[0].size()
        init_dtype = pyramid[0].dtype
        init_device =pyramid[0].device
        flow = torch.zeros(B, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float().detach()
        context = torch.cat((img0, img1), dim=1)  # [B, 2C, H, W]
        context = self.context(context )
        # 0
        # torch.Size([4, 512, 4, 4])
        # 1
        # torch.Size([4, 256, 8, 8])
        # 2
        # torch.Size([4, 128, 16, 16])
        # 3
        # torch.Size([4, 64, 32, 32])
        # 4
        # torch.Size([4, 32, 64, 64])


        # 0
        # torch.Size([4, 256, 16, 16])
        # 1
        # torch.Size([4, 128, 32, 32])
        # 2
        # torch.Size([4, 64, 64, 64])
        output_dict = []
        for l,(x,y) in  enumerate(zip(pyramid,context)):

            x1,x2 = torch.split(x, [B, B], dim=0)

            _, c, hdim, wdim = y.shape
            net, inp = torch.split(y, [c//2, c//2], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

            if l == 2:  # 迭代细化。
                flow1= flow.detach()
                output_dict.append(flow1)
                for i in range(iters):

                    x2_warp = self.warping_layer(x2, flow1)
                    out_corr = self.compute_cost_volume(x1, x2_warp, self.max_disp)
                    # out_corr = self.CBAM[l](out_corr)
                    motion = self.MotionEncoder[l](out_corr,flow1,x1, x2_warp,x2_warp - x1)
                    net, mask, delt_flow = self.updata_layers[l](net, inp, motion )
                    flow1 = flow1 + delt_flow

                    flow= self.upsample_flow(flow1, mask, 4)
                    output_dict.append(flow)
                break
            if l == 0:
                x2_warp = x2
            else:
                x2_warp = self.warping_layer(x2, flow)
            out_corr = self.compute_cost_volume(x1, x2_warp,self.max_disp)
            if l == 0:
                motion = self.MotionEncoder[l](out_corr, x1)
            else:
                motion = self.MotionEncoder[l](out_corr, flow, x1, x2_warp, x2_warp - x1)
            # out_corr = self.CBAM[l](out_corr)
            #


            net, mask, flow=self.updata_layers[l](net, inp, motion)
            flow = self.upsample_flow(flow, mask, 2)
        # return output_dict
        # if self.training:
        #
        #     return output_dict
        # else:
            # final_disp = self.upsample2d_as(output_dict[-1],  mode="bilinear")
            # return self.upsample_as(output_dict[-1],img1)
        return output_dict

def compute_cost_volume(feat1, feat2, max_disp=4):
        """
        only implemented for:
            kernel_size = 1
            stride1 = 1
            stride2 = 1
        """
        _, _, height, width = feat1.size()
        num_shifts = 2 * max_disp + 1
        feat2_padded = F.pad(feat2, (max_disp, max_disp, max_disp, max_disp), "constant", 0)
        cost_list = []
        for i in range(num_shifts):
            for j in range(num_shifts):
                corr = torch.mean(feat1 * feat2_padded[:, :, i:(height + i), j:(width + j)], axis=1, keepdims=True)
                print(corr.shape)
                cost_list.append(corr)
        cost_volume = torch.cat(cost_list, axis=1)
        return cost_volume


if __name__ == '__main__':
    feat1 = torch.randn(2, 1, 256, 256)
    # d = feat1 * feat1
    # print(d.shape)
    # compute_cost_volume(feat1, feat1, max_disp=4)
    model = DIC(max_disp=4)  # 3131424
    output = model(feat1 ,feat1 , 12)
    print(output)

# if __name__ == '__main__':
#     model = DIC(max_disp=4)  #3131424
#     # model=BasicEncoder()
#     img1 = torch.randn(2, 1, 256, 256)
#     # img1 = torch.randn(1, 1, 501, 2000)
#     # img2 = torch.randn(2, 3, 256, 256)
#     # p=compute_cost_volume(img1, img1, max_disp=1)
#     # p = corr(img1, img1)
#     # print(p.shape)
#     #
#     # # output = model(img1)
#     output = model(img1,img1,12)
#     # # print(output[0].shape)
#     # # print(len(output))
#     # # print(output[-1].shape)
#     # # print(output[0].shape)
#     # # print(output[4].shape)
#     # #
#     num_params = sum(param.numel() for param in model.parameters())
#     print(num_params)
#
#     print(model.context)
#     # for param in model.parameters():
#     #     print(param)
