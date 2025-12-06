import torch.nn as nn

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, drop=False):
    """
    concolutional layer
        args :
            batchNorm(optional) : if true, batchNorm was applied
            in_planes : the number of the input channel
            out_planes : the number of the output channel
            kernel_size :  the kernel size of the convolutional layer
            stride : the stride of the convolution
            drop(optional) : if true, dropout was applied
        Returns:
            Sequential of operation
    """
    if drop:
        if batchNorm:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes),
                nn.Dropout2d(0.4),
                nn.LeakyReLU(0.1, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                          bias=True),
                nn.Dropout2d(0.4),
                nn.LeakyReLU(0.1, inplace=True)
            )
    else:
        if batchNorm:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                          bias=True),
                nn.LeakyReLU(0.1, inplace=True)
            )


def predict_flow(in_planes, out_planes, drop=False):
    """
    convolutional layer for prediction
    default kernel size: 3
    default stride : 1
        args :
            in_planes : the number of the input channel
            out_planes : the number of the output channel
            drop(optional) : if true, dropout was applied
        Returns:
            Sequential of operation in convolutional layer for prediction
    """
    if drop:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
                             nn.Dropout2d(0.4))
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)



def deconv(in_planes, out_planes, stride=2, drop=False):
    """
    deconvolutional layer
    default kernel size: 4
        args :
            in_planes : the number of the input channel
            out_planes : the number of the output channel
            stride : the stride of the convolution
            drop(optional) : if true, dropout was applied
        Returns:
            Sequential of operation in deconvolutional layer
    """
    if drop:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=stride, padding=1, bias=False),
            nn.Dropout2d(0.4),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=stride, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )


def crop_like(input_img, target):

    if input_img.size()[2:] == target.size()[2:]:  # 输入图像的数据张量第一维是凑数的空维或者样本数，2是层数或者通道数，
        # 3，4，5则是体图像的尺寸大小，此处是判断图像大小是否一致
        return input_img
    else:
        return input_img[:, :, :target.size(2), :target.size(3)]
