import torch


def EPE(input_flow, target_flow, rank, real=False, mean=True):
    """
        Args:
            input_flow: the predicted displacement
            target_flow : the real displacement
            rank : the device ID
            real : if false, gauss weight was used; else the endpoint error was calculated directly.
            mean : if true,the average endpoint error for a single point was calculated.

        Returns:
            endpoint error
    """
    if ~real:
        b, c, h, w = input_flow.size()
        '''
        X = torch.arange(-int(w / 2), int(w / 2)).to(rank)
        Y = torch.arange(-int(h / 2), int(h / 2)).to(rank)

        # gauss weight
        [x, y] = torch.meshgrid(X, Y)
        gauss_weight = torch.exp(-(x ** 2 + y ** 2) / (h * w / 8))
        '''
        # gauss_weight = torch.from_numpy(gauss_weight).float()
        # gauss_weight = gauss_weight
        gauss_weight = (input_flow - target_flow) / torch.max(torch.abs(input_flow - target_flow))

        # weighted endpoint error
        EPE_map = torch.norm(input_flow.mul(gauss_weight) - target_flow.mul(gauss_weight), 2, 1)  # 二阶范数，差值的平方和
        # EPE_map = torch.norm(input_flow - target_flow, 2, 1)
    else:
        EPE_map = torch.norm(input_flow - target_flow, 2, 1)
    batch_size = EPE_map.size(0)

    if mean:
        # return pow(pow(EPE_map, 2).mean(), 0.5)
        return EPE_map.mean()
        #    math.sqrt(pow(EPE_map, 2).sum() / batch_size/np.prod(EPE_map.size()))
    else:
        # return pow(pow(EPE_map, 2).sum(), 0.5) / batch_size  # 矩阵范数，结果是输出与实际值差的平方和
        return EPE_map.sum() / batch_size


def multiscaleEPE(network_output, target_flow, rank, weights=None):
    """
        Args:
            network_output: the output of the network model
            target_flow : the real displacement
            rank : the device ID
            weights : the weight parameters for multi-scale predicted displacement to form the loss function

        Returns:
            loss :
    """

    def one_scale(output, target, rank):
        """
            Args:
                output: the output of the network model for a specific scale
                target : the real displacement
                rank : the device ID

            Returns:
                loss :
        """

        b, _, h, w = output.size()

        # obtain multi-scale target displacement field
        target_scaled = torch.nn.functional.interpolate(target, (h, w), mode='area')  # 根据图像尺寸实现插值和上采样

        return EPE(output, target_scaled, rank, real=False, mean=False)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        # the defalut weight for multi-scale loss
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article 不同层的尺度分辨率不同，权重也不同
    assert (len(weights) == len(network_output))

    loss = 0
    for predicted_flow, weight in zip(network_output, weights):
        loss += weight * one_scale(predicted_flow, target_flow, rank)
    return loss


def realEPE(output, target, rank):
    """
        Args:
            output: the predicted displacement
            target : the real displacement
            rank : the device ID

        Returns:
            endpoint error
    """

    b, _, h, w = target.size()
    # sub_output = output[:, :, int(w / 4):int(w * 3 / 4), int(h / 4):int(h * 3 / 4)]
    # sub_target = target[:, :, int(w / 4):int(w * 3 / 4), int(h / 4):int(h * 3 / 4)]

    # evaluate the error of the predicted result without the influence of the image edge
    sub_output = output[:, :, 8:w - 8, 8:h - 8]
    sub_target = target[:, :, 8:w - 8, 8:h - 8]

    return EPE(sub_output, sub_target, rank, real=True, mean=True)
