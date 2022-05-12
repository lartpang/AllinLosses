import torch
from torch import Tensor
from typing import Tuple
from scipy.ndimage import distance_transform_edt
from torch import nn
from torch.autograd import Variable
from torch import einsum
import torch.nn.functional as F
import numpy as np
from skimage.segmentation import find_boundaries

softmax_helper = lambda x: F.softmax(x, 1)


def get_einsum_index(out_shape: Tuple):
    """
    兼容3d和2d的计算，获得einsum计算索引
    """
    if len(out_shape) == 5:
        one_index = "bcxyz->bc"
        two_index = "bcxyz, bcxyz->bc"
    elif len(out_shape) == 4:
        one_index = "bcxy->bc"
        two_index = "bcxy, bcxy->bc"
    else:
        raise ValueError("数据维度错误")
    return one_index, two_index


def get_einsum_index_original_shape(out_shape: Tuple):
    """
        兼容3d和2d的计算，获得einsum计算索引
    """
    if len(out_shape) == 5:
        two_index = "bcxyz, bcxyz->bcxyz"
    elif len(out_shape) == 4:
        two_index = "bcxy, bcxy->bcxy"
    else:
        raise ValueError("数据维度错误")
    return two_index


def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def log_cosh_smooth(loss):
    """
    https://arxiv.org/pdf/2006.14822.pdf
    :param loss:tenor.back
    :return:
    """
    return torch.log((torch.exp(loss) + torch.exp(-loss)) / 2.0)


def get_tp_fp_fn(output, target, axes=None, mask=None, square=False):
    """
    output must be (b, c, x, y(, z)))
    target must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param output:
    :param target:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(output.size())))

    shp_x = output.shape
    shp_y = target.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            target = target.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(output.shape, target.shape)]):
            # if this is the case then target is probably already a one hot encoding
            y_onehot = target
        else:
            target = target.long()
            y_onehot = torch.zeros(shp_x)
            if output.device.type == "cuda":
                y_onehot = y_onehot.cuda(output.device.index)
            y_onehot.scatter_(1, target, 1)

    tp = output * y_onehot
    fp = output * (1 - y_onehot)
    fn = (1 - output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        #
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class FocalLossV2(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLossV2, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        #
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class CrossentropyND(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """

    def forward(self, inp, target):
        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape):  # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1, )

        return super(CrossentropyND, self).forward(inp, target)


class CrossentropyNDTopK(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """

    def forward(self, inp, target):
        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape):  # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)
        with torch.no_grad():
            prob = torch.softmax(inp, -1)

        target = target.view(-1, )

        return super(CrossentropyND, self).forward(inp, target)


class TopKThreshold(CrossentropyND):
    """
    Network has to have NO LINEARITY!
    """

    def __init__(self, weight=None, ignore_index=-100, threshold=None):
        self.threshold = threshold
        super(TopKThreshold, self).__init__(weight, False, ignore_index, reduce=False)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKThreshold, self).forward(inp, target)
        with torch.no_grad():
            prob = softmax_helper(inp)
            num_classes = inp.size()[1]
            i0 = 1
            i1 = 2

            while i1 < len(prob.shape):  # this is ugly but torch only allows to transpose two axes at once
                prob = prob.transpose(i0, i1)
                i0 += 1
                i1 += 1

            prob = prob.contiguous()
            prob = prob.view(-1, num_classes)
            prob, _ = torch.max(prob, -1)

        # print('res.shape:', res.shape, 'inp.shape:', inp.shape, 'prob.shape:', prob.shape)
        res = res[prob < self.threshold]
        return res.mean()


class WeightedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """

    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight

    def forward(self, inp, target):
        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape):  # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1, )
        wce_loss = torch.nn.CrossEntropyLoss(weight=self.weight)

        return wce_loss(inp, target)


class WeightedCrossEntropyLossV2(torch.nn.Module):
    """
    WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    Network has to have NO LINEARITY!
    copy from: https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py#L121
    """

    def forward(self, output, target):
        # compute weight
        # shp_x = output.shape
        # shp_y = target.shape
        # print(shp_x, shp_y)
        # with torch.no_grad():
        #     if len(shp_x) != len(shp_y):
        #         target = target.view((shp_y[0], 1, *shp_y[1:]))

        #     if all([i == j for i, j in zip(output.shape, target.shape)]):
        #         # if this is the case then target is probably already a one hot encoding
        #         y_onehot = target
        #     else:
        #         target = target.long()
        #         y_onehot = torch.zeros(shp_x)
        #         if output.device.type == "cuda":
        #             y_onehot = y_onehot.cuda(output.device.index)
        #         y_onehot.scatter_(1, target, 1)
        # y_onehot = y_onehot.transpose(0,1).contiguous()
        # class_weights = (torch.einsum("cbxyz->c", y_onehot).type(torch.float32) + 1e-10)/torch.numel(y_onehot)
        # print('class_weights', class_weights)
        # class_weights = class_weights.view(-1)
        class_weights = torch.cuda.FloatTensor([0.2, 0.8])
        target = target.long()
        num_classes = output.size()[1]
        # class_weights = self._class_weights(inp)

        i0 = 1
        i1 = 2

        while i1 < len(output.shape):  # this is ugly but torch only allows to transpose two axes at once
            output = output.transpose(i0, i1)
            i0 += 1
            i1 += 1

        output = output.contiguous()
        output = output.view(-1, num_classes)  # shape=(vox_num, class_num)

        target = target.view(-1, )
        # print('*'*20)
        return F.cross_entropy(output, target)  # , weight=class_weights

    # @staticmethod
    # def _class_weights(input):
    #     # normalize the input first
    #     input = F.softmax(input, _stacklevel=5)
    #     flattened = flatten(input)
    #     nominator = (1. - flattened).sum(-1)
    #     denominator = flattened.sum(-1)
    #     class_weights = Variable(nominator / denominator, requires_grad=False)
    #     return class_weights


# def flatten(tensor):
#     """Flattens a given tensor such that the channel axis is first.
#     The shapes are transformed as follows:
#        (N, C, D, H, W) -> (C, N * D * H * W)
#     """
#     C = tensor.size(1)
#     # new axis order
#     axis_order = (1, 0) + tuple(range(2, tensor.dim()))
#     # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
#     transposed = tensor.permute(axis_order)
#     # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
#     transposed = transposed.contiguous()
#     return transposed.view(C, -1)
def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order).contiguous()
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)


def compute_edts_forPenalizedLoss(target):
    """
    target.shape = (batch_size, x,y,z)
    only for binary segmentation
    """
    target = np.squeeze(target)
    res = np.zeros(target.shape)
    for i in range(target.shape[0]):
        posmask = target[i]
        negmask = ~posmask
        pos_edt = distance_transform_edt(posmask)
        pos_edt = (np.max(pos_edt) - pos_edt) * posmask
        neg_edt = distance_transform_edt(negmask)
        neg_edt = (np.max(neg_edt) - neg_edt) * negmask
        res[i] = pos_edt / np.max(pos_edt) + neg_edt / np.max(neg_edt)
    return res


class DistPenalizedCE(torch.nn.Module):
    """
    Only for binary 3D segmentation

    Network has to have NO NONLINEARITY!
    """

    def forward(self, inp, target):
        # print(inp.shape, target.shape) # (batch, 2, xyz), (batch, 2, xyz)
        # compute distance map of ground truth
        with torch.no_grad():
            dist = compute_edts_forPenalizedLoss(target.cpu().numpy() > 0.5) + 1.0

        dist = torch.from_numpy(dist)
        if dist.device != inp.device:
            dist = dist.to(inp.device).type(torch.float32)
        dist = dist.view(-1, )

        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape):  # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)
        log_sm = torch.nn.LogSoftmax(dim=1)
        inp_logs = log_sm(inp)

        target = target.view(-1, )
        # loss = nll_loss(inp_logs, target)
        loss = -inp_logs[range(target.shape[0]), target]
        # print(loss.type(), dist.type())
        weighted_loss = loss * dist

        return loss.mean()


def nll_loss(input, target):
    """
    customized nll loss
    source: https://medium.com/@zhang_yang/understanding-cross-entropy-
    implementation-in-pytorch-softmax-log-softmax-nll-cross-entropy-416a2b200e34
    """
    loss = -input[range(target.shape[0]), target]
    return loss.mean()


class TopKLoss(CrossentropyND):
    """
    Network has to have NO LINEARITY!
    """

    def __init__(self, weight=None, ignore_index=-100, k=10):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape)
        res, _ = torch.topk(res.view((-1,)), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice; TODO:注意，该类容易使loss产生-inf,发生在587行求解intersection时，暂时还没有分析出原因
        Copy from: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GeneralizedDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, output, target):
        shp_x = output.shape  # (batch size,class_num,x,y,z)
        # shp_y = target.shape  # (batch size,1,x,y,z)
        # # one hot code for target
        # with torch.no_grad():
        #     if len(shp_x) != len(shp_y):
        #         target = target.view((shp_y[0], 1, *shp_y[1:]))
        #
        #     if all([i == j for i, j in zip(output.shape, target.shape)]):
        #         # if this is the case then target is probably already a one hot encoding
        #         y_onehot = target
        #     else:
        #         target = target.long()
        #         y_onehot = torch.zeros(shp_x)
        #         if output.device.type == "cuda":
        #             y_onehot = y_onehot.cuda(output.device.index)
        #         y_onehot.scatter_(1, target, 1)
        y_onehot = gt2onehot(output, target)  # (b,x,y(,z))->(b,c,x,y(,z))

        if self.apply_nonlin is not None:
            softmax_output = self.apply_nonlin(output)
        else:
            softmax_output = output
        # copy from https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        one_index, two_index = get_einsum_index(shp_x)
        w: torch.Tensor = 1 / (einsum(one_index, y_onehot).type(torch.float32) + 1e-10) ** 2
        intersection: torch.Tensor = w * einsum(two_index, softmax_output, y_onehot)
        union: torch.Tensor = w * (einsum(one_index, softmax_output) + einsum(one_index, y_onehot))
        divided: torch.Tensor = 1 - 2 * (einsum("bc->b", intersection) + self.smooth) / (
                einsum("bc->b", union) + self.smooth)
        gdc = divided.mean()

        return gdc


class GeneralizedDiceLossV2(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py#L75
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GeneralizedDiceLossV2, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, output, target):
        shp_x = output.shape  # (batch size,class_num,x,y,z)
        # shp_y = target.shape  # (batch size,1,x,y,z)
        # # one hot code for target
        # with torch.no_grad():
        #     if len(shp_x) != len(shp_y):
        #         target = target.view((shp_y[0], 1, *shp_y[1:]))
        #
        #     if all([i == j for i, j in zip(output.shape, target.shape)]):
        #         # if this is the case then target is probably already a one hot encoding
        #         y_onehot = target
        #     else:
        #         target = target.long()
        #         y_onehot = torch.zeros(shp_x)
        #         if output.device.type == "cuda":
        #             y_onehot = y_onehot.cuda(output.device.index)
        #         y_onehot.scatter_(1, target, 1)

        y_onehot = gt2onehot(output, target)  # (b,c,x,y,z)

        if self.apply_nonlin is not None:
            softmax_output = self.apply_nonlin(output)
        else:
            softmax_output = output

        input = flatten(softmax_output)
        target = flatten(y_onehot)
        target = target.float()
        target_sum = target.sum(-1)
        class_weights = Variable(1. / (target_sum * target_sum).clamp(min=self.smooth), requires_grad=False)

        intersect = (input * target).sum(-1) * class_weights
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return 1. - 2. * intersect / denominator.clamp(min=self.smooth)


class SensitivitySpecifityLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        Sensitivity-Specifity loss
        paper: http://www.rogertam.ca/Brosch_MICCAI_2015.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/df0f86733357fdc92bbc191c8fec0dcf49aa5499/niftynet/layer/loss_segmentation.py#L392
        """
        super(SensitivitySpecifityLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.r = 0.1  # weight parameter in SS paper

    def forward(self, output, target, loss_mask=None):
        shp_x = output.shape
        # shp_y = target.shape
        # class_num = shp_x[1]

        # with torch.no_grad():
        #     if len(shp_x) != len(shp_y):
        #         target = target.view((shp_y[0], 1, *shp_y[1:]))
        #
        #     if all([i == j for i, j in zip(output.shape, target.shape)]):
        #         # if this is the case then target is probably already a one hot encoding
        #         y_onehot = target
        #     else:
        #         target = target.long()
        #         y_onehot = torch.zeros(shp_x)
        #         if output.device.type == "cuda":
        #             y_onehot = y_onehot.cuda(output.device.index)
        #         y_onehot.scatter_(1, target, 1)

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))
        y_onehot = gt2onehot(output, target, axes)  # (b,c,x,y,z)

        if self.apply_nonlin is not None:
            softmax_output = self.apply_nonlin(output)
        else:
            softmax_output = output

        # no object value
        bg_onehot = 1 - y_onehot
        squared_error = (y_onehot - softmax_output) ** 2
        specificity_part = sum_tensor(squared_error * y_onehot, axes) / (sum_tensor(y_onehot, axes) + self.smooth)
        sensitivity_part = sum_tensor(squared_error * bg_onehot, axes) / (sum_tensor(bg_onehot, axes) + self.smooth)

        ss = self.r * specificity_part + (1 - self.r) * sensitivity_part

        if not self.do_bg:
            if self.batch_dice:
                ss = ss[1:]
            else:
                ss = ss[:, 1:]
        ss = ss.mean()

        return ss


def gt2onehot(output, target, axes=None):
    """
    output must be (b, c, x, y(, z)))
    target must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param output:
    :param target:
    :param axes:
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(output.size())))

    shp_x = output.shape
    shp_y = target.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            target = target.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(output.shape, target.shape)]):
            # if this is the case then target is probably already a one hot encoding
            y_onehot = target
        else:
            target = target.long()
            y_onehot = torch.zeros(shp_x)
            if output.device.type == "cuda":
                y_onehot = y_onehot.cuda(output.device.index)
            y_onehot.scatter_(1, target, 1)

    return y_onehot


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1e-5,
                 square=False):
        """
        Drozdzal et al. https://arxiv.org/abs/1608.04117
        """
        super(SoftDiceLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, output, target, loss_mask=None):
        shp_x = output.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            output = self.apply_nonlin(output)

        tp, fp, fn = get_tp_fp_fn(output, target, axes, loss_mask, self.square)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return 1 - dc


class SoftDiceLossV2(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        paper: Milletari et al. https://arxiv.org/abs/1606.04797
        """
        super(SoftDiceLossV2, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            output = self.apply_nonlin(x)  # (b,c,x,y,z)

        gt_onehot = gt2onehot(output, y, axes)  # (b,c,x,y,z)

        intersection = sum_tensor(output * gt_onehot, axes, keepdim=False)
        ground_o = sum_tensor(gt_onehot ** 2, axes, keepdim=False)
        pred_o = sum_tensor(output ** 2, axes, keepdim=False)
        dc = 2.0 * (intersection + self.smooth) / (ground_o + pred_o + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return - dc


class IoULoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://link.springer.com/chapter/10.1007/978-3-319-50835-1_22

        """
        super(IoULoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        iou = (tp + self.smooth) / (tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                iou = iou[1:]
            else:
                iou = iou[:, 1:]
        iou = iou.mean()

        return -iou


class TverskyLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://arxiv.org/pdf/1706.05721.pdf
        """
        super(TverskyLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.alpha = 0.3
        self.beta = 0.7

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                tversky = tversky[1:]
            else:
                tversky = tversky[:, 1:]
        tversky = tversky.mean()

        return -tversky


class FocalTverskyLoss(nn.Module):
    """
    paper: https://arxiv.org/pdf/1810.07842.pdf
    author code: https://github.com/nabsabraham/focal-tversky-unet/blob/347d39117c24540400dfe80d106d2fb06d2b99e1/losses.py#L65
    """

    def __init__(self, tversky_kwargs, gamma=0.75):
        super(FocalTverskyLoss, self).__init__()
        self.gamma = gamma
        self.tversky = TverskyLoss(**tversky_kwargs)

    def forward(self, output, target):
        tversky_loss = 1 + self.tversky(output, target)  # = 1-tversky(output, target)
        focal_tversky = torch.pow(tversky_loss, self.gamma)
        return focal_tversky


class AsymLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8573779
        """
        super(AsymLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.beta = 1.5

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)  # shape: (batch size, class num)
        weight = (self.beta ** 2) / (1 + self.beta ** 2)
        asym = (tp + self.smooth) / (tp + weight * fn + (1 - weight) * fp + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                asym = asym[1:]
            else:
                asym = asym[:, 1:]
        asym = asym.mean()

        return -asym


class DiceWithCrossentropyNDLoss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum"):
        super(DiceWithCrossentropyNDLoss, self).__init__()
        self.aggregate = aggregate
        self.ce = CrossentropyND(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, output, target):
        dc_loss = self.dc(output, target)
        ce_loss = self.ce(output, target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son")  # reserved for other stuff (later)
        return result, dc_loss, ce_loss


class PenaltyGeneralizedDiceLoss(nn.Module):
    """
    paper: https://openreview.net/forum?id=H1lTh8unKN
    """

    def __init__(self, gdice_kwargs):
        super(PenaltyGeneralizedDiceLoss, self).__init__()
        self.k = 2.5
        self.gdc = GeneralizedDiceLoss(apply_nonlin=softmax_helper, **gdice_kwargs)

    def forward(self, output, target):
        gdc_loss = self.gdc(output, target)
        penalty_gdc = gdc_loss / (1 + self.k * (1 - gdc_loss))

        return penalty_gdc


class DiceWithTopKLoss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum"):
        super(DiceWithTopKLoss, self).__init__()
        self.aggregate = aggregate
        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, output, target):
        dc_loss = self.dc(output, target)
        ce_loss = self.ce(output, target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son")  # reserved for other stuff (later?)
        return result, dc_loss, ce_loss


class ExpLogLoss(nn.Module):
    """
    paper: 3D Segmentation with Exponential Logarithmic Loss for Highly Unbalanced Object Sizes
    https://arxiv.org/pdf/1809.00076.pdf
    """

    def __init__(self, soft_dice_kwargs, wce_kwargs, gamma=0.3):
        super(ExpLogLoss, self).__init__()
        self.wce = WeightedCrossEntropyLoss(**wce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        self.gamma = gamma

    def forward(self, output, target):
        dc_loss = -self.dc(output, target)  # weight=0.8
        wce_loss = self.wce(output, target)  # weight=0.2
        # with torch.no_grad():
        #     print('dc loss:', dc_loss.cpu().numpy(), 'ce loss:', ce_loss.cpu().numpy())
        #     a = torch.pow(-torch.log(torch.clamp(dc_loss, 1e-6)), self.gamma)
        #     b = torch.pow(-torch.log(torch.clamp(ce_loss, 1e-6)), self.gamma)
        #     print('ExpLog dc loss:', a.cpu().numpy(), 'ExpLogce loss:', b.cpu().numpy())
        #     print('*'*20)
        explog_loss = 0.8 * torch.pow(-torch.log(torch.clamp(dc_loss, 1e-6)), self.gamma) + \
                      0.2 * wce_loss

        return explog_loss


class DiceWithFocalLoss(nn.Module):
    def __init__(self, soft_dice_kwargs, focal_kwargs):
        super(DiceWithFocalLoss, self).__init__()
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        self.focal = FocalLoss(apply_nonlin=softmax_helper, **focal_kwargs)

    def forward(self, output, target):
        dc_loss = self.dc(output, target)
        focal_loss = self.focal(output, target)

        result = dc_loss + focal_loss
        return result, dc_loss, focal_loss


class GeneralizedDiceWithFocalLoss(nn.Module):
    def __init__(self):
        super(GeneralizedDiceWithFocalLoss, self).__init__()
        self.gdc = GeneralizedDiceLossV2(apply_nonlin=softmax_helper)
        self.focal = FocalLoss(apply_nonlin=softmax_helper)
        self.smooth = False

    def forward(self, output, target):
        gdc_loss = self.gdc(output, target)
        focal_loss = self.focal(output, target)
        if target.dim() == 4:
            target[target == 4] = 3  # label [2] -> [0]
            target = expand_target(target, n_class=output.size()[1])  # [N,H,W,D] -> [N,1，H,W,D]
        entropy_criterion = nn.BCEWithLogitsLoss()
        bce_l = entropy_criterion(output, target)
        result = gdc_loss + focal_loss + bce_l
        if self.smooth:
            result = log_cosh_smooth(result)
        return result, gdc_loss, focal_loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    def __init__(self, reduction='mean'):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction

    def prob_flatten(self, input, target):
        assert input.dim() in [4, 5]
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, output, target):
        num_classes = output.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (target == c).float()
            if num_classes == 1:
                output_c = output[:, 0]
            else:
                output_c = output[:, c]
            loss_c = (torch.autograd.Variable(target_c) - output_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, output, target):
        """
        output:(batch size, class_num, x,y,z)
        target:(batch size, 1, x,y,z)
        """
        # print(output.shape, target.shape) # (batch size, class_num, x,y,z), (batch size, 1, x,y,z)
        output, target = self.prob_flatten(output, target)
        # print(output.shape, target.shape)
        losses = self.lovasz_softmax_flat(output, target)
        return losses


def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """

    img_gt = img_gt.astype(np.uint8)

    gt_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        for c in range(1, out_shape[1]):  # channel or class
            posmask = img_gt[b][c].astype(np.bool_)
            if posmask.any():
                negmask = ~posmask
                posdis = distance_transform_edt(posmask)
                negdis = distance_transform_edt(negmask)
                boundary = find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = negdis - posdis
                sdf[boundary == 1] = 0
                gt_sdf[b][c] = sdf

    return gt_sdf


class BoudaryLoss(nn.Module):
    def __init__(self):
        """
        compute boudary loss
        only compute the loss of foreground
        ref: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L74
        """
        super(BoudaryLoss, self).__init__()

    def forward(self, output, target):
        """
        output: (batch_size, class, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        bound: precomputed distance map, shape (batch_size, class, x,y,z)
        """
        if False: output = softmax_helper(output)
        out_shape = output.shape
        # with torch.no_grad():
        #     if len(output.shape) != len(target.shape):
        #         target = target.view((target.shape[0], 1, *target.shape[1:]))
        #
        #     if all([i == j for i, j in zip(output.shape, target.shape)]):
        #         # if this is the case then target is probably already a one hot encoding
        #         y_onehot = target
        #     else:
        #         target = target.long()
        #         y_onehot = torch.zeros(output.shape)
        #         if output.device.type == "cuda":
        #             y_onehot = y_onehot.cuda(output.device.index)
        #         y_onehot.scatter_(1, target, 1)
        y_onehot = gt2onehot(output, target, axes)  # (b,c,x,y,z)
        gt_sdf = compute_sdf(y_onehot.cpu().numpy(), out_shape)

        phi = torch.from_numpy(gt_sdf)
        if phi.device != output.device:
            phi = phi.to(output.device).type(torch.float32)
        # pred = output[:, 1:, ...].type(torch.float32)
        # phi = phi[:,1:, ...].type(torch.float32)
        two_index = get_einsum_index_original_shape(out_shape)
        multipled = einsum(two_index, output[:, 1:, ...], phi[:, 1:, ...])
        bd_loss = multipled.mean()

        return bd_loss


class DiceWithBoundaryLoss(nn.Module):
    def __init__(self, soft_dice_kwargs, aggregate="sum"):
        super(DiceWithBoundaryLoss, self).__init__()
        self.aggregate = aggregate
        self.dc = SoftDiceLoss(apply_nonlin=None, **soft_dice_kwargs)
        self.bd = BoudaryLoss()

    def forward(self, output, target, alpha=0.01):
        dc_loss = self.dc(output, target)
        bd_loss = self.bd(output, target)
        if self.aggregate == "sum":
            result = alpha * dc_loss + (1 - alpha) * bd_loss
        else:
            raise NotImplementedError("nah son")
        return result, dc_loss, bd_loss


class GeneralizedDiceWithBoundaryLoss(nn.Module):
    def __init__(self):
        super(GeneralizedDiceWithBoundaryLoss, self).__init__()
        aggregate = "sum"
        self.aggregate = aggregate
        self.dc = GeneralizedDiceLossV2()
        self.bd = BoudaryLoss()

    def forward(self, output, target, alpha=0.01):
        dc_loss = self.dc(output, target)
        bd_loss = self.bd(output, target)
        if self.aggregate == "sum":
            result = alpha * dc_loss + (1 - alpha) * bd_loss
        else:
            raise NotImplementedError("nah son")
        return result, dc_loss, bd_loss


#####################################################
def compute_dtm(img_gt, out_shape, is_label=True):
    """
     compute the distance transform map of foreground in ground gruth or prediction.
     input: segmentation, shape = (batch_size, class, x, y, z)
     output: the foreground Distance Map (SDM)
     dtm(x) = 0; x in segmentation boundary
              inf|x-y|; x in segmentation

    """

    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        for c in range(1, out_shape[1]):  # class; exclude the background class
            if is_label:
                posmask = img_gt[b][c].astype(np.bool_)
            else:
                posmask = img_gt[b][c] > 0.5
            if posmask.any():
                posdis = distance_transform_edt(posmask)
                fg_dtm[b][c] = posdis

    return fg_dtm


class HDLoss(nn.Module):
    def __init__(self):
        """
        compute haudorff loss for binary segmentation
        https://arxiv.org/pdf/1904.10030v1.pdf
        """
        super(HDLoss, self).__init__()

    def forward(self, output, target):
        """
        output: (batch_size, c, x,y,z)
        target: ground truth, shape: (batch_size, c, x,y,z)
        """
        output = softmax_helper(output)
        # # one hot code for target
        # with torch.no_grad():
        #     if len(output.shape) != len(target.shape):
        #         target = target.view((target.shape[0], 1, *target.shape[1:]))
        #
        #     if all([i == j for i, j in zip(output.shape, target.shape)]):
        #         # if this is the case then target is probably already a one hot encoding
        #         y_onehot = target
        #     else:
        #         target = target.long()
        #         y_onehot = torch.zeros(output.shape)
        #         if output.device.type == "cuda":
        #             y_onehot = y_onehot.cuda(output.device.index)
        #         y_onehot.scatter_(1, target, 1)
        # # print('hd loss_function.py', output.shape, y_onehot.shape)

        y_onehot = gt2onehot(output, target)  # (b,c,x,y,z)

        with torch.no_grad():
            output_shape = output.shape
            # pc_dist = compute_pred_dtm(output.cpu().numpy(), output.shape)
            # gt_dist = compute_gt_dtm(y_onehot.cpu().numpy(), output.shape)
            pc_dist = compute_dtm(output.cpu().numpy(), output_shape, is_label=False)
            gt_dist = compute_dtm(y_onehot.cpu().numpy(), output_shape, is_label=True)
            dist = pc_dist ** 2 + gt_dist ** 2  # \alpha=2 in eq(8)
            # print('pc_dist.shape: ', pc_dist.shape, 'gt_dist.shape', gt_dist.shape)

        pred_error = (output - y_onehot) ** 2

        dist = torch.from_numpy(dist)
        if dist.device != pred_error.device:
            dist = dist.to(pred_error.device).type(torch.float32)

        two_index = get_einsum_index_original_shape(output_shape)
        multipled = einsum(two_index, pred_error[:, 1:, ...], dist[:, 1:, ...])
        hd_loss = multipled.mean()

        return hd_loss


class DiceWithHDLoss(nn.Module):
    def __init__(self, soft_dice_kwargs, aggregate="sum"):
        super(DiceWithHDLoss, self).__init__()
        self.aggregate = aggregate
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        self.hd = HDLoss()

    def forward(self, output, target):
        dc_loss = self.dc(output, target)
        hd_loss = self.hd(output, target)
        if self.aggregate == "sum":
            with torch.no_grad():
                alpha = hd_loss / (dc_loss + 1e-5)
            result = alpha * dc_loss + hd_loss
        else:
            raise NotImplementedError("nah son")
        return result, dc_loss, hd_loss


def GeneralizedDiceLossV3(output, target, eps=1e-5, weight_type='square'):  # Generalized dice loss
    """
        该实现方式更容易理解多分类问题
        Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
    """

    # target = target.float()

    if target.dim() == 4:
        target[target == 4] = 3  # label [2] -> [0]
        target = expand_target(target, n_class=output.size()[1])  # [N,H,W,D] -> [N,1，H,W,D]

    # output = flatten(output)[1:, ...]  # transpose [N,1，H,W,D] -> [1，N,H,W,D] -> [3, N*H*W*D] voxels
    output = flatten(output)[0:, ...]  # transpose [N,1，H,W,D] -> [1，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[0:, ...]  # [class, N*H*W*D]

    target_sum = target.sum(-1)  # sub_class_voxels [3,1] -> 3个voxels
    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :', weight_type)

    intersect = (output * target).sum(-1)
    intersect_sum = (intersect * class_weights).sum()
    denominator = (output + target).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps
    # print("intersect_sum:{},\tdenominator_sum:{}".format(intersect_sum.data, denominator_sum.data))
    loss = 1 - 2. * (intersect_sum / denominator_sum)

    if len(intersect) == 3:
        loss1 = 2 * intersect[0] / (denominator[0] + eps)
        loss2 = 2 * intersect[1] / (denominator[1] + eps)
        loss3 = 2 * intersect[2] / (denominator[2] + eps)
        print(
            '\n0:{:.4f} | 1:{:.4f} | 2:{:.4f} | - | Total loss:{:.4f}'.format(loss1.data, loss2.data, loss3.data,
                                                                              loss.data))

    return loss


def expand_target(x, n_class, mode='softmax'):
    """
        Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :return: 5D output image (NxCxDxHxW)
    """
    assert x.dim() == 4

    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == 'softmax':
        for c in range(n_class):
            if c == 1:
                xx[:, c, :, :, :] = (x >= c)  # 动脉+动脉瘤
            else:
                xx[:, c, :, :, :] = (x == c)

    if mode.lower() == 'sigmoid':
        xx[:, 0, :, :, :] = (x == 0)
        xx[:, 1, :, :, :] = (x == 1)
        xx[:, 2, :, :, :] = (x == 3)
    return xx.to(x.device)


if __name__ == "__main__":
    output = torch.zeros(2, 2, 64, 64, 64)
    output[:, 0, 10:20, 10:20, 10:20] = 0
    output[:, 1, 12:20, 12:20, 12:20] = 1

    target = torch.zeros(2, 64, 64, 64)
    # target[:, 5:15, 5:15, 5:15] = 1
    target[:, 10:20, 10:20, 10:20] = 1

    # dice_loss = SoftDiceLoss(smooth=1e-5)
    # dice_lv = dice_loss(output, target)
    # print(dice_lv)
    #
    # gdl = GeneralizedDiceLoss()
    # dice_lv = gdl(output, target)
    # print(dice_lv)
    # BDL = BoudaryLoss()
    # print(BDL(output, target))
    # para = dict(batch_dice=False, do_bg=True, smooth=1e-5)
    # DBDL = DiceWithBoundaryLoss(para)
    # print(DBDL(output, target))
    # DHDL = DiceWithHDLoss(para)
    # print(DHDL(output, target))
    # GDBL = GeneralizedDiceWithBoundaryLoss()
    # print(GDBL(output, target))

    GDFL = GeneralizedDiceWithFocalLoss()
    print(GDFL(output, target))