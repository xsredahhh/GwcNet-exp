from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class reprojection(nn.Module):

    def __init__(self):
        super(reprojection, self).__init__()

    def forward(self, disp, Q):
        b, h, w = disp.shape
        xx = torch.range(0, w - 1).cuda()
        yy = torch.range(0, h - 1).cuda()
        Y, X = torch.meshgrid([yy, xx])
        YY = torch.empty(b, h, w).cuda()
        YY[:, :, :] = Y
        XX = torch.empty(b, h, w).cuda()
        XX[:, :, :] = X
        ww = torch.ones([b, h, w]).cuda()
        Disp = torch.stack((XX, YY, disp, ww), dim=1)
        Disp = Disp.view([b, 4, h * w])
        del xx
        del yy
        del Y
        del X
        del YY
        del XX
        del ww
        Q = torch.tensor(Q, dtype=torch.float32).cuda()
        depth = torch.matmul(Q, Disp)
        W = depth[:, 3, :]
        depth = depth.permute([1, 0, 2])
        depth = torch.div(depth, W)
        depth = depth.permute([1, 0, 2])
        depth = depth.view([b, 4, h, w])
        depth = depth[:, 0:3, :, :]
        # depth = depth.permute([0, 2, 3, 1])     # 输出是b*3*h*w
        # depth = torch.zeros_like(disp).cuda()    # 此时depth和disp都是b*h*w，只考虑z方向
        # b, h, w = disp.shape
        # for i in range(b):
        #     depth[i, :, :] = (bb[i] * ff[i])/disp[i, :, :]
        return depth