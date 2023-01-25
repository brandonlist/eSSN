import pickle

import numpy as np
import torch
from torch.nn import init

from Offline.models.NeuralNet.ShallowConv import ExtractorMarkI
from torch import nn

class EncoderMarkIV(nn.Module):
    def __init__(self,n_chan,time_steps,
                 n_filter_time=40,kernel_size_time=25,kernel_size_pool=75,stride=15,cuda=False):
        super(EncoderMarkIV, self).__init__()
        self.extractor = ExtractorMarkI(kernel_size_time=kernel_size_time,kernel_size_pool=kernel_size_pool,stride=stride,
                                        n_chan=n_chan,n_filter_time=n_filter_time)
        self.f_dim = self.extractor(torch.randn(1,1,n_chan,time_steps)).shape
        if cuda:
            self.cuda()

    def forward(self,x):
        feature = self.extractor(x)
        feature = torch.reshape(feature,[x.shape[0],-1])
        return feature

class SSN(nn.Module):
    def __init__(self, n_chan, time_steps, n_classes, encoder, cuda=False):
        super(SSN, self).__init__()
        self.encoder = encoder

        self.clf = nn.Sequential(
            nn.Linear(in_features=self.encoder.f_dim[1]*self.encoder.f_dim[2],out_features=n_classes)
        )
        init.xavier_uniform_(self.clf[0].weight, gain=1)
        init.constant_(self.clf[0].bias, 0)

        if cuda:
            self.cuda()

    def forward(self, x):
        feature = self.encoder(x)
        feature = torch.reshape(feature, [x.shape[0], -1])

        logit = self.clf(feature)
        return logit

from torch.autograd import Function
class GRL(Function):
    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        return input
    @staticmethod
    def backward(ctx,grad_output):
        grad_input = grad_output.neg()
        return grad_input

class DomainClassifier(nn.Module):
    def __init__(self, n_chan, time_steps, n_classes, encoder, cuda=False):
        super(DomainClassifier, self).__init__()

        self.grl = GRL()

        self.clf = nn.Sequential(
            nn.Linear(in_features=encoder.f_dim[1]*encoder.f_dim[2],out_features=n_classes)
        )
        init.xavier_uniform_(self.clf[0].weight, gain=1)
        init.constant_(self.clf[0].bias, 0)

        if cuda:
            self.cuda()

    def init_param(self):
        init.xavier_uniform_(self.clf[0].weight, gain=1)
        init.constant_(self.clf[0].bias, 0)

    def forward(self, x):
        x = self.grl.apply(x)
        logit = self.clf(x)
        return logit

class DecoderMarkIV(nn.Module):
    def __init__(self,n_chan,time_steps,extractor,cuda=False):
        super(DecoderMarkIV, self).__init__()
        self.extractor = extractor

        tmp = torch.randn((1,n_chan,time_steps))
        if cuda:
            tmp = tmp.cuda()
            self.extractor.cuda()
        out = self.extractor(tmp)
        self.dim_ch = out.shape[1]; self.dim_ti = out.shape[2]

        kernel_size = 5
        self.channel_decoder = nn.ConvTranspose1d(in_channels=self.dim_ch,out_channels=n_chan,kernel_size=kernel_size)
        self.time_decoder = nn.Linear(in_features=self.dim_ti+kernel_size-1,out_features=time_steps)

        if cuda:
            self.cuda()

    def forward(self,feature):
        feature = torch.reshape(feature,[feature.shape[0],self.dim_ch,self.dim_ti])

        de_feature = self.channel_decoder(feature)
        re_signal = self.time_decoder(de_feature)
        return re_signal

class ScaleInvariantMSE(nn.Module):
    def __init__(self):
        super(ScaleInvariantMSE, self).__init__()
        return

    def forward(self,x1, x2):
        assert x1.shape[1]==x2.shape[1]

        cuda = False
        if x1.device.type == 'cuda' and x2.device.type == 'cuda':
            cuda = True
        if cuda:
            loss = torch.Tensor([0.]).cuda()
            k = torch.Tensor([x1.shape[1]]).cuda()
        else:
            loss = torch.Tensor([0.])
            k = torch.Tensor([x1.shape[1]])

        diff = torch.sub(x1, x2)
        loss = torch.add(loss, torch.mul(1/k,torch.mul(diff,diff).sum()))
        #loss = torch.sub(loss, torch.mul(1/torch.mul(k,k),torch.mul(diff.sum(),diff.sum())))

        return loss

class OrthogonalLoss(nn.Module):
    def __init__(self):
        super(OrthogonalLoss, self).__init__()
        return

    def forward(self, Hcs, Hct, Hps, Hpt):
        cuda = False
        if Hcs.device.type == 'cuda' and Hct.device.type == 'cuda' and Hps.device.type == 'cuda' and Hpt.device.type == 'cuda':
            cuda = True
        if cuda:
            loss = torch.Tensor([0.]).cuda()
            k = torch.Tensor([Hcs.shape[1]]).cuda()
        else:
            loss = torch.Tensor([0.])
            k = torch.Tensor([Hcs.shape[1]])

        loss = torch.add(torch.square(torch.norm(torch.matmul(Hcs,Hps.T))), torch.square(torch.norm(torch.matmul(Hct,Hpt.T))))
        loss = torch.mul(1/k, loss)
        return loss

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """计算Gram核矩阵
    source: sample_size_1 * feature_size 的数据
    target: sample_size_2 * feature_size 的数据
    kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
    kernel_num: 表示的是多核的数量
    fix_sigma: 表示是否使用固定的标准差
        return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
                        矩阵，表达形式:
                        [   K_ss K_st
                            K_ts K_tt ]
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)  # 合并在一起

    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)  # 计算高斯核中的|x-y|

    # 计算多核中每个核的bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # 高斯核的公式，exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val)  # 将多个核合并在一起

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n = int(source.size()[0])
    m = int(target.size()[0])

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)  # K_ss矩阵，Source<->Source
    XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st矩阵，Source<->Target

    YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)  # K_ts矩阵,Target<->Source
    YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)  # K_tt矩阵,Target<->Target

    loss = (XX + XY).sum() + (YX + YY).sum()
    return loss

class MMDLoss(nn.Module):
    def __init__(self):
        super(MMDLoss, self).__init__()

    def forward(self,source, target):
        return mmd(source=source, target=target)