import torch
import torch.nn as nn
from utils.activations import *
import math

# Adapted from YoloV5
class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, k=1, s=1, p=0, g=1, act=SiLU(), bias=False):
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, k, s, p, groups = g, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut = True):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        self.cv1 = Conv(in_channels, out_channels, 1, 1)
        self.cv2 = Conv(out_channels, out_channels, 3, 1, p = 1)
        self.add = in_channels == out_channels and shortcut

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut = True):
        super().__init__()
        self.cv1 = Conv(in_channels, out_channels//2, 1, 1)
        self.cv2 = Conv(in_channels, out_channels//2, 1, 1)
        self.cv3 = Conv(2 * out_channels//2, out_channels, 1)
        self.m = Bottleneck(out_channels//2, out_channels//2, shortcut = shortcut)

    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], 1))

class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, k=(5, 9, 13)):
        super(SPP, self).__init__()
        self.cv1 = Conv(in_channels, out_channels // 2, 1, 1)
        self.cv2 = Conv(out_channels // 2 * (len(k) + 1), out_channels, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Detect(nn.Module):
    stride = None  # strides computed during build

    def __init__(self, nc=1, anchors=(), ch=(), stride=(), export=False):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor 85
        self.nl = len(anchors)  # number of detection layers 3
        self.na = len(anchors[0]) // 2  # number of anchors 3
        self.grid = [torch.zeros(1)] * self.nl  # init grid 
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv  
        self.stride = torch.Tensor(stride)
        self.export = export

        self.anchors /= self.stride.float().view(-1, 1, 1)
        self._initialize_biases()

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv

            if not self.export:
                bs, _, ny, nx = x[i].shape  # x(bs,255,w,w) to x(bs,3,w,w,85)
                x[i]=x[i].view(bs, self.na, self.no, ny*nx).permute(0, 1, 3, 2).view(bs, self.na, ny, nx, self.no).contiguous()
            else:
                x[i] = torch.sigmoid(x[i])
            if not self.training and not self.export:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()

                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i] # ADD ME BACK IF REMOVED DIV IN LOSS* self.stride[i] # wh

                z.append(y.view(bs, -1, self.no))
        return x if self.training or self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def _initialize_biases(self, cf=None):
        for mi, s in zip(self.m, self.stride):  # from
            b = mi.bias.view(self.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (self.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)