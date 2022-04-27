import torch
import torch.nn as nn
import torchvision

from models.layers import Conv, C3, Detect

class SegmentationHead(nn.Module):

    def __init__(self, ch = 256, num_classes = 1):
        super(SegmentationHead, self).__init__()
        self.conv_1 = Conv(ch, ch//2, 3, 1, p = 1)
        self.up_1 = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.c3_1 = C3(ch//2, ch//4, shortcut = False)

        self.conv_2 = Conv(ch//4, ch//8, 3, 1, p = 1)
        self.up_2 = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.conv_3 = Conv(ch//8, ch//16, 3, 1, p = 1)
        self.c3_2 = C3(ch//16, ch//32, shortcut = False)
        self.up_3 = nn.UpsamplingBilinear2d(scale_factor = 2)

        self.conv_4 = Conv(ch//32, num_classes+1, 3, 1, p = 1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.up_1(x)
        x = self.c3_1(x)
        x = self.conv_2(x)
        x = self.up_2(x)
        x = self.conv_3(x)
        x = self.c3_2(x)
        x = self.up_3(x)
        x = self.conv_4(x)
        #print(x.shape)

        return x

class DetectionHead(nn.Module):

    def __init__(self, num_classes, anchors, strides, ch = 256):
        super(DetectionHead, self).__init__()

        self.num_classes = anchors
        self.anchors = anchors
        self.stides = strides

        self.c3_1 = C3(ch, ch//2, shortcut=False)
        self.conv_1 = Conv(ch//2, ch//2, 3, 2, p=1)
        
        self.c3_2 = C3(ch, ch, shortcut=False)
        self.conv_2 = Conv(ch, ch, 3, 2, p=1)

        self.c3_3 = C3(ch*2, ch*2, shortcut=False)
        self.detect = Detect(num_classes, anchors, [128, 256, 512], strides)

    def forward(self, x, x_prev1, x_prev2):

        c3_1 = self.c3_1(x)
        x = self.conv_1(c3_1)

        x= torch.cat([x, x_prev1], dim = 1)
        c3_2 = self.c3_2(x)
        x = self.conv_2(c3_2)

        x = torch.cat([x, x_prev2], dim = 1)
        c3_3 = self.c3_3(x)
        #print(c3_3.shape)
        x = self.detect([c3_1, c3_2, c3_3])

        return x