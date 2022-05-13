import torch
import torch.nn as nn
import torchvision

from models.layers import Conv, SPP, C3, SPPF

class ResNet18(nn.Module):

    def __init__(self, pretrained = True):

        super(self, ResNet18).__init__()
        self.bb = torchvision.model.resnet18(pretrained = pretrained)

    def forward(self, x):

        x = self.bb.conv1(x)
        x = self.bb.bn1(x)
        x = self.bb.relu(x)
        x = self.bb.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        return


class YoloPC3(nn.Module):

    def __init__(self, ch = 32):
        super(YoloPC3, self).__init__()

        self.conv_1 = Conv(3, ch, 6, 2, 2)  # inspired by YoloV5 instead of Focus
        self.conv_2 = Conv(ch, ch*2, 3, 2, 1)
        self.c3_1 = C3(ch*2, ch*2, n = 1)
        self.conv_3 = Conv(ch*2, ch*4, 3, 2, 1)
        self.c3_2 = C3(ch*4, ch*4, n = 2)
        self.conv_4 = Conv(ch*4, ch*8, 3, 2, 1)
        self.c3_3 = C3(ch*8, ch*8, n = 3)
        self.conv_5 = Conv(ch*8, ch*16, 3, 2, 1)
        self.c3_4 = C3(ch*16, ch*16, n = 1)

        self.spp = SPPF(ch*16, ch*16)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        c3_1 = self.c3_1(x)
        x = self.conv_3(c3_1)
        c3_2 = self.c3_2(x)
        x = self.conv_4(c3_2)
        c3_3 = self.c3_3(x)
        x = self.conv_5(c3_3)
        x = self.c3_4(x)
        x = self.spp(x)

        return x, c3_3, c3_2, c3_1


        
