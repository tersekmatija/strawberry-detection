import torch
import torch.nn as nn
import torchvision

from models.layers import C3, Conv

class YoloPC3Decoder(nn.Module):

    def __init__(self, ch = 512):
        super(YoloPC3Decoder, self).__init__()
        self.c3u_1 = C3(ch, ch, shortcut = False)
        self.conv_1 = Conv(ch, ch // 2, 1, 1)
        self.up1 = nn.UpsamplingBilinear2d(scale_factor = 2)

        self.c3u_2 = C3(ch, ch//2, shortcut = False)
        self.conv_2 = Conv(ch//2, ch//4, 1, 1)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor = 2)

    def forward(self, x, x_prev1, x_prev2):
        x = self.c3u_1(x)
        c1 = self.conv_1(x)
        x = self.up1(c1)

        x = torch.cat([x, x_prev1], dim = 1)
        
        x = self.c3u_2(x)
        c2 = self.conv_2(x)
        x = self.up2(c2)

        x = torch.cat([x, x_prev2], dim = 1)

        return x, c2, c1