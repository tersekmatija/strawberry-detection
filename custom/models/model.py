import torch
import torch.nn as nn
import torchvision

from models.backbone import *
from models.heads import *
from models.decoder import *

class Model(nn.Module):

    def __init__(self, num_classes):
        super(Model, self).__init__()

        self.nc = num_classes
        self.gr = 1.0

        self.backbone = YoloPC3()
        self.decoder = YoloPC3Decoder()
        self.seg_head = SegmentationHead(num_classes = self.nc)
        self.det_head = DetectionHead(num_classes = self.nc)


    def forward(self, x):
        x, c3_3, c3_2, c3_1 = self.backbone(x)
        x, c2, c1 = self.decoder(x, c3_3, c3_2)

        seg = self.seg_head(x)
        det = self.det_head(x, c2, c1)

        return seg, det