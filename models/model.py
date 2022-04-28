import torch
import torch.nn as nn
import torchvision

from models.backbone import *
from models.heads import *
from models.decoder import *

class Model(nn.Module):

    def __init__(self, num_classes, anchors, strides, reduction = 1):
        super(Model, self).__init__()

        ch_ = int(32 / reduction) if isinstance(reduction, int) and reduction >= 1 else 32

        self.nc = num_classes
        self.gr = 1.0
        self.anchors = anchors
        self.strides = strides

        self.backbone = YoloPC3(ch_)
        self.decoder = YoloPC3Decoder(ch=ch_*16)
        self.seg_head = SegmentationHead(num_classes = self.nc, ch=ch_*8)
        self.det_head = DetectionHead(num_classes=self.nc, anchors=self.anchors, strides=self.strides, ch=ch_*8)


    def forward(self, x):
        x, c3_3, c3_2, c3_1 = self.backbone(x)
        x, c2, c1 = self.decoder(x, c3_3, c3_2)

        seg = self.seg_head(x)
        det = self.det_head(x, c2, c1)

        return seg, det