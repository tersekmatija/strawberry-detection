from models.model import Model
from models.layers import C3
from utils.loss import *

import torch


inp = torch.rand(16, 3, 704, 768)
model = Model(1)
model.eval()
out =  model(inp)
print(out[0].shape)
print(out[1][0].shape)
print(out[1][1][0].shape)
print(out[1][1][1].shape)
print(out[1][1][2].shape)

loss1 = SegmentationLoss()
print(loss1(out[0], out[0]))

loss2 = DetectionLoss(model.nc, model.gr, anchors = [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]])
targets = torch.rand(7, 6)
targets[:, 0] = 0
targets[:, 1] = 0
print(loss2(out[1][1], targets))

loss3 = CombinedLoss(model.nc, model.gr, anchors = [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]])
targets = (out[0], targets)
print(loss3(out, targets))