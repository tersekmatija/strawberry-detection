import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter

import os
import matplotlib.pyplot as plt
import numpy as np

from utils.draw import draw_overlay
from utils.config import load_config
from utils.boxutils import non_max_suppression
import utils.augmentations as A
from datasets.strawberrydi import StrawDIDataset
from models.model import Model
from datasets.loaders import get_loader

parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--config', type=str, help="Path to training config", required=True)
args = parser.parse_args()

cfg = load_config(args.config)

batch_size = 1

transforms = A.Compose([
    A.Resize(cfg.img_shape)
])

trainloader = get_loader(cfg.dataset, "test", cfg.dataset_dir, 1, transforms=transforms)

model = Model(cfg.num_classes, cfg.anchors, cfg.strides, cfg.reduction)

if cfg.demo_weights is None:
    raise RuntimeError("Demo run not set!")
state_dict = torch.load(cfg.demo_weights, map_location="cpu")
#state_dict.pop("det_head.detect.anchor_grid")
#state_dict.pop("det_head.detect.anchors")
#print(state_dict.keys())

model.load_state_dict(state_dict)
model.cuda()
model.eval()

with tqdm(total=len(trainloader.dataset), desc ='Demo', unit='chunks') as prog_bar:
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs = inputs.cuda()

        outputs = model(inputs)

        img_out = inputs.detach().cpu()
        boxes_out = outputs[1][0].detach().cpu()
        seg_out = outputs[0].detach().cpu()

        #print(boxes_out)
        for idx in range(batch_size):
            draw_overlay(img_out[idx], boxes_out[idx].unsqueeze(0), seg_out[idx][1], cfg.thr_conf, cfg.thr_iou)


        prog_bar.set_postfix(**{'run:': cfg.demo_run})
        prog_bar.update(batch_size)