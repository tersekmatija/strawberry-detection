import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes

import os
import matplotlib.pyplot as plt
import numpy as np

from utils.config import load_config
import utils.augmentations as A
from datasets.strawberrydi import StrawDIDataset
from models.model import Model

parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--config', type=str, help="Path to training config", required=True)
args = parser.parse_args()

cfg = load_config(args.config)

batch_size = 1

transforms = A.Compose([
    A.Resize(cfg.img_shape)
])

train_dataset = StrawDIDataset(split="test", root=cfg.dataset_dir, transforms=transforms)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, collate_fn=StrawDIDataset.collate_fn)

model = Model(cfg.num_classes, cfg.anchors, cfg.strides)

if cfg.demo_weights is None:
    raise RuntimeError("Demo run not set!")
model.load_state_dict(torch.load(cfg.demo_weights, map_location="cpu"))
model.cuda()
model.eval()

with tqdm(total=len(trainloader.dataset), desc ='Demo', unit='chunks') as prog_bar:
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs = inputs.cuda()

        outputs = model(inputs)

        labels[0] = labels[0].cuda()
        labels[1] = labels[1].cuda()

        img = (inputs.detach().cpu()[0] * 255).type(torch.uint8)
        seg = torch.argmax(outputs[0].detach().cpu()[0], dim = 0).bool()
        img = draw_segmentation_masks(img, seg, alpha=0.8, colors="blue")
        img = img.numpy()
        img = img.astype(np.uint8).transpose(1,2,0)
        plt.imshow(img)
        plt.show()

        prog_bar.set_postfix(**{'run:': cfg.demo_run})
        prog_bar.update(batch_size)