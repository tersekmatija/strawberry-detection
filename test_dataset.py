import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.ops.boxes import box_convert


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

transforms = A.Compose([
    A.RandomCropToAspect(cfg.img_shape),
    A.Resize(cfg.img_shape)
])

trainloader = get_loader(cfg.dataset, "train", cfg.dataset_dir, 1, transforms=transforms)

writer = SummaryWriter("runs/test_dataset")


with tqdm(total=len(trainloader.dataset), desc ='Demo', unit='chunks') as prog_bar:
    for i, data in enumerate(trainloader):
        inputs, labels = data

        
        img_out = inputs.detach().cpu()[0]
        boxes_out = labels[1].detach().cpu()
        seg_out = labels[0][0].detach().cpu()

        print(boxes_out)
        boxes_out = boxes_out[:, 2:]
        boxes_out = box_convert(boxes_out, "cxcywh", "xyxy")
        print(boxes_out)
        print(img_out.shape)
        boxes_out *= torch.Tensor([img_out.shape[2], img_out.shape[1], img_out.shape[2], img_out.shape[1]])
        boxes_out = boxes_out.type(torch.int)

        img = (img_out * 255).type(torch.uint8)

        print(seg_out.shape)
        seg = (seg_out[1] > 0.5).bool() # 0.5 equal to argmax
        img = draw_segmentation_masks(img, seg, alpha=0.8, colors="blue")

        img = draw_bounding_boxes(img, boxes_out, colors="red")
        writer.add_images("train/images", img.unsqueeze(0), i)


        img_show = img.numpy()
        img_show = img_show.astype(np.uint8).transpose(1,2,0)
        plt.imshow(img_show)
        plt.show()

        prog_bar.set_postfix(**{'run:': cfg.demo_run})
        prog_bar.update(1)