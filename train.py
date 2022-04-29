import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
import os
import random
import numpy as np

from utils.loss import CombinedLoss
from utils.config import load_config
from utils.draw import draw_overlay
from utils.schedulers import CosineAnnealingLR
import utils.augmentations as A
from datasets.loaders import get_loader
from models.model import Model

parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--config', type=str, help="Path to training config", required=True)
args = parser.parse_args()

cfg = load_config(args.config)

batch_size = cfg.batch_size
learning_rate = cfg.learning_rate
epochs=cfg.epochs

writer = SummaryWriter(cfg.save_dir)

transforms = A.Compose([
    A.RandomGaussianBlur(cfg.blur_p, cfg.blur_ks),
    A.RandomHFlip(cfg.flip_p),
    A.RandomRotate(cfg.rotate_p),
    A.RandomCrop(cfg.min_scale),
    A.Resize(cfg.img_shape)
])

transforms_val = A.Resize(cfg.img_shape)

trainloader = get_loader(cfg.dataset, "train", cfg.dataset_dir, cfg.batch_size, transforms=transforms)
valloader = get_loader(cfg.dataset, "val", cfg.dataset_dir, cfg.batch_size, transforms=transforms)


model = Model(cfg.num_classes, cfg.anchors, cfg.strides, cfg.reduction)
model.train()
model.cuda()

criterion = CombinedLoss(cfg.num_classes, model.gr, anchors = cfg.anchors, stride=cfg.strides)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
iters_per_epoch = len(trainloader)
scheduler = CosineAnnealingLR(optimizer, cfg.epochs * iters_per_epoch, eta_min = 0, warmup = cfg.warmup, warmup_iters = cfg.warmup_iters)


n_iter = 0
val_loss_best = torch.inf
for epoch in range(epochs):

    # train
    model.train()
    with tqdm(total=len(trainloader.dataset), desc ='Training - Epoch: '+str(epoch)+"/"+str(epochs), unit='chunks') as prog_bar:
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)

            labels[0] = labels[0].cuda()
            labels[1] = labels[1].cuda()

            lseg, liou, lbox, lobj, lcls = criterion(outputs,labels)
            loss = lseg + liou + lbox + lobj + lcls
            loss.backward()
            scheduler.step(n_iter)

            logs = {
                "train/l_combined":loss.item(),
                "train/l_seg":lseg.item(),
                "train/l_iou":liou.item(),
                "train/l_box":lbox.item(),
                "train/l_obj":lobj.item(),
                "train/l_cls":lcls.item(),
                "train/lr":scheduler.optimizer.param_groups[0]['lr']
            }

            for loss_name, loss_val in logs.items():
                writer.add_scalar(loss_name, loss_val, n_iter)
            n_iter += 1
            
            optimizer.step()
            prog_bar.set_postfix(**{'run:': "model_name",
                                     **{key.split("/")[1] : value for key,value in logs.items()}})
            prog_bar.update(batch_size)
    
    # validate
    model.eval()
    idx_draw = random.sample(range(len(valloader)), min(len(valloader),cfg.val_plot_num))
    losses = []
    imgs = []
    ious = []
    for i, data in tqdm(enumerate(valloader)):
        inputs, labels = data
        inputs = inputs.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)

        labels[0] = labels[0].cuda()
        labels[1] = labels[1].cuda()

        # compute loss
        lseg, liou, lbox, lobj, lcls = criterion([outputs[0], outputs[1][1]],labels)
        loss = lseg + liou + lbox + lobj + lcls
        losses.append(loss.item())
        ious.append(1 - liou.item())

        if i in idx_draw:
            img_out = inputs.detach().cpu()
            boxes_out = outputs[1][0].detach().cpu()
            seg_out = outputs[0].detach().cpu()

            img = draw_overlay(img_out[0], boxes_out[0].unsqueeze(0), seg_out[0][1], show=False)
            imgs.append(img)

    val_loss = np.mean(losses)
    val_iou = np.mean(ious)
    print(f"Val loss: {val_loss}, Val mIoU: {val_iou}")
    writer.add_scalar("val/l_combined", val_loss, epoch)
    writer.add_scalar("val/mIoU", val_iou, epoch)
    writer.add_images("val/images", torch.stack(imgs), epoch)

    torch.save(model.state_dict(), os.path.join(cfg.save_dir, "last.pt"))
    if val_loss < val_loss_best:
        torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best.pt"))
