import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
import os
import random
import math
import numpy as np

from utils.loss import CombinedLoss
from utils.config import load_config
from utils.draw import draw_overlay
from utils.schedulers import CosineAnnealingLR, LinearLR
from torch.optim.lr_scheduler import LambdaLR#CosineAnnealingLR
import utils.augmentations as A
from datasets.loaders import get_loader
from models.model import Model

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--config', type=str, help="Path to training config", required=True)
args = parser.parse_args()

np.testing.suppress_warnings()

cfg = load_config(args.config)

batch_size = cfg.batch_size
learning_rate = cfg.learning_rate
epochs=cfg.epochs

transforms = A.Compose([
    A.RandomGaussianBlur(cfg.blur_p, cfg.blur_ks),
    A.RandomHFlip(cfg.flip_p),
    A.RandomRotate(cfg.rotate_p),
    A.RandomCropToAspect(cfg.img_shape),
    A.AutoContrast(),
    A.ColorJitter(),
    A.Occlusion(),
    #A.RandomCrop(cfg.min_scale),
    A.Resize(cfg.img_shape)
])

transforms_val = A.Compose([A.RandomCropToAspect(cfg.img_shape),A.Resize(cfg.img_shape)])

trainloader = get_loader(cfg.dataset, "train", cfg.dataset_dir, cfg.batch_size, transforms=transforms, num_workers=cfg.num_workers, pin_memory=True, shuffle=True)
valloader = get_loader(cfg.dataset, "val", cfg.dataset_dir, 1, transforms=transforms_val, num_workers=cfg.num_workers, pin_memory=True)


model = Model(cfg.num_classes, cfg.anchors, cfg.strides, cfg.reduction)
model.train()
model.cuda()

if cfg.pretrained is not None:
    #state_dict = torch.load(cfg.pretrained, map_location="cpu")
    #model.load_state_dict(state_dict)
    pretrained_dict = torch.load(cfg.pretrained, map_location="cpu")
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == pretrained_dict[k].shape}
    print(f"Following weight not loaded: {[k for k in model_dict.keys() if k not in pretrained_dict]}")
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)

if not cfg.backbone:
    print("Freezing backbone.")
    for k, v in model.backbone.named_parameters():
        v.requires_grad = False

if not cfg.decoder:
    print("Freezing decoder.")
    for k, v in model.decoder.named_parameters():
        v.requires_grad = False

if not cfg.head_seg:
    print("Freezing seg head.")
    for k, v in model.seg_head.named_parameters():
        v.requires_grad = False
        cfg.w_seg, cfg.w_iou = 0, 0

if not cfg.head_det:
    print("Freezing det head.")
    for k, v in model.det_head.named_parameters():
        v.requires_grad = False
        cfg.w_cls, cfg.w_obj, cfg.w_box = 0, 0, 0

criterion = CombinedLoss(cfg, cfg.num_classes, model.gr, anchors = cfg.anchors, stride=cfg.strides, device = next(model.parameters()).device)

if cfg.optimizer.lower() == "adam":
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(cfg.momentum, 0.999), weight_decay = cfg.weight_decay)
elif cfg.optimizer.lower() == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=cfg.momentum, nesterov=True)
else:
    raise RuntimeError(f"Optimizer {cfg.optimizer} not supported.")

iters_per_epoch = len(trainloader)

#scheduler = CosineAnnealingLR(optimizer,
#    iters_per_epoch * cfg.epochs, eta_min = 0, warmup = cfg.warmup, warmup_iters = cfg.warmup_iters)
scheduler = LinearLR(optimizer,
    iters_per_epoch * cfg.epochs, eta_min = 0, warmup = cfg.warmup, warmup_iters = cfg.warmup_iters)#scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

writer = SummaryWriter(cfg.save_dir)

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

    with torch.no_grad():
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

                img = draw_overlay(img_out[0], boxes_out[0].unsqueeze(0), seg_out[0][1], cfg.thr_conf, cfg.thr_iou, show=False)
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
