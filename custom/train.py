import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
import os

from utils.loss import CombinedLoss
from utils.config import load_config
from utils.schedulers import CosineAnnealingLR
import utils.augmentations as A
from datasets.strawberrydi import StrawDIDataset
from models.model import Model

# TODO: validation, evaluation, plotting, learning rate scheduler

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

train_dataset = StrawDIDataset(split="train", root=cfg.dataset_dir, transforms=transforms)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, collate_fn=StrawDIDataset.collate_fn)

model = Model(cfg.num_classes, cfg.anchors, cfg.strides)
model.train()
model.cuda()

criterion = CombinedLoss(cfg.num_classes, model.gr, anchors = cfg.anchors)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
iters_per_epoch = len(trainloader)
scheduler = CosineAnnealingLR(optimizer, cfg.epochs * iters_per_epoch, eta_min = 0, warmup = cfg.warmup, warmup_iters = cfg.warmup_iters)


n_iter = 0
for epoch in range(epochs):
    with tqdm(total=len(trainloader.dataset), desc ='Training - Epoch: '+str(epoch)+"/"+str(epochs), unit='chunks') as prog_bar:
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)

            #print(labels[0].shape)
            #labels[0] = torch.nn.functional.interpolate(labels[0], scale_factor=0.25, mode='nearest')
            labels[0] = labels[0].cuda()
            labels[1] = labels[1].cuda()
            #print(outputs[1].shape)
            #print(labels[1].shape)


            lseg, liou, lbox, lobj, lcls = criterion(outputs,labels)
            loss = lseg + liou + lbox + lobj + lcls
            loss.backward()
            scheduler.step(n_iter)

            #torch.nn.utils.clip_grad_norm_(model.parameters(),
            #                          max_norm=10.0)
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
    torch.save(model.state_dict(), os.path.join(cfg.save_dir, "last.pt"))