from PIL import Image
import os
import torch
import numpy as np
import warnings
import json
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.ops import masks_to_boxes, box_convert

import utils.augmentations as A

class PeopleDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transforms = None):
        
        # define path and split
        self.root = root
        self.split = split        
        if self.split not in ["train", "val", "test"]: raise ValueErorr(f"Unrecognized split: {self.split}")
        
        transforms_new = None
        if transforms is not None:
            transforms_new = []
            for t in transforms:
                if not isinstance(t, A.Resize):
                    warnings.warn(f"Transform {t} not supported. Dropping.")
                transforms_new.append(t)
        self.transforms = transforms_new

        # check if "splits.json" exists
        if not os.path.exists(os.path.join(self.root, "splits.json")): raise RuntimeError(f"Not found splits.json in {self.root}!")
        
        f = open(os.path.join(self.root, "splits.json"))
        # list of images
        splits = json.load(f)
        self.path_pairs = splits[self.split]
        
        
    def __len__(self):
        return len(self.path_pairs)
        
    def __getitem__(self, idx):
        
        # image path
        pair = self.path_pairs[idx]
        img_path = os.path.join(self.root, pair[0])
        mask_path = os.path.join(self.root, pair[1])
        boxs_path = os.path.join(self.root, pair[2])
        
        # read
        #print(img_path)
        img = read_image(img_path)
        mask = read_image(mask_path, ImageReadMode.RGB)
        boxs = torch.Tensor(np.genfromtxt(boxs_path,delimiter=' '))
        
        # to float
        img = F.convert_image_dtype(img, dtype=torch.float)
        mask = F.convert_image_dtype(mask, dtype=torch.float)
        
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        
        # get unique ids and remove bg
        sem = (mask == 0).all(dim = 0).int()
        #print(sem.shape)
        sem = torch.stack([sem, 1-sem]) # bg, cls1
        #print(sem.shape)
        
        # get boxes and normalize them - use ones as only one class
        boxes = torch.ones((boxs.shape[0], 6))
        if len(boxes) > 0:
            boxes[:, 1:] = boxs
        
        return img, [sem, boxes]

    @staticmethod
    def collate_fn(batch):
        img, label = zip(*batch)
        label_box, label_seg = [], []
        for i, l in enumerate(label):
            l_seg, l_box = l
            l_box[:, 0] = i  # add target image index for build_targets()
            label_box.append(l_box)
            label_seg.append(l_seg)
            #print(l_seg)
        imgs = torch.stack(img, 0)
        boxs = torch.cat(label_box, 0)
        segs = torch.stack(label_seg, 0)
        return imgs, [segs, boxs]