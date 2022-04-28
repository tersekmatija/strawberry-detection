from PIL import Image
import os
import torch

import torchvision
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.ops import masks_to_boxes, box_convert

class StrawDIDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transforms = None):
        
        # define path and split
        self.root = root
        self.split = split        
        if self.split not in ["train", "val", "test"]: raise ValueErorr(f"Unrecognized split: {self.split}")
        
        self.transforms = transforms
        
        # list of images
        self.imgs = list(sorted(os.listdir(os.path.join(root, split, "img"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, split, "label"))))
        
        
    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, idx):
        
        # image path
        img_path = os.path.join(self.root, self.split, "img", self.imgs[idx])
        mask_path = os.path.join(self.root, self.split, "label", self.masks[idx])
        
        # read
        img = read_image(img_path)
        mask = read_image(mask_path)
        
        # to float
        img = F.convert_image_dtype(img, dtype=torch.float)
        mask = F.convert_image_dtype(mask, dtype=torch.float)
        
        # TODO: add transform
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
 
        
        # get unique ids and remove bg
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]

        # to bool masks
        masks = mask == obj_ids[:, None, None]
        
        # get semantic segmentation mask
        if masks.shape[0] == 0: masks = torch.zeros(1, mask.shape[1], mask.shape[2])
        sem = torch.max(masks, dim = 0)[0].float()
        sem = torch.stack([1-sem, sem]) # bg, cls1
        #print(sem.shape)
        
        # get boxes and normalize them - use ones as only one class
        boxes = torch.ones((len(obj_ids), 6))
        if len(obj_ids) > 0:
            boxes[:, 2:] = masks_to_boxes(masks)
            boxes[:, 2:] = box_convert(boxes[:, 2:], "xyxy", "cxcywh")
            boxes[:, [3, 5]] /= img.shape[1]  # height
            boxes[:, [2, 4]] /= img.shape[2]
        #print(boxes[0])
        
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