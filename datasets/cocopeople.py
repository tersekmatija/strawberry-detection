from PIL import Image
import os
import torch
import json

import torchvision
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.ops import masks_to_boxes, box_convert

class COCOPeopleDataset(torch.utils.data.Dataset):
    # File scripts/download_coco.py uses FiftyOne to download the data and process the person
    # into instance segmentation masks. We split train into train and val and use validation
    # as a test split here.

    def __init__(self, root, split, transforms = None):
        
        # define path and split
        self.root = root
        self.split = split        
        if self.split not in ["train", "val", "test"]: raise ValueErorr(f"Unrecognized split: {self.split}")

        if not os.path.exists(os.path.join(self.root, "splits_person.json")):
            raise RuntimeError("Dataset not loaded correctly. Please use scripts/download_coco.py!")

        f = open(os.path.join(self.root, "splits_person.json"))
        imgs = json.load(f)

        self.transforms = transforms
        
        # list of images
        self.imgs = imgs[self.split]

        self.data_path = os.path.join(self.root, "validation", "data") if self.split == "test" else os.path.join(self.root, "train", "data")
        
        
        
    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, idx):
        
        fn = self.imgs[idx]
        fn_mask = f"{self.imgs[idx].split('.')[0]}_label.png"

        # image path
        img_path = os.path.join(self.data_path, fn)
        mask_path = os.path.join(self.data_path, fn_mask)
        
        # read
        img = read_image(img_path)
        mask = read_image(mask_path)
        
        # to float
        img = F.convert_image_dtype(img, dtype=torch.float)
        mask = F.convert_image_dtype(mask, dtype=torch.float)
        
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