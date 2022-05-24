from PIL import Image
import os
import torch

import torchvision
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.ops import masks_to_boxes, box_convert

class DemoDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transforms = None):
        
        # define path and split
        self.root = root
        self.split = split        
        
        self.transforms = transforms
        
        # list of images
        self.imgs = list(sorted(os.listdir(root)))
        
        
    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, idx):
        
        # image path
        img_path = os.path.join(self.root, self.imgs[idx])
        
        # read
        img = read_image(img_path)
        
        # to float
        img = F.convert_image_dtype(img, dtype=torch.float)
        print(img.shape)
        mask = torch.zeros((1, img.shape[1], img.shape[2]), dtype=torch.float)
        
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
                
        sem = torch.stack([1-mask, mask]) # bg, cls1
        
        # get boxes and normalize them - use ones as only one class
        boxes = torch.ones((0, 6))
        
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